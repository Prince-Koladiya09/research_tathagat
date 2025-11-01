import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import List, Union, Callable, Tuple
import os
import traceback
import keras
import gc

from cnn_base.Models import get_model, get_all_models
from cnn_base.Models.base_model import Base_Model
from cnn_base.utils import Visualizer
from cnn_base.loggers import Logger
from cnn_base.configs.base_config import Global_Config, RESULTS_DIR

def cnn_fine_tune_strategy(model: Base_Model, fine_tune_layers: int = 20):
    """Fine-tuning strategy for CNN models"""
    try:
        if hasattr(model, 'unfreeze_later_n'):
            model.unfreeze_later_n(fine_tune_layers)
        model.compile()
        return model
    except Exception as e:
        model.logger.error(f"Error in CNN fine-tuning: {e}")
        return model

def transformer_fine_tune_strategy(model: Base_Model, fine_tune_layers: int = 2):
    """Fine-tuning strategy for Transformer models"""
    try:
        if hasattr(model, 'unfreeze_last_n_blocks'):
            model.unfreeze_last_n_blocks(fine_tune_layers)
        elif hasattr(model, 'freeze_patch_embeddings'):
            model.freeze_patch_embeddings()
        model.compile()
        return model
    except Exception as e:
        model.logger.error(f"Error in Transformer fine-tuning: {e}")
        return model

def get_fine_tune_strategy(model_type: str):
    """Get appropriate fine-tuning strategy based on model type"""
    if 'transformer' in model_type.lower() or 'vit' in model_type.lower():
        return transformer_fine_tune_strategy
    else:
        return cnn_fine_tune_strategy

class Cross_Validator:
    def __init__(self, model_names: Union[str, List[str]], n_splits: int = 5,
                 logger: Logger = None, class_names: List[str] = None,
                 create_xai_plots: bool = True, create_embedding_plots: bool = True,
                 update_config_dict : dict[str, dict] = None):
        """
        Example for update_config_dict :
        ```python
        update_config_dict = {
            "training" : {
                "epochs" : 10
            }
        }
        ```
        """
        self.n_splits = n_splits
        self.logger = logger if logger else Logger("Cross_Validation_Logger", "cv_info.log", "cv_error.log")
        self.output = RESULTS_DIR
        self.class_names = class_names
        self.create_xai_plots = create_xai_plots
        self.create_embedding_plots = create_embedding_plots
        self.create_embedding_plots = create_embedding_plots
        self.update_config_dict = update_config_dict
        self.results = pd.DataFrame()

        if model_names == "all" :
            self.model_names = get_all_models()[0] # temporary testing all CNN models
        self.model_names = [model_names] if isinstance(model_names, str) else model_names
        
        # Initialize visualizer
        self.visualizer = Visualizer(logger=self.logger)

    def run(self, 
            X: np.ndarray, 
            y: np.ndarray,
            fine_tune_strategy: Callable[[Base_Model], None] = None,
            fine_tune_layers: int = None,
            summary : bool = False,
            background_data: np.ndarray = None) -> pd.DataFrame:
                
        self.logger.info(f"Starting {self.n_splits}-fold cross-validation for models: {self.model_names}")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, 
                            random_state=Global_Config().training.seed)
        
        background_data = background_data or X[:min(100, len(X))]

        all_results = []

        for model_name in self.model_names:
            self.logger.info(f"--- Validating model: {model_name} ---")
            fold_results = []

            model_dir = os.path.join(self.output, model_name)
            graphs_dir = os.path.join(model_dir, "graphs")
            os.makedirs(model_dir, exist_ok = True)
            os.makedirs(graphs_dir, exist_ok = True)
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                self.logger.info(f"--- Fold {fold + 1}/{self.n_splits} ---")
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = None
                try:
                    # Get model instance using get_model
                    self.logger.info(f"Initializing model {model_name} using get_model...")
                    model : Base_Model = get_model(model_name)

                    fine_tune_layers = fine_tune_layers or model.config.model.n_layers_to_tune
                    fine_tune_strategy = fine_tune_strategy or cnn_fine_tune_strategy
                    self.logger.info(f"Applying fine-tuning on {fine_tune_layers} layers...")
                    model = fine_tune_strategy(model, fine_tune_layers)

                    if self.update_config_dict :
                        model.update_config(self.update_config_dict)
                    
                    if summary :
                        model.summary()

                    # Train the model
                    self.logger.info(f"Starting training for {model.config.training.epochs} epochs...")
                    history = model.fit(
                        x=X_train,
                        y=y_train,
                        validation_data=(X_val, y_val),
                        verbose=0
                    )
                    
                    # Evaluate the model
                    self.logger.info("Evaluating model...")
                    eval_metrics = model.evaluate(
                        X = X_val,
                        y = y_val,
                        verbose=0
                        )
                    
                    if isinstance(eval_metrics, (list, tuple)):
                        # Get metric names from the compiled model
                        if hasattr(model, 'config') and hasattr(model.config.training, 'metrics'):
                            metric_names = model.config.training.metrics
                        else:
                            # Default metric names
                            metric_names = ['loss'] + [f'metric_{i}' for i in range(len(eval_metrics)-1)]
                        metric_dict = {name: val for name, val in zip(metric_names, eval_metrics)}
                    else:
                        metric_dict = eval_metrics
                    
                    metric_dict['fold'] = fold + 1
                    metric_dict['model'] = model_name
                    fold_results.append(metric_dict)
                    
                    self.logger.debug(f"Fold {fold + 1} metrics: {eval_metrics}")
                    self.logger.info(f"Fold {fold + 1} results: {metric_dict}")

                    self._create_fold_visualizations(model, X_val, y_val, history, model_name, fold + 1, background_data, model_dir)
                    
                except Exception as e:
                    self.logger.error(f"Error in fold {fold + 1} for model {model_name}: {e}")
                    self.logger.error(traceback.format_exc())
                    continue

                finally :
                    if model :
                        del model
                    
                    keras.backend.clear_session()
                    gc.collect()
                    self.logger.info(f"Cleaned up memory after fold {fold + 1}.")

            # Aggregate results for current model
            if fold_results:
                model_agg_results = self._aggregate_results(model_name, fold_results)
                model_results_path = os.path.join(model_dir, f"{self.n_splits}_folds_results.csv")
                model_agg_results.to_csv(model_results_path, index = False)
                all_results.append(model_agg_results)
            else:
                self.logger.warning(f"No successful folds for model {model_name}")

        # Combine all results
        if all_results:
            self.results = pd.concat(all_results, ignore_index=True)
            self.save_results(os.path.join(self.output, f"{self.n_splits}_folds_cross_validation_results.csv"))
        
        return self.results
    
    def _create_fold_visualizations(self, model_wrapper, X_val, y_val, history, model_name, fold_num, background_data, graphs_dir):
        """
        Create visualization plots for a specific fold and save to appropriate directory.
        """
        try:
            # Create directory structure: graphs/model_name/fold_{num}
            fold_dir = os.path.join(graphs_dir, f"fold_{fold_num}")
            os.makedirs(fold_dir, exist_ok=True)
            
            # Use visualizer to create all plots
            self.visualizer.create_cv_plots(
                model_wrapper=model_wrapper,
                X_val=X_val,
                y_val=y_val,
                history=history,
                fold_dir=fold_dir,
                class_names=self.class_names,
                create_xai_plots=self.create_xai_plots,
                create_embedding_plots=self.create_embedding_plots,
                background_data=background_data
            )
            
            self.logger.info(f"Visualizations for {model_name} fold {fold_num} saved to {fold_dir}")
            
        except Exception as e:
            self.logger.error(f"Error creating visualizations for {model_name} fold {fold_num}: {e}")

    def _aggregate_results(self, model_name: str, fold_results: List[dict]) -> pd.DataFrame:
        df_folds = pd.DataFrame(fold_results)

        metric_cols = [col for col in df_folds.columns if col not in ['fold', 'model']]
        
        # Calculate mean and std for each metric
        summary = {'model': model_name}
        for col in metric_cols :
            summary[f'mean_{col}'] = df_folds[col].mean()
            summary[f'std_{col}'] = df_folds[col].std()
            summary[f'min_{col}'] = df_folds[col].min()
            summary[f'max_{col}'] = df_folds[col].max()

        self.logger.info(f"Aggregated results for {model_name}: {summary}")
        return pd.DataFrame([summary])

    def save_results(self, filepath: str = None):
        if filepath is None:
            filepath = os.path.join(self.results_dir, "cross_validation_results.csv")
        
        try:
            if filepath.endswith('.csv'):
                self.results.to_csv(filepath, index=False)
            elif filepath.endswith('.pkl'):
                self.results.to_pickle(filepath)
            else:
                filepath += '.csv'
                self.results.to_csv(filepath, index=False)
                
            self.logger.info(f"Cross-validation results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {filepath}: {e}")

    def get_best_model(self, metric: str = 'mean_accuracy', ascending: bool = False) -> str:
        """Get the best performing model based on specified metric"""
        if self.results.empty:
            self.logger.warning("No results available. Run cross-validation first.")
            return None
        
        if metric not in self.results.columns:
            available_metrics = [col for col in self.results.columns if col.startswith('mean_')]
            self.logger.warning(f"Metric {metric} not found. Available metrics: {available_metrics}")
            metric = available_metrics[0] if available_metrics else 'model'
        
        best_model = self.results.sort_values(by=metric, ascending=ascending).iloc[0]
        return best_model['model']