import pandas as pd
from sklearn.model_selection import StratifiedKFold
import numpy as np
from typing import List, Union, Callable

from ..Models import get_model
from ..Models.base_model import Base_Model
from ..loggers import Logger
from ..configs.base_config import Global_Config

def default_fine_tune_strategy(model: Base_Model, fine_tune_layers: int = 20):
    """
    A simple default strategy for fine-tuning.
    Change the startegy here as required.
    """
    # if hasattr(model, 'unfreeze_later_n'):
    #     model.freeze_all()
    #     model.unfreeze_later_n(fine_tune_layers)
    # elif hasattr(model, 'unfreeze_last_n_blocks'):
    #     model.unfreeze_last_n_blocks(2) # Default to unfreezing last 2 blocks for transformers
    # else:
    #     model.unfreeze_all()
    
    # model.compile()
    pass

class Cross_Validator:
    def __init__(self, model_names: Union[str, List[str]], n_splits: int = 5,
                 config: Global_Config = None, logger: Logger = None):
        self.model_names = [model_names] if isinstance(model_names, str) else model_names
        self.n_splits = n_splits
        self.config = config
        self.logger = logger if logger else Logger("Cross_Validation_Logger", "cv_info.log", "cv_error.log")
        self.results = pd.DataFrame()

    def run(self, 
            X: np.ndarray, y: np.ndarray,
            fine_tune_strategy: Callable[[Base_Model], None] = default_fine_tune_strategy,
            epochs: int = 10, batch_size: int = 32):
                
        self.logger.info(f"Starting {self.n_splits}-fold cross-validation for models: {self.model_names}")
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.config.training.seed if self.config else 42)

        for model_name in self.model_names:
            self.logger.info(f"--- Validating model: {model_name} ---")
            fold_results = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
                self.logger.info(f"--- Fold {fold + 1}/{self.n_splits} ---")
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                model = get_model(model_name, name=f"{model_name}_fold{fold+1}", config=self.config)
                
                fine_tune_strategy(model)

                model.fit(
                    train_data=(X_train, y_train),
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size,
                    verbose=0
                )
                
                eval_metrics = model.evaluate(data=(X_val, y_val), batch_size=batch_size, verbose=0)
                metric_dict = {name: val for name, val in zip(model.model.metrics_names, eval_metrics)}
                fold_results.append(metric_dict)
                self.logger.info(f"Fold {fold + 1} results: {metric_dict}")
            
            self._aggregate_results(model_name, fold_results)

        return self.results

    def _aggregate_results(self, model_name: str, fold_results: List[dict]):
        df_folds = pd.DataFrame(fold_results)
        agg_metrics = {}
        for col in df_folds.columns:
            agg_metrics[f'mean_{col}'] = df_folds[col].mean()
            agg_metrics[f'std_{col}'] = df_folds[col].std()

        self.logger.info(f"Aggregated results for {model_name}: {agg_metrics}")
        model_summary = pd.DataFrame([agg_metrics], index=[model_name])
        self.results = pd.concat([self.results, model_summary])

    def save_results(self, filepath: str):
        try:
            if filepath.endswith('.csv'):
                self.results.to_csv(filepath)
            elif filepath.endswith('.pkl'):
                self.results.to_pickle(filepath)
            else:
                raise ValueError("Filepath must end with .csv or .pkl")
            self.logger.info(f"Cross-validation results saved to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save results to {filepath}: {e}")