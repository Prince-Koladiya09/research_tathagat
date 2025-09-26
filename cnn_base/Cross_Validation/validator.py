import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from ..Models.base_model import Model
from ..loggers import Logger
from ..config import CV_DIR
import gc
import os

class Cross_Validator:
    def __init__(self, k_folds: int = 5, random_state: int = 42):
        self.k_folds = k_folds
        self.random_state = random_state
        self.logger = Logger("Cross_Validation_Logger", "cv_info.log", "cv_error.log")
        self.results = None

    def run(self, model_names: list, X: np.ndarray, y: np.ndarray, file_name : str = 'results.csv') -> pd.DataFrame :
        skf = StratifiedKFold(n_splits=self.k_folds, shuffle=True, random_state=self.random_state)
        results_data = []

        self.logger.info(f"Starting {self.k_folds}-fold cross-validation for models: {model_names}")

        for model_name in model_names:
            self.logger.info(f"--- Processing Model: {model_name} ---")
            fold_scores = []
            
            for fold, (train_index, val_index) in enumerate(skf.split(X, y)):
                self.logger.info(f"--- Fold {fold+1}/{self.k_folds} ---")
                
                X_train, X_val = X[train_index], X[val_index]
                y_train, y_val = y[train_index], y[val_index]
                
                model_wrapper = Model()
                model_wrapper.get_base_model(name=model_name)
                # model_wrapper.add_custom_layers()
                
                # if model_wrapper.is_transformer:
                #     model_wrapper.base_model.trainable = True
                # else:
                #     model_wrapper.unfreeze_later_N(10)

                model_wrapper.compile()
                model_wrapper.fit(
                    (X_train, y_train),
                    validation_data=(X_val, y_val),
                    epochs=model_wrapper.config.epochs,
                )

                scores = model_wrapper.model.evaluate(X_val, y_val, verbose=0)
                score_dict = {metric.name: score for metric, score in zip(model_wrapper.model.metrics, scores)}
                fold_scores.append(score_dict)
                
                del model_wrapper, X_train, X_val, y_train, y_val
                gc.collect()

            avg_scores = pd.DataFrame(fold_scores).mean().to_dict()
            std_scores = pd.DataFrame(fold_scores).std().to_dict()
            
            model_results = {'model': model_name}
            for metric in avg_scores:
                model_results[f'avg_{metric}'] = avg_scores[metric]
                model_results[f'std_{metric}'] = std_scores[metric]
            results_data.append(model_results)

        self.results = pd.DataFrame(results_data).set_index('model')
        self.logger.info("Cross-validation finished.")
        print("\n--- Cross-Validation Results ---")
        print(self.results)

        self._save_results(file_name)
        return self.results

    def _save_results(self, filepath: str):
        path = os.path.join(CV_DIR, filepath)
        self.results.to_csv(path)
        self.logger.info(f"Results saved to {path}")