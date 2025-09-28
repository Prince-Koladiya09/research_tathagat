import pandas as pd
import os
from datetime import datetime

from ..loggers import Logger
from ..configs.base_config import TRACK_DIR

class Experiment_Tracker:
    def __init__(self, tracking_file: str = "experiment_log.csv"):
        self.log_dir = TRACK_DIR
        self.filepath = os.path.join(self.log_dir, tracking_file)
        self.logger = Logger("Experiment_Tracker", "tracking_info.log", "tracking_error.log")

    def log_experiment(self, model_obj, history, model_path: str):
        self.logger.info(f"Logging experiment results to {self.filepath}")
        try:
            config_dict = model_obj.config.model_dump(exclude={'metrics'})
            
            flat_config = pd.json_normalize(config_dict, sep='_').to_dict(orient='records')[0]
            
            final_metrics = {f"final_{k}": v[-1] for k, v in history.history.items()}

            experiment_data = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "model_name": model_obj.name,
                "base_model": model_obj.base_model.name,
                "saved_model_path": model_path,
                **flat_config,
                **final_metrics
            }

            new_log_df = pd.DataFrame([experiment_data])

            if os.path.exists(self.filepath):
                new_log_df.to_csv(self.filepath, mode='a', header=False, index=False)
            else:
                new_log_df.to_csv(self.filepath, mode='w', header=True, index=False)
            
            self.logger.info("Experiment logged successfully.")

        except Exception as e:
            self.logger.error(f"Failed to log experiment: {e}")