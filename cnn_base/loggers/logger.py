import os
import logging
from cnn_base.config import LOG_DIR

class Logger :
    def __init__(self, info_file : str = "training_info.txt", error_file : str = "training_error.log") :
        self.logger = logging.getLogger("LOGGER")
        self.logger.setLevel(logging.DEBUG)

        info_handler = logging.FileHandler(os.path.join(LOG_DIR, info_file))
        error_handler = logging.FileHandler(os.path.join(LOG_DIR, error_file))

        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        info_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)

        info_handler.setLevel(logging.INFO)
        error_handler.setLevel(logging.ERROR)

        if not self.logger.handlers :
            self.logger.addHandler(info_handler)
            self.logger.addHandler(error_handler)
    
    def info(self, message : str) -> None :
        self.logger.info(message)
    
    def error(self, message : str) -> None :
        self.logger.error(message)