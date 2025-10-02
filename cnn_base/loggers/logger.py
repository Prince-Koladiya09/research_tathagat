import os
import logging
from ..configs.base_config import LOG_DIR
import re
import time

class Logger:
    def __init__(self, name: str = "My_Lib_Logger", info_file: str = "training_info.log", error_file: str = "training_error.log", new_file : bool = True, timestamped_name : bool = True):
        self.logger = logging.getLogger(name)
        self.new_file = new_file
        self.timestamped_name = timestamped_name
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
            console_formatter = logging.Formatter("=====%(name)s=====\n%(message)s")

            info_path = self._get_name(info_file)
            info_handler = logging.FileHandler(info_path, encoding = "utf-8")
            info_handler.setLevel(logging.DEBUG)
            # info_handler.setLevel(logging.INFO)
            info_handler.setFormatter(formatter)
            
            error_path = self._get_name(error_file)
            error_handler = logging.FileHandler(error_path)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(console_formatter)

            self.logger.addHandler(info_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)

    def _get_name(self, name : str) -> str :
        if self.timestamped_name :
            # If you want log file naming patterns with timestamps
            # use this code below
            return os.path.join(LOG_DIR, f"{name}__{time.strftime("%d_%m_%Y__%H_%M_%S")}")
        if os.path.exists(os.path.join(LOG_DIR, name)) and self.new_file :
            # If you want log file naming patterns without timestamps
            # and instead want to use names like {file (1).log} and {file (2).log}
            # for duplicate names, uncomment below code
            pattern = r"[a-zA-Z0-9-_.]+ \([0-9]+\)"

            if re.fullmatch(pattern, name) :
                sub_pattern = "([0-9]+)"
                splitted = name.split()
                splitted[-1] = re.sub(sub_pattern, lambda m : str(int(m.group(1)) + 1), splitted[-1])
                name = " ".join(splitted)
            else :
                name += " (1)"
            return self._get_name(name)
        else :
            return os.path.join(LOG_DIR, name)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def warning(self, message: str) -> None:
        self.logger.warning(message)

    def error(self, message: str) -> None:
        self.logger.error(message, exc_info=True)
    
    def debug(self, message: str) -> None:
        self.logger.debug(message)