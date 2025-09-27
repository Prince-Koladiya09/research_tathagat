import os
import logging
from ..configs.base_config import LOG_DIR
import re

class Logger:
    def __init__(self, name: str = "My_Lib_Logger", info_file: str = "training_info.log", error_file: str = "training_error.log"):
        self.logger = logging.getLogger(name)
        
        if not self.logger.handlers:
            self.logger.setLevel(logging.DEBUG)
            
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

            info_path = self._get_name(info_file)
            info_handler = logging.FileHandler(info_path)
            info_handler.setLevel(logging.INFO)
            info_handler.setFormatter(formatter)
            
            error_path = self._get_name(error_file)
            error_handler = logging.FileHandler(error_path)
            error_handler.setLevel(logging.ERROR)
            error_handler.setFormatter(formatter)
            
            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)

            self.logger.addHandler(info_handler)
            self.logger.addHandler(error_handler)
            self.logger.addHandler(console_handler)

    def _get_name(self, name : str) -> str :
        if os.path.exists(os.path.join(LOG_DIR, name)) :
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