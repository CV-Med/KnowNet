import os
import sys
import logging
from tqdm import tqdm


class Logger:
    def __init__(self, log_dir: str = "./logs", name: str = "train"):
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.handlers.clear()
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(file_handler)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter("%(asctime)s | %(message)s"))
        self.logger.addHandler(stream_handler)

    def info(self, message: str):
        self.logger.info(message)

    def close(self):
        for handler in self.logger.handlers:
            handler.close()
