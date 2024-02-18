from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

from logging.handlers import TimedRotatingFileHandler
import logging
import sys
import os


class Logger:
    def __init__(self, config) -> None:
        self.config = config
        self.FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def get_console_handler(self):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(self.FORMATTER)
        return console_handler
    
    def get_file_handler(self):
        LOG_FILE = self.config['log_file']
        os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
        file_handler = TimedRotatingFileHandler(LOG_FILE, when='midnight')
        file_handler.setFormatter(self.FORMATTER)
        return file_handler
    
    def get_logger(self, logger_name):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        
        if not logger.hasHandlers():
            logger.addHandler(self.get_console_handler())
            logger.addHandler(self.get_file_handler())

        logger.propagate = False
        return logger