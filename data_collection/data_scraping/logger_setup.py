"""
This module sets up a logger for the scraper application.
Functions:
    setup_logger(log_file: str = "scraper.log") -> logging.Logger:
        Configures and returns a logger with handlers for both file and console output.
Variables:
    scraper_logger (logging.Logger): The logger instance configured by setup_logger.
"""

import logging

def setup_logger(log_file="scraper.log"):
    logger = logging.getLogger("ScraperLogger")
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file)
    console_handler = logging.StreamHandler()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

scraper_logger = setup_logger()