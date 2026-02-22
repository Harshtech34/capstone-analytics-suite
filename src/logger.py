# src/logger.py
import logging
import sys
from src.config import settings

def get_logger(name=__name__):
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = "%(asctime)s %(levelname)s %(name)s - %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    return logger

logger = get_logger("capstone")
