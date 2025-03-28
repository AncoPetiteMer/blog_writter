import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler

class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[94m",    # Bleu
        logging.INFO: "\033[92m",     # Vert
        logging.WARNING: "\033[93m",  # Jaune
        logging.ERROR: "\033[91m",    # Rouge
        logging.CRITICAL: "\033[95m", # Magenta
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, self.RESET)
        formatted = super().format(record)
        return f"{color}{formatted}{self.RESET}"

def get_logger(name: str = "seo_blog") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger  # Évite les doublons de handlers

    logger.setLevel(logging.INFO)

    # Handler console avec couleurs
    console_handler = logging.StreamHandler()
    console_formatter = ColorFormatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Handler fichier (logs archivés)
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_file = os.path.join(log_dir, f"{name}_{timestamp}.log")

    file_handler = RotatingFileHandler(
        os.path.join(log_dir, f"{name}.log"),
        maxBytes=1_000_000,  # 1 MB
        backupCount=5,  # max 5 fichiers d'archive
        encoding="utf-8"
    )
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    logger.info("Logger initialized with colored console output and file logging.")
    return logger
