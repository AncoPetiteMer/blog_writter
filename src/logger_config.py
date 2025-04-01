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
    log_dir = "../logs"
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

def log_startup_context(logger, topic: str, language: str, config):
    logger.warning(
        "\n" + "*" * 60 + "\n" +
        "*********************  START  *********************" + "\n" +
        "*" * 60
    )

    logger.info(
        f"\n📌 Generating enhanced SEO blog"
        f"\n→ Topic: '{topic}'"
        f"\n→ Language: {language}\n"
    )

    # Safely access nested fields using dot notation
    try:
        logger.info("🧠 LLM Configuration:")
        logger.info(f"→ Model: {config.llm.model}")
        logger.info(f"→ Temperature: {config.llm.temperature}")

        logger.info("🔍 Tavily Search Configuration:")
        logger.info(f"→ Enabled: {config.tavily.enabled}")
        logger.info(f"→ Search Depth: {config.tavily.search_depth}")
        logger.info(f"→ Max Results: {config.tavily.max_results}")

        logger.info("📊 SEMrush Configuration:")
        logger.info(f"→ Display Limit: {config.semrush.display_limit}")
        logger.info(f"→ Database: {config.semrush.database}")

        logger.info("📝 Blog Output Settings:")
        logger.info(f"→ Language: {config.blog.language}")
        logger.info(
            f"→ Word Count Range: {config.blog.word_count.min}–{config.blog.word_count.max}"
        )
        logger.info(f"→ HTML Format: {config.blog.html_format}")
    except AttributeError as e:
        logger.warning(f"⚠️ Some configuration fields could not be logged: {e}")
