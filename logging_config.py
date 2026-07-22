import logging
import os

# Create logs directory if it doesn't exist
os.makedirs("logs", exist_ok=True)

# Create logger
logger = logging.getLogger("CrosswalkAI")
logger.setLevel(logging.INFO)

# Avoid duplicate handlers
if not logger.handlers:

    file_handler = logging.FileHandler("logs/crosswalk.log")

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)