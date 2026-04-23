import logging
import os
from datetime import datetime


def mute_other_logging():
    logging.getLogger().setLevel(logging.WARNING)
    for name in [
        "httpx",
        "httpcore",
        "httpcore.http11",
        "openai",
        "openai._base_client",
        "urllib3.connectionpool",
    ]:
        logger = logging.getLogger(name)
        logger.setLevel(logging.WARNING)  # Keep WARNING+ only
        logger.propagate = True  # Let logs bubble to root handlers


def setup_logging():
    """Configure logging and create a new log file for each run."""
    # Ensure logs directory exists
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)

    # Create timestamped log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"logs/xtalk_{timestamp}.log"

    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            # Console handler
            logging.StreamHandler(),
            # File handler
            logging.FileHandler(log_filename, encoding="utf-8"),
        ],
    )

    # Return xtalk logger
    logger = logging.getLogger("xtalk")

    return logger


# Initialize logger on import
logger = setup_logging()
