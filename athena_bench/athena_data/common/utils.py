import os
import logging

def setup_logger(name="app", log_dir="data/logs/"):
    """
    Set up a logger that prints to console and writes to a log file.
    Creates the log directory if it doesn't exist.
    """
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{name}.log")

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Prevent duplicate handlers if logger is reused
    if not logger.handlers:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        ch = logging.StreamHandler()

        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(ch)

    return logger
