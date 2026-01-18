"""Logging infrastructure for the project."""
import logging
import sys
from pathlib import Path


def setup_logger(name, log_path="logs/markov.log", level=logging.INFO):
    """
    Set up a logger with both file and console handlers.
    
    Args:
        name: Logger name
        log_path: Path to log file
        level: Logging level
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create log directory
    log_dir = Path(log_path).parent
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(name):
    """Get existing logger by name."""
    return logging.getLogger(name)


# Create project logger
project_logger = setup_logger("markov_mlops", "logs/markov.log")
