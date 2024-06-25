"""
This module is used to configure the logger for the fast-causal-inference package.
"""

__all__ = ["get_logger"]

import logging.config

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "customFormatter": {
            "format": "[%(levelname)s] %(asctime)s - %(filename)s:%(lineno)d - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "consoleHandler": {
            "level": "INFO",
            "class": "logging.StreamHandler",
            "formatter": "customFormatter",
            "stream": "ext://sys.stdout",
        },
        "fileHandler": {
            "level": "DEBUG",
            "class": "logging.handlers.RotatingFileHandler",
            "filename": "fast-causal-inference.log",
            "mode": "a",
            "maxBytes": 1024 * 1024 * 200,
            "backupCount": 3,
            "formatter": "customFormatter",
        },
    },
    "loggers": {
        "": {  # root logger
            "handlers": ["consoleHandler"],
        },
        "fci": {
            "level": "DEBUG",
            "handlers": ["fileHandler", "consoleHandler"],
            "propagate": 0,
        },
    },
}


def get_logger():
    logging.config.dictConfig(LOGGING_CONFIG)

    logger = logging.getLogger("fci")
    return logger
