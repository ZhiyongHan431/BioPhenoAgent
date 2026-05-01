"""
日志系统 — 统一的日志记录与控制台输出
"""

from __future__ import annotations

import logging
import sys
from typing import Optional

_loggers: dict[str, logging.Logger] = {}


def setup_logger(
    name: str = "AgentFramework",
    level: str = "INFO",
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    # 控制台 handler
    console = logging.StreamHandler(sys.stderr)
    console.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
    logger.addHandler(console)

    # 文件 handler
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    _loggers[name] = logger
    return logger


def get_logger(name: str = "AgentFramework") -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)
