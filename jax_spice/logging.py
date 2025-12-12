"""Logging configuration for JAX-SPICE.

Provides two logging modes:
- Default: WARNING level only (quiet)
- Performance tracing: INFO level with flush (for Cloud Run visibility)

Usage:
    from jax_spice.logging import logger, enable_performance_logging

    # Default - only warnings
    logger.warning("This will show")
    logger.info("This won't show")

    # Enable for performance tracing (e.g., Cloud Run)
    enable_performance_logging()
    logger.info("Now this shows and flushes immediately")
"""

import logging
import sys

# Create the jax_spice logger
logger = logging.getLogger("jax_spice")

# Default: WARNING level only (quiet operation)
logger.setLevel(logging.WARNING)

# Add a default handler if none exists
if not logger.handlers:
    _default_handler = logging.StreamHandler(sys.stdout)
    _default_handler.setLevel(logging.WARNING)
    _default_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(_default_handler)


class FlushingHandler(logging.StreamHandler):
    """StreamHandler that flushes after every emit (for Cloud Run log visibility)."""

    def emit(self, record):
        super().emit(record)
        self.flush()


def enable_performance_logging():
    """Enable DEBUG level logging with immediate flush for performance tracing.

    Use this when running on Cloud Run or when you need to see logs
    in real-time during long-running operations.
    """
    logger.setLevel(logging.DEBUG)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Add flushing handler
    handler = FlushingHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def set_log_level(level: int):
    """Set the logging level.

    Args:
        level: logging.DEBUG, logging.INFO, logging.WARNING, etc.
    """
    logger.setLevel(level)
    for handler in logger.handlers:
        handler.setLevel(level)
