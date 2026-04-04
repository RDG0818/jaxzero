import sys
from loguru import logger

logger.level("INFO", color="<white>")
logger.level("DEBUG", color="<cyan>")
logger.level("WARNING", color="<yellow>")
logger.level("ERROR", color="<red>")
logger.level("CRITICAL", color="<white><bg red>")

logger.remove()

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<level>{message}</level>"
)

logger.add(sys.stderr, level="INFO", format=_LOG_FORMAT, colorize=True)


def enable_debug():
    """Switch the logger to DEBUG level for the current process."""
    logger.remove()
    logger.add(sys.stderr, level="DEBUG", format=_LOG_FORMAT, colorize=True)
