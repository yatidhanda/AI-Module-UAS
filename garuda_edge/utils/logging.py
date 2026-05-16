from __future__ import annotations

try:
    from loguru import logger
except Exception:
    import logging

    logger = logging.getLogger("garuda_edge")


def configure_logging() -> None:
    if hasattr(logger, "remove") and hasattr(logger, "add"):
        logger.remove()
        logger.add(lambda msg: print(msg, end=""), level="INFO")
    else:
        logging.basicConfig(level=logging.INFO)
