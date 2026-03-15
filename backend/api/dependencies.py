"""FastAPI dependencies."""

import logging
from functools import lru_cache

from src.pipeline import SlurringDetectionPipeline

logger = logging.getLogger(__name__)


@lru_cache
def get_pipeline() -> SlurringDetectionPipeline:
    """
    Get singleton pipeline instance with trained model.

    Returns:
        SlurringDetectionPipeline instance (cached, using trained HuBERT model)
    """
    logger.info("Initializing pipeline with trained model (singleton)")
    return SlurringDetectionPipeline(use_placeholder=False)
