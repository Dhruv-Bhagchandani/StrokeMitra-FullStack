"""Severity classification logic."""

import logging

from src.scoring.schemas import SeverityLevel

logger = logging.getLogger(__name__)


class SeverityClassifier:
    """Classify dysarthria severity from slurring score."""

    # Default thresholds (can be customized)
    DEFAULT_THRESHOLDS = [20.0, 45.0, 70.0]

    def __init__(self, thresholds: list[float] = None):
        """
        Initialize severity classifier.

        Args:
            thresholds: List of 3 thresholds [none/mild, mild/moderate, moderate/severe]
        """
        self.thresholds = thresholds or self.DEFAULT_THRESHOLDS

        if len(self.thresholds) != 3:
            raise ValueError("Must provide exactly 3 thresholds")

        logger.info(f"SeverityClassifier initialized with thresholds: {self.thresholds}")

    def classify(self, slurring_score: float) -> SeverityLevel:
        """
        Classify severity level.

        Args:
            slurring_score: Slurring score (0-100)

        Returns:
            SeverityLevel enum
        """
        t1, t2, t3 = self.thresholds

        if slurring_score <= t1:
            return SeverityLevel.NONE
        elif slurring_score <= t2:
            return SeverityLevel.MILD
        elif slurring_score <= t3:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE

    def get_clinical_interpretation(self, severity: SeverityLevel) -> str:
        """Get clinical interpretation for severity level."""
        return severity.clinical_note
