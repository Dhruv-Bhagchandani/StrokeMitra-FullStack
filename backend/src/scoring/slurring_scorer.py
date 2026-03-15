"""Slurring score computation from calibrated probabilities."""

import logging

from src.scoring.schemas import SlurringResult, SeverityLevel

logger = logging.getLogger(__name__)


class SlurringScorer:
    """Compute slurring score (0-100) from model output."""

    def __init__(self):
        """Initialize slurring scorer."""
        pass

    def compute_score(
        self,
        raw_probability: float,
        calibrated_probability: float,
        confidence: float,
        model_version: str = "unknown",
    ) -> SlurringResult:
        """
        Compute slurring score from probabilities.

        Args:
            raw_probability: Raw model probability of dysarthria
            calibrated_probability: Calibrated probability after Platt scaling
            confidence: Model confidence (0-1)
            model_version: Model version string

        Returns:
            SlurringResult with score and severity
        """
        # Slurring score: calibrated probability × 100
        slurring_score = round(calibrated_probability * 100, 1)

        # Classify severity based on score
        severity = self._classify_severity(slurring_score)

        logger.info(
            f"Slurring score computed: {slurring_score} (severity={severity.value})"
        )

        return SlurringResult(
            slurring_score=slurring_score,
            raw_probability=raw_probability,
            calibrated_probability=calibrated_probability,
            severity=severity,
            confidence=confidence,
            model_version=model_version,
        )

    def _classify_severity(self, slurring_score: float) -> SeverityLevel:
        """
        Classify severity based on slurring score.

        Thresholds:
        - 0-20: None
        - 21-45: Mild
        - 46-70: Moderate
        - 71-100: Severe
        """
        if slurring_score <= 20.0:
            return SeverityLevel.NONE
        elif slurring_score <= 45.0:
            return SeverityLevel.MILD
        elif slurring_score <= 70.0:
            return SeverityLevel.MODERATE
        else:
            return SeverityLevel.SEVERE
