"""Platt scaling calibration for model outputs."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class PlattScaling:
    """
    Platt scaling for probability calibration.

    Transforms raw model outputs using: sigmoid(a * logit + b)
    """

    def __init__(self, a: float = 1.0, b: float = 0.0):
        """
        Initialize Platt scaling.

        Args:
            a: Scaling parameter
            b: Bias parameter
        """
        self.a = a
        self.b = b

        logger.info(f"PlattScaling initialized (a={a}, b={b})")

    def transform(self, raw_probability: float) -> float:
        """
        Apply Platt scaling to raw probability.

        Args:
            raw_probability: Raw model probability

        Returns:
            Calibrated probability
        """
        # Convert probability to logit
        logit = np.log(raw_probability / (1 - raw_probability + 1e-8))

        # Apply Platt scaling
        calibrated_logit = self.a * logit + self.b

        # Convert back to probability
        calibrated_prob = 1 / (1 + np.exp(-calibrated_logit))

        logger.debug(f"Calibration: {raw_probability:.3f} → {calibrated_prob:.3f}")

        return float(calibrated_prob)

    @classmethod
    def identity(cls) -> "PlattScaling":
        """Return identity calibration (no transformation)."""
        return cls(a=1.0, b=0.0)
