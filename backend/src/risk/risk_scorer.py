"""Stroke risk score computation."""

import logging
import numpy as np
from typing import Optional

from src.risk.schemas import RiskAssessment, RiskTier

logger = logging.getLogger(__name__)


class RiskScorer:
    """Compute stroke risk score from slurring score + clinical context."""

    def __init__(
        self,
        weights: dict = None,
        age_min: int = 40,
        age_max: int = 80,
        golden_window_hours: float = 4.5,
    ):
        """
        Initialize risk scorer.

        Args:
            weights: Logistic regression weights {w0, w1, w2, w3}
            age_min: Minimum age for scaling
            age_max: Maximum age for scaling
            golden_window_hours: Stroke golden window (hours)
        """
        # Default weights
        self.weights = weights or {
            "w0": -2.5,  # Intercept
            "w1": 0.05,  # Slurring score coefficient
            "w2": 1.2,   # Age factor coefficient
            "w3": 0.8,   # Onset factor coefficient
        }

        self.age_min = age_min
        self.age_max = age_max
        self.golden_window_hours = golden_window_hours

        logger.info(f"RiskScorer initialized with weights: {self.weights}")

    def compute_risk(
        self,
        slurring_score: float,
        patient_age: Optional[int] = None,
        onset_hours: Optional[float] = None,
    ) -> RiskAssessment:
        """
        Compute stroke risk score.

        Args:
            slurring_score: Slurring score (0-100)
            patient_age: Patient age (optional)
            onset_hours: Hours since symptom onset (optional)

        Returns:
            RiskAssessment with risk score and tier
        """
        # Compute factors
        age_factor = self._compute_age_factor(patient_age) if patient_age else 0.0
        onset_factor = self._compute_onset_factor(onset_hours) if onset_hours else 0.0

        # Logistic regression: risk_score = sigmoid(w0 + w1*slurring + w2*age + w3*onset) * 100
        logit = (
            self.weights["w0"]
            + self.weights["w1"] * slurring_score
            + self.weights["w2"] * age_factor
            + self.weights["w3"] * onset_factor
        )

        # Sigmoid
        risk_probability = 1 / (1 + np.exp(-logit))
        risk_score = round(risk_probability * 100, 1)

        # Classify risk tier
        risk_tier = self._classify_risk_tier(risk_score)

        # Emergency alert if critical
        emergency_alert = risk_tier == RiskTier.CRITICAL

        logger.info(
            f"Risk score computed: {risk_score} "
            f"(tier={risk_tier.value}, emergency={emergency_alert})"
        )

        return RiskAssessment(
            risk_score=risk_score,
            risk_tier=risk_tier,
            slurring_score=slurring_score,
            age_factor=age_factor if patient_age else None,
            onset_factor=onset_factor if onset_hours else None,
            patient_age=patient_age,
            onset_hours=onset_hours,
            logistic_weights=self.weights,
            logit_value=float(logit),
            emergency_alert=emergency_alert,
        )

    def _compute_age_factor(self, age: int) -> float:
        """
        Compute age factor (0-1 scaling).

        Formula: min(1.0, max(0.0, (age - age_min) / (age_max - age_min)))
        """
        age_factor = (age - self.age_min) / (self.age_max - self.age_min)
        return float(np.clip(age_factor, 0.0, 1.0))

    def _compute_onset_factor(self, onset_hours: float) -> float:
        """
        Compute onset factor.

        Returns:
            1.0 if within golden window (urgent)
            0.5 if outside golden window
        """
        if onset_hours <= self.golden_window_hours:
            return 1.0
        else:
            return 0.5

    def _classify_risk_tier(self, risk_score: float) -> RiskTier:
        """
        Classify risk tier.

        Thresholds:
        - 0-25: Low
        - 26-50: Moderate
        - 51-75: High
        - 76-100: Critical
        """
        if risk_score <= 25.0:
            return RiskTier.LOW
        elif risk_score <= 50.0:
            return RiskTier.MODERATE
        elif risk_score <= 75.0:
            return RiskTier.HIGH
        else:
            return RiskTier.CRITICAL
