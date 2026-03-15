"""Data schemas for risk assessment module."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class RiskTier(str, Enum):
    """Risk tier classification for stroke risk."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    @property
    def label(self) -> str:
        """Human-readable label."""
        return self.value.capitalize()

    @property
    def color(self) -> str:
        """Color for visualization."""
        color_map = {
            "low": "green",
            "moderate": "yellow",
            "high": "orange",
            "critical": "red",
        }
        return color_map[self.value]

    @property
    def color_hex(self) -> str:
        """Hex color code."""
        hex_map = {
            "low": "#4CAF50",
            "moderate": "#FFC107",
            "high": "#FF9800",
            "critical": "#F44336",
        }
        return hex_map[self.value]

    @property
    def description(self) -> str:
        """Risk description."""
        desc_map = {
            "low": "Low stroke risk based on speech analysis",
            "moderate": "Moderate stroke risk indicators present",
            "high": "High stroke risk, immediate evaluation recommended",
            "critical": "Critical stroke risk, emergency intervention required",
        }
        return desc_map[self.value]

    @property
    def action(self) -> str:
        """Recommended action."""
        action_map = {
            "low": "Continue monitoring. No immediate intervention needed.",
            "moderate": "Schedule clinical evaluation within 24-48 hours.",
            "high": "Seek medical evaluation within 12 hours. Monitor closely.",
            "critical": "EMERGENCY: Call 911 or go to nearest emergency department immediately.",
        }
        return action_map[self.value]

    @property
    def icon(self) -> str:
        """Icon for display."""
        icon_map = {
            "low": "✓",
            "moderate": "⚠",
            "high": "⚠⚠",
            "critical": "🚨",
        }
        return icon_map[self.value]

    @property
    def is_emergency(self) -> bool:
        """Whether this tier represents an emergency."""
        return self == RiskTier.CRITICAL


class RiskAssessment(BaseModel):
    """Comprehensive stroke risk assessment result."""

    # Risk score
    risk_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Stroke risk score (0-100)"
    )

    # Risk tier
    risk_tier: RiskTier = Field(..., description="Risk tier classification")

    # Contributing factors
    slurring_score: float = Field(..., description="Slurring score contribution")
    age_factor: Optional[float] = Field(None, description="Age factor (0-1)")
    onset_factor: Optional[float] = Field(None, description="Onset factor (0.5 or 1.0)")

    # Input context
    patient_age: Optional[int] = Field(None, ge=18, le=120, description="Patient age")
    onset_hours: Optional[float] = Field(
        None,
        ge=0,
        le=168,
        description="Hours since symptom onset"
    )

    # Logistic regression details
    logistic_weights: dict[str, float] = Field(
        ...,
        description="Logistic regression weights used"
    )
    logit_value: float = Field(..., description="Raw logit before sigmoid")

    # Emergency flag
    emergency_alert: bool = Field(
        default=False,
        description="Whether to trigger emergency alert"
    )

    # Additional risk factors (optional)
    additional_risk_factors: Optional[list[str]] = Field(
        None,
        description="Additional risk factors if provided"
    )

    def get_tier_info(self) -> dict:
        """Get risk tier information."""
        return {
            "tier": self.risk_tier.value,
            "label": self.risk_tier.label,
            "color": self.risk_tier.color,
            "color_hex": self.risk_tier.color_hex,
            "description": self.risk_tier.description,
            "action": self.risk_tier.action,
            "icon": self.risk_tier.icon,
            "is_emergency": self.risk_tier.is_emergency,
        }

    def get_contributing_factors_summary(self) -> dict:
        """Get summary of contributing factors."""
        factors = {
            "slurring_score": round(self.slurring_score, 1),
        }

        if self.age_factor is not None:
            factors["age_factor"] = round(self.age_factor, 2)
            factors["patient_age"] = self.patient_age

        if self.onset_factor is not None:
            factors["onset_factor"] = round(self.onset_factor, 2)
            factors["onset_hours"] = round(self.onset_hours, 1) if self.onset_hours else None
            factors["within_golden_window"] = self.onset_factor == 1.0

        return factors

    def is_within_golden_window(self) -> Optional[bool]:
        """Check if symptom onset is within golden window (4.5 hours)."""
        if self.onset_hours is None:
            return None
        return self.onset_hours <= 4.5

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "risk_score": round(self.risk_score, 1),
            "risk_tier": self.risk_tier.value,
            "emergency_alert": self.emergency_alert,
            "contributing_factors": self.get_contributing_factors_summary(),
        }
