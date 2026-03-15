"""Data schemas for scoring module."""

from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    """Severity level classification for speech slurring."""

    NONE = "none"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"

    @property
    def label(self) -> str:
        """Human-readable label."""
        return self.value.capitalize()

    @property
    def color(self) -> str:
        """Color for visualization."""
        color_map = {
            "none": "green",
            "mild": "yellow",
            "moderate": "orange",
            "severe": "red",
        }
        return color_map[self.value]

    @property
    def color_hex(self) -> str:
        """Hex color code."""
        hex_map = {
            "none": "#4CAF50",
            "mild": "#FFC107",
            "moderate": "#FF9800",
            "severe": "#F44336",
        }
        return hex_map[self.value]

    @property
    def description(self) -> str:
        """Clinical description."""
        desc_map = {
            "none": "No detectable speech slurring",
            "mild": "Slight articulatory imprecision, generally intelligible",
            "moderate": "Noticeable slurring, reduced intelligibility",
            "severe": "Marked dysarthria, significantly impaired intelligibility",
        }
        return desc_map[self.value]

    @property
    def clinical_note(self) -> str:
        """Clinical interpretation note."""
        note_map = {
            "none": "Speech patterns within normal limits for this assessment.",
            "mild": "Mild articulatory inconsistencies detected. Recommend monitoring.",
            "moderate": "Moderate dysarthric features present. Clinical evaluation recommended.",
            "severe": "Severe dysarthria detected. Immediate clinical assessment advised.",
        }
        return note_map[self.value]


class SlurringResult(BaseModel):
    """Result of slurring score computation and severity classification."""

    # Scores
    slurring_score: float = Field(
        ...,
        ge=0,
        le=100,
        description="Slurring score (0-100)"
    )
    raw_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Raw model probability before scaling"
    )
    calibrated_probability: float = Field(
        ...,
        ge=0,
        le=1,
        description="Calibrated probability after Platt scaling"
    )

    # Severity classification
    severity: SeverityLevel = Field(..., description="Severity level")

    # Confidence
    confidence: float = Field(
        ...,
        ge=0,
        le=1,
        description="Model confidence in prediction"
    )

    # Confidence interval (optional)
    confidence_interval_lower: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Lower bound of 95% CI for slurring score"
    )
    confidence_interval_upper: Optional[float] = Field(
        None,
        ge=0,
        le=100,
        description="Upper bound of 95% CI for slurring score"
    )

    # Model metadata
    model_version: str = Field(..., description="Model version used for inference")
    ensemble_weights: Optional[dict[str, float]] = Field(
        None,
        description="Ensemble branch weights used"
    )

    def get_severity_info(self) -> dict:
        """Get severity level information."""
        return {
            "level": self.severity.value,
            "label": self.severity.label,
            "color": self.severity.color,
            "color_hex": self.severity.color_hex,
            "description": self.severity.description,
            "clinical_note": self.severity.clinical_note,
        }

    def is_high_confidence(self, threshold: float = 0.7) -> bool:
        """Check if prediction has high confidence."""
        return self.confidence >= threshold

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "slurring_score": round(self.slurring_score, 1),
            "severity": self.severity.value,
            "confidence": round(self.confidence, 2),
            "model_version": self.model_version,
        }
