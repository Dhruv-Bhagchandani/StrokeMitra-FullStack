"""Unit tests for individual pipeline components."""

import pytest
import numpy as np

from src.scoring.slurring_scorer import SlurringScorer
from src.scoring.severity_classifier import SeverityClassifier
from src.risk.risk_scorer import RiskScorer


def test_slurring_scorer():
    """Test slurring score computation."""
    scorer = SlurringScorer()

    result = scorer.compute_score(
        raw_probability=0.45,
        calibrated_probability=0.50,
        confidence=0.85,
        model_version="test-v1.0",
    )

    assert result.slurring_score == 50.0
    assert result.severity.value == "moderate"
    assert result.confidence == 0.85


def test_severity_classifier():
    """Test severity classification."""
    classifier = SeverityClassifier()

    assert classifier.classify(15.0).value == "none"
    assert classifier.classify(35.0).value == "mild"
    assert classifier.classify(55.0).value == "moderate"
    assert classifier.classify(85.0).value == "severe"


def test_risk_scorer_with_age_and_onset():
    """Test risk score with all factors."""
    scorer = RiskScorer()

    assessment = scorer.compute_risk(
        slurring_score=60.0,
        patient_age=70,
        onset_hours=2.0,
    )

    assert assessment.risk_score > 0
    assert assessment.risk_score <= 100
    assert assessment.risk_tier.value in ["low", "moderate", "high", "critical"]
    assert assessment.age_factor is not None
    assert assessment.onset_factor == 1.0  # Within golden window


def test_risk_scorer_without_context():
    """Test risk score without age/onset."""
    scorer = RiskScorer()

    assessment = scorer.compute_risk(
        slurring_score=30.0,
        patient_age=None,
        onset_hours=None,
    )

    assert assessment.risk_score > 0
    assert assessment.age_factor is None
    assert assessment.onset_factor is None


def test_risk_tier_thresholds():
    """Test risk tier classification thresholds."""
    scorer = RiskScorer()

    # Low
    low = scorer.compute_risk(slurring_score=5.0)
    assert low.risk_tier.value == "low"

    # Critical (high slurring, age, onset)
    critical = scorer.compute_risk(
        slurring_score=90.0,
        patient_age=80,
        onset_hours=1.0,
    )
    assert critical.risk_tier.value in ["high", "critical"]
    assert critical.emergency_alert == (critical.risk_tier.value == "critical")
