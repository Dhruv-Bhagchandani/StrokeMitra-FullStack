"""Risk tier utilities."""

from src.risk.schemas import RiskTier


def get_risk_tier(risk_score: float) -> RiskTier:
    """
    Get risk tier from risk score.

    Args:
        risk_score: Risk score (0-100)

    Returns:
        RiskTier enum
    """
    if risk_score <= 25.0:
        return RiskTier.LOW
    elif risk_score <= 50.0:
        return RiskTier.MODERATE
    elif risk_score <= 75.0:
        return RiskTier.HIGH
    else:
        return RiskTier.CRITICAL


def is_emergency(risk_tier: RiskTier) -> bool:
    """Check if risk tier represents an emergency."""
    return risk_tier == RiskTier.CRITICAL
