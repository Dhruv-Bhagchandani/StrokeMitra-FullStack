"""
Prometheus metrics exporter for model monitoring.

Tracks:
- Request count and latency
- Model predictions (distribution, confidence)
- Error rates
- Drift scores
- System health (memory, CPU)

Usage:
    from monitoring.metrics_logger import metrics_logger

    # In API endpoint
    metrics_logger.log_prediction(
        slurring_score=45.2,
        confidence=0.87,
        latency_ms=1234,
        severity="moderate",
    )
"""

import time
import logging
from typing import Optional
from prometheus_client import (
    Counter,
    Histogram,
    Gauge,
    Summary,
    CollectorRegistry,
    generate_latest,
    CONTENT_TYPE_LATEST,
)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Definition
# ══════════════════════════════════════════════════════════════════════════════

# Custom registry (optional, can use default)
registry = CollectorRegistry()

# Request metrics
request_count = Counter(
    "slurring_api_requests_total",
    "Total number of API requests",
    labelnames=["endpoint", "method", "status"],
    registry=registry,
)

request_latency = Histogram(
    "slurring_api_request_duration_seconds",
    "Request latency in seconds",
    labelnames=["endpoint"],
    buckets=(0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
    registry=registry,
)

# Prediction metrics
prediction_count = Counter(
    "slurring_predictions_total",
    "Total number of predictions made",
    labelnames=["severity"],
    registry=registry,
)

slurring_score_distribution = Histogram(
    "slurring_score",
    "Distribution of slurring scores",
    buckets=(0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
    registry=registry,
)

confidence_distribution = Histogram(
    "slurring_confidence",
    "Distribution of model confidence scores",
    buckets=(0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99, 1.0),
    registry=registry,
)

# Risk metrics
risk_score_distribution = Histogram(
    "slurring_risk_score",
    "Distribution of risk scores",
    buckets=(0, 10, 25, 50, 75, 90, 100),
    registry=registry,
)

risk_tier_count = Counter(
    "slurring_risk_tier_total",
    "Count of predictions by risk tier",
    labelnames=["tier"],
    registry=registry,
)

emergency_alert_count = Counter(
    "slurring_emergency_alerts_total",
    "Number of emergency alerts triggered",
    registry=registry,
)

# Drift metrics
drift_score_gauge = Gauge(
    "slurring_drift_score",
    "Current data drift score",
    registry=registry,
)

drift_alert_count = Counter(
    "slurring_drift_alerts_total",
    "Number of drift alerts triggered",
    registry=registry,
)

# Model metrics
model_inference_latency = Summary(
    "slurring_model_inference_duration_seconds",
    "Model inference latency",
    registry=registry,
)

feature_extraction_latency = Summary(
    "slurring_feature_extraction_duration_seconds",
    "Feature extraction latency",
    registry=registry,
)

# Error metrics
error_count = Counter(
    "slurring_errors_total",
    "Total number of errors",
    labelnames=["error_type"],
    registry=registry,
)


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Logger
# ══════════════════════════════════════════════════════════════════════════════

class MetricsLogger:
    """
    Centralized metrics logging for Prometheus.

    This class provides a high-level API for logging various metrics
    throughout the application lifecycle.
    """

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    # ─────────────────────────────────────────────────────────────────────────
    # Request Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def log_request(
        self,
        endpoint: str,
        method: str,
        status: int,
        latency_seconds: float,
    ):
        """Log API request."""
        request_count.labels(
            endpoint=endpoint,
            method=method,
            status=str(status),
        ).inc()

        request_latency.labels(endpoint=endpoint).observe(latency_seconds)

    # ─────────────────────────────────────────────────────────────────────────
    # Prediction Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def log_prediction(
        self,
        slurring_score: float,
        confidence: float,
        severity: str,
        risk_score: Optional[float] = None,
        risk_tier: Optional[str] = None,
        is_emergency: bool = False,
        inference_latency_ms: Optional[float] = None,
    ):
        """Log model prediction metrics."""
        # Severity
        prediction_count.labels(severity=severity).inc()

        # Slurring score
        slurring_score_distribution.observe(slurring_score)

        # Confidence
        confidence_distribution.observe(confidence)

        # Risk (if provided)
        if risk_score is not None:
            risk_score_distribution.observe(risk_score)

        if risk_tier is not None:
            risk_tier_count.labels(tier=risk_tier).inc()

        if is_emergency:
            emergency_alert_count.inc()

        # Inference latency
        if inference_latency_ms is not None:
            model_inference_latency.observe(inference_latency_ms / 1000.0)

    # ─────────────────────────────────────────────────────────────────────────
    # Feature Extraction Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def log_feature_extraction(self, latency_seconds: float):
        """Log feature extraction latency."""
        feature_extraction_latency.observe(latency_seconds)

    # ─────────────────────────────────────────────────────────────────────────
    # Drift Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def log_drift(self, drift_score: float, has_drift: bool):
        """Log data drift detection."""
        drift_score_gauge.set(drift_score)

        if has_drift:
            drift_alert_count.inc()
            self.logger.warning(f"⚠️  Drift alert: score={drift_score:.4f}")

    # ─────────────────────────────────────────────────────────────────────────
    # Error Metrics
    # ─────────────────────────────────────────────────────────────────────────

    def log_error(self, error_type: str):
        """Log error occurrence."""
        error_count.labels(error_type=error_type).inc()

    # ─────────────────────────────────────────────────────────────────────────
    # Metrics Export
    # ─────────────────────────────────────────────────────────────────────────

    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format."""
        return generate_latest(registry)

    def get_content_type(self) -> str:
        """Get Prometheus content type."""
        return CONTENT_TYPE_LATEST


# ══════════════════════════════════════════════════════════════════════════════
# Singleton Instance
# ══════════════════════════════════════════════════════════════════════════════

metrics_logger = MetricsLogger()


# ══════════════════════════════════════════════════════════════════════════════
# Context Manager for Timing
# ══════════════════════════════════════════════════════════════════════════════

class TimingContext:
    """Context manager for timing operations."""

    def __init__(self, metric_observer):
        self.metric_observer = metric_observer
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start_time
        self.metric_observer.observe(elapsed)


def time_inference():
    """Context manager for timing model inference."""
    return TimingContext(model_inference_latency)


def time_feature_extraction():
    """Context manager for timing feature extraction."""
    return TimingContext(feature_extraction_latency)


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI Integration
# ══════════════════════════════════════════════════════════════════════════════

def create_metrics_endpoint():
    """
    Create FastAPI endpoint for Prometheus scraping.

    Add this to your FastAPI app:
        from monitoring.metrics_logger import create_metrics_endpoint
        app.add_route("/metrics", create_metrics_endpoint())
    """
    from fastapi import Response

    def metrics_endpoint():
        return Response(
            content=metrics_logger.get_metrics(),
            media_type=metrics_logger.get_content_type(),
        )

    return metrics_endpoint


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Simulate some predictions
    import random

    for i in range(100):
        severity = random.choice(["none", "mild", "moderate", "severe"])
        slurring_score = random.uniform(0, 100)
        confidence = random.uniform(0.7, 0.99)
        risk_tier = random.choice(["low", "moderate", "high", "critical"])

        metrics_logger.log_prediction(
            slurring_score=slurring_score,
            confidence=confidence,
            severity=severity,
            risk_score=random.uniform(0, 100),
            risk_tier=risk_tier,
            is_emergency=(risk_tier == "critical"),
            inference_latency_ms=random.uniform(500, 3000),
        )

    # Simulate API requests
    for i in range(50):
        metrics_logger.log_request(
            endpoint="/v1/speech/analyse",
            method="POST",
            status=random.choice([200, 200, 200, 400, 500]),
            latency_seconds=random.uniform(1.0, 5.0),
        )

    # Simulate drift
    metrics_logger.log_drift(drift_score=0.35, has_drift=False)
    metrics_logger.log_drift(drift_score=0.67, has_drift=True)

    # Print metrics
    print("\n" + "=" * 80)
    print("PROMETHEUS METRICS")
    print("=" * 80)
    print(metrics_logger.get_metrics().decode("utf-8"))
