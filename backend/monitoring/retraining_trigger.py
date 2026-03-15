"""
Retraining trigger system for model maintenance.

Monitors:
- Data drift thresholds
- Performance degradation
- Prediction distribution shifts

When thresholds are breached, triggers alerts and retraining workflows.

Usage:
    trigger = RetrainingTrigger(
        drift_threshold=0.5,
        performance_threshold=0.85,
    )

    should_retrain = trigger.evaluate(
        drift_score=0.65,
        current_accuracy=0.82,
    )
"""

import logging
from typing import Optional, Dict, Any, List
from pathlib import Path
from datetime import datetime
import json
import yaml


# ══════════════════════════════════════════════════════════════════════════════
# Alert Types
# ══════════════════════════════════════════════════════════════════════════════

class AlertLevel:
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


class Alert:
    """Represents a retraining alert."""

    def __init__(
        self,
        level: str,
        message: str,
        metric_name: str,
        metric_value: float,
        threshold: float,
        timestamp: Optional[datetime] = None,
    ):
        self.level = level
        self.message = message
        self.metric_name = metric_name
        self.metric_value = metric_value
        self.threshold = threshold
        self.timestamp = timestamp or datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "level": self.level,
            "message": self.message,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self) -> str:
        return (
            f"Alert(level={self.level}, metric={self.metric_name}, "
            f"value={self.metric_value:.4f}, threshold={self.threshold:.4f})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# Retraining Trigger
# ══════════════════════════════════════════════════════════════════════════════

class RetrainingTrigger:
    """
    Monitor model health and trigger retraining when needed.

    Conditions for retraining:
    1. Data drift exceeds threshold
    2. Model performance drops below threshold
    3. Prediction distribution shifts significantly
    4. Manual trigger
    """

    def __init__(
        self,
        drift_threshold: float = 0.5,
        performance_threshold: float = 0.85,
        prediction_shift_threshold: float = 0.3,
        alert_log_path: Optional[Path] = None,
    ):
        """
        Initialize retraining trigger.

        Args:
            drift_threshold: Max acceptable drift score (0-1)
            performance_threshold: Min acceptable model accuracy/AUC
            prediction_shift_threshold: Max acceptable prediction distribution shift
            alert_log_path: Path to save alert logs
        """
        self.drift_threshold = drift_threshold
        self.performance_threshold = performance_threshold
        self.prediction_shift_threshold = prediction_shift_threshold

        self.alert_log_path = alert_log_path or Path("logs/retraining_alerts.jsonl")
        self.alert_log_path.parent.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.alerts: List[Alert] = []

    # ─────────────────────────────────────────────────────────────────────────
    # Main Evaluation
    # ─────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        drift_score: Optional[float] = None,
        current_accuracy: Optional[float] = None,
        current_auc: Optional[float] = None,
        prediction_shift: Optional[float] = None,
    ) -> bool:
        """
        Evaluate whether retraining should be triggered.

        Args:
            drift_score: Current data drift score (0-1)
            current_accuracy: Current model accuracy (0-1)
            current_auc: Current model AUC-ROC (0-1)
            prediction_shift: Prediction distribution shift (0-1)

        Returns:
            True if retraining should be triggered
        """
        self.alerts = []
        should_retrain = False

        # Check drift
        if drift_score is not None:
            if drift_score > self.drift_threshold:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"Data drift exceeded threshold: {drift_score:.4f} > {self.drift_threshold}",
                    metric_name="drift_score",
                    metric_value=drift_score,
                    threshold=self.drift_threshold,
                )
                self.alerts.append(alert)
                should_retrain = True
                self.logger.critical(alert.message)

        # Check accuracy
        if current_accuracy is not None:
            if current_accuracy < self.performance_threshold:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"Accuracy dropped below threshold: {current_accuracy:.4f} < {self.performance_threshold}",
                    metric_name="accuracy",
                    metric_value=current_accuracy,
                    threshold=self.performance_threshold,
                )
                self.alerts.append(alert)
                should_retrain = True
                self.logger.critical(alert.message)

        # Check AUC
        if current_auc is not None:
            if current_auc < self.performance_threshold:
                alert = Alert(
                    level=AlertLevel.CRITICAL,
                    message=f"AUC dropped below threshold: {current_auc:.4f} < {self.performance_threshold}",
                    metric_name="auc",
                    metric_value=current_auc,
                    threshold=self.performance_threshold,
                )
                self.alerts.append(alert)
                should_retrain = True
                self.logger.critical(alert.message)

        # Check prediction shift
        if prediction_shift is not None:
            if prediction_shift > self.prediction_shift_threshold:
                alert = Alert(
                    level=AlertLevel.WARNING,
                    message=f"Prediction distribution shifted: {prediction_shift:.4f} > {self.prediction_shift_threshold}",
                    metric_name="prediction_shift",
                    metric_value=prediction_shift,
                    threshold=self.prediction_shift_threshold,
                )
                self.alerts.append(alert)
                self.logger.warning(alert.message)
                # Note: prediction shift alone doesn't trigger retraining, but combined with other factors it might

        # Log alerts
        if self.alerts:
            self._log_alerts()

        # Summary
        if should_retrain:
            self.logger.critical(
                f"🔴 RETRAINING RECOMMENDED: {len(self.alerts)} critical issue(s) detected"
            )
        elif self.alerts:
            self.logger.warning(
                f"⚠️  Model monitoring alert: {len(self.alerts)} warning(s) detected"
            )
        else:
            self.logger.info("✓ Model health check passed")

        return should_retrain

    # ─────────────────────────────────────────────────────────────────────────
    # Alert Logging
    # ─────────────────────────────────────────────────────────────────────────

    def _log_alerts(self):
        """Save alerts to log file."""
        with open(self.alert_log_path, "a") as f:
            for alert in self.alerts:
                f.write(json.dumps(alert.to_dict()) + "\n")

    def get_recent_alerts(self, n: int = 10) -> List[Alert]:
        """Get most recent alerts from log."""
        if not self.alert_log_path.exists():
            return []

        alerts = []
        with open(self.alert_log_path, "r") as f:
            for line in f:
                alert_dict = json.loads(line)
                alert = Alert(
                    level=alert_dict["level"],
                    message=alert_dict["message"],
                    metric_name=alert_dict["metric_name"],
                    metric_value=alert_dict["metric_value"],
                    threshold=alert_dict["threshold"],
                    timestamp=datetime.fromisoformat(alert_dict["timestamp"]),
                )
                alerts.append(alert)

        # Return most recent n alerts
        return alerts[-n:]

    # ─────────────────────────────────────────────────────────────────────────
    # Manual Trigger
    # ─────────────────────────────────────────────────────────────────────────

    def trigger_manual_retraining(self, reason: str):
        """Manually trigger retraining."""
        alert = Alert(
            level=AlertLevel.CRITICAL,
            message=f"Manual retraining triggered: {reason}",
            metric_name="manual_trigger",
            metric_value=1.0,
            threshold=0.0,
        )
        self.alerts = [alert]
        self._log_alerts()
        self.logger.critical(alert.message)
        return True

    # ─────────────────────────────────────────────────────────────────────────
    # Notification Integration
    # ─────────────────────────────────────────────────────────────────────────

    def send_notification(self, webhook_url: Optional[str] = None):
        """
        Send notification via webhook (Slack, email, etc.).

        Args:
            webhook_url: Webhook endpoint for notifications
        """
        if not self.alerts:
            return

        if webhook_url:
            # Example: Send to Slack webhook
            import requests

            critical_alerts = [a for a in self.alerts if a.level == AlertLevel.CRITICAL]
            warning_alerts = [a for a in self.alerts if a.level == AlertLevel.WARNING]

            message = {
                "text": f"🔴 Model Retraining Alert",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*Model Monitoring Alert*\n\n"
                                    f"Critical: {len(critical_alerts)}\n"
                                    f"Warnings: {len(warning_alerts)}",
                        },
                    },
                ],
            }

            for alert in self.alerts:
                message["blocks"].append({
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"• {alert.message}",
                    },
                })

            try:
                response = requests.post(webhook_url, json=message, timeout=5)
                response.raise_for_status()
                self.logger.info("✓ Notification sent successfully")
            except Exception as e:
                self.logger.error(f"Failed to send notification: {e}")
        else:
            self.logger.warning("No webhook URL configured for notifications")

    # ─────────────────────────────────────────────────────────────────────────
    # Report Generation
    # ─────────────────────────────────────────────────────────────────────────

    def generate_report(self, output_path: Optional[Path] = None) -> str:
        """
        Generate human-readable alert report.

        Returns:
            Report as string
        """
        if output_path is None:
            output_path = Path("reports/retraining_alerts_report.txt")
            output_path.parent.mkdir(parents=True, exist_ok=True)

        lines = []
        lines.append("=" * 80)
        lines.append("MODEL RETRAINING ALERT REPORT")
        lines.append("=" * 80)
        lines.append(f"Generated: {datetime.utcnow().isoformat()}")
        lines.append(f"Total Alerts: {len(self.alerts)}")
        lines.append("")

        critical = [a for a in self.alerts if a.level == AlertLevel.CRITICAL]
        warnings = [a for a in self.alerts if a.level == AlertLevel.WARNING]

        lines.append(f"Critical Alerts: {len(critical)}")
        for alert in critical:
            lines.append(f"  🔴 {alert.message}")
            lines.append(f"     Metric: {alert.metric_name}")
            lines.append(f"     Value: {alert.metric_value:.4f}")
            lines.append(f"     Threshold: {alert.threshold:.4f}")
            lines.append("")

        lines.append(f"Warning Alerts: {len(warnings)}")
        for alert in warnings:
            lines.append(f"  ⚠️  {alert.message}")
            lines.append(f"     Metric: {alert.metric_name}")
            lines.append(f"     Value: {alert.metric_value:.4f}")
            lines.append(f"     Threshold: {alert.threshold:.4f}")
            lines.append("")

        lines.append("=" * 80)
        lines.append("")

        if len(critical) > 0:
            lines.append("RECOMMENDATION: Immediate retraining required")
        elif len(warnings) > 0:
            lines.append("RECOMMENDATION: Monitor closely, consider retraining soon")
        else:
            lines.append("RECOMMENDATION: No action needed")

        lines.append("=" * 80)

        report = "\n".join(lines)

        # Save to file
        with open(output_path, "w") as f:
            f.write(report)

        self.logger.info(f"✓ Alert report saved to {output_path}")
        return report


# ══════════════════════════════════════════════════════════════════════════════
# CLI Tool
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Model retraining trigger evaluation")
    parser.add_argument("--drift-score", type=float, help="Current drift score")
    parser.add_argument("--accuracy", type=float, help="Current accuracy")
    parser.add_argument("--auc", type=float, help="Current AUC")
    parser.add_argument("--manual", type=str, help="Manually trigger with reason")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    trigger = RetrainingTrigger(
        drift_threshold=0.5,
        performance_threshold=0.85,
        prediction_shift_threshold=0.3,
    )

    if args.manual:
        trigger.trigger_manual_retraining(args.manual)
        print("\n✓ Manual retraining triggered")
    else:
        should_retrain = trigger.evaluate(
            drift_score=args.drift_score,
            current_accuracy=args.accuracy,
            current_auc=args.auc,
        )

        print("\n" + "=" * 80)
        print(f"RETRAINING NEEDED: {should_retrain}")
        print("=" * 80)

        if trigger.alerts:
            report = trigger.generate_report()
            print("\n" + report)
