"""Render reports as JSON."""

import logging
import json
from pathlib import Path

from src.reporting.schemas import ReportData, ReportOutput

logger = logging.getLogger(__name__)


class JSONRenderer:
    """Render report as JSON."""

    def render(self, report_data: ReportData, save_path: Path | None = None) -> ReportOutput:
        """
        Render report as JSON.

        Args:
            report_data: Report data
            save_path: Optional path to save JSON file

        Returns:
            ReportOutput with JSON data
        """
        logger.info(f"Rendering JSON report: {report_data.report_id}")

        # Build JSON structure
        json_data = {
            "report_id": report_data.report_id,
            "generated_at": report_data.generated_at.isoformat(),
            "summary": report_data.get_summary(),
            "slurring_analysis": {
                "slurring_score": report_data.slurring_result.slurring_score,
                "severity": report_data.slurring_result.severity.value,
                "severity_info": report_data.slurring_result.get_severity_info(),
                "confidence": report_data.slurring_result.confidence,
                "model_version": report_data.slurring_result.model_version,
            },
            "risk_assessment": {
                "risk_score": report_data.risk_assessment.risk_score,
                "risk_tier": report_data.risk_assessment.risk_tier.value,
                "tier_info": report_data.risk_assessment.get_tier_info(),
                "contributing_factors": report_data.risk_assessment.get_contributing_factors_summary(),
                "emergency_alert": report_data.risk_assessment.emergency_alert,
            },
            "acoustic_summary": report_data.acoustic_summary,
            "segments": [
                {
                    "start_ms": seg.start_ms,
                    "end_ms": seg.end_ms,
                    "duration_ms": seg.duration_ms,
                    "label": seg.label,
                    "weight": seg.weight,
                    "time_range": seg.get_time_range_str(),
                }
                for seg in report_data.segments
            ],
            "processing_metadata": {
                "processing_time_ms": report_data.processing_time_ms,
                "audio_duration_sec": report_data.audio_duration_sec,
            },
        }

        # Save to file if requested
        json_path = None
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, "w") as f:
                json.dump(json_data, f, indent=2)
            json_path = save_path
            logger.info(f"JSON report saved: {save_path}")

        return ReportOutput(
            report_id=report_data.report_id,
            json_data=json_data,
            json_path=json_path,
        )
