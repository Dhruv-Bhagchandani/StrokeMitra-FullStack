"""Build report data from analysis results."""

import logging
from datetime import datetime
from uuid import uuid4

from src.reporting.schemas import ReportData, SegmentAnnotation
from src.scoring.schemas import SlurringResult
from src.risk.schemas import RiskAssessment

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Build comprehensive report from analysis results."""

    def build(
        self,
        slurring_result: SlurringResult,
        risk_assessment: RiskAssessment,
        acoustic_summary: dict,
        segments: list[SegmentAnnotation],
        processing_time_ms: float,
        audio_duration_sec: float,
        patient_age: int | None = None,
        onset_hours: float | None = None,
    ) -> ReportData:
        """
        Build report data.

        Args:
            slurring_result: Slurring analysis result
            risk_assessment: Risk assessment
            acoustic_summary: Acoustic features summary
            segments: Annotated time segments
            processing_time_ms: Total processing time
            audio_duration_sec: Audio duration
            patient_age: Patient age (optional)
            onset_hours: Symptom onset hours (optional)

        Returns:
            ReportData ready for rendering
        """
        report_id = str(uuid4())

        logger.info(f"Building report: {report_id}")

        report = ReportData(
            report_id=report_id,
            generated_at=datetime.utcnow(),
            slurring_result=slurring_result,
            risk_assessment=risk_assessment,
            acoustic_summary=acoustic_summary,
            segments=segments,
            processing_time_ms=processing_time_ms,
            audio_duration_sec=audio_duration_sec,
            patient_age=patient_age,
            onset_hours=onset_hours,
        )

        logger.info(f"Report built: {report_id}")

        return report
