"""Data schemas for report generation module."""

from typing import Optional
from datetime import datetime
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict

from src.scoring.schemas import SlurringResult
from src.risk.schemas import RiskAssessment


class SegmentAnnotation(BaseModel):
    """Annotated time segment with explanation."""

    start_ms: int = Field(..., ge=0, description="Segment start time (milliseconds)")
    end_ms: int = Field(..., gt=0, description="Segment end time (milliseconds)")
    label: str = Field(..., description="Segment label (e.g., 'imprecise_consonants')")
    weight: float = Field(..., ge=0, le=1, description="Importance weight")

    # Optional detailed description
    description: Optional[str] = Field(None, description="Human-readable description")

    @property
    def duration_ms(self) -> int:
        """Segment duration in milliseconds."""
        return self.end_ms - self.start_ms

    @property
    def duration_sec(self) -> float:
        """Segment duration in seconds."""
        return self.duration_ms / 1000.0

    def get_time_range_str(self) -> str:
        """Get formatted time range string."""
        start_sec = self.start_ms / 1000.0
        end_sec = self.end_ms / 1000.0
        return f"{start_sec:.2f}s - {end_sec:.2f}s"


class ReportData(BaseModel):
    """Complete data bundle for report generation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Identification
    report_id: str = Field(..., description="Unique report ID")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # Results
    slurring_result: SlurringResult = Field(..., description="Slurring analysis result")
    risk_assessment: RiskAssessment = Field(..., description="Risk assessment result")

    # Acoustic summary
    acoustic_summary: dict = Field(..., description="Key acoustic features")

    # Explainability
    segments: list[SegmentAnnotation] = Field(
        default_factory=list,
        description="Annotated time segments"
    )

    # Visualizations (file paths to generated plots)
    waveform_plot_path: Optional[Path] = Field(None, description="Path to waveform plot PNG")
    spectrogram_plot_path: Optional[Path] = Field(None, description="Path to spectrogram PNG")
    heatmap_plot_path: Optional[Path] = Field(None, description="Path to Grad-CAM heatmap PNG")

    # Patient context (optional)
    patient_age: Optional[int] = Field(None, description="Patient age")
    onset_hours: Optional[float] = Field(None, description="Symptom onset hours")

    # Processing metadata
    processing_time_ms: float = Field(..., description="Total processing time (milliseconds)")
    audio_duration_sec: float = Field(..., description="Audio duration (seconds)")

    # Configuration used
    feature_config: Optional[dict] = Field(None, description="Feature extraction config")
    model_configuration: Optional[dict] = Field(None, description="Model config")

    def get_report_title(self) -> str:
        """Get report title."""
        return f"Speech Slurring Analysis Report - {self.report_id}"

    def get_summary(self) -> dict:
        """Get concise summary for quick reference."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "slurring_score": self.slurring_result.slurring_score,
            "severity": self.slurring_result.severity.value,
            "risk_score": self.risk_assessment.risk_score,
            "risk_tier": self.risk_assessment.risk_tier.value,
            "confidence": self.slurring_result.confidence,
            "emergency_alert": self.risk_assessment.emergency_alert,
        }


class ReportOutput(BaseModel):
    """Generated report output (PDF and/or JSON)."""

    # Report identification
    report_id: str = Field(..., description="Unique report ID")
    generated_at: datetime = Field(default_factory=datetime.utcnow)

    # PDF report
    pdf_bytes: Optional[bytes] = Field(None, description="PDF report binary data")
    pdf_path: Optional[Path] = Field(None, description="Path to saved PDF file")

    # JSON report
    json_data: Optional[dict] = Field(None, description="JSON report data")
    json_path: Optional[Path] = Field(None, description="Path to saved JSON file")

    # URLs (for API responses)
    pdf_url: Optional[str] = Field(None, description="URL to retrieve PDF report")
    json_url: Optional[str] = Field(None, description="URL to retrieve JSON report")

    # Metadata
    file_size_bytes: Optional[int] = Field(None, description="PDF file size in bytes")

    def has_pdf(self) -> bool:
        """Check if PDF report is available."""
        return self.pdf_bytes is not None or self.pdf_path is not None

    def has_json(self) -> bool:
        """Check if JSON report is available."""
        return self.json_data is not None or self.json_path is not None

    def to_dict(self) -> dict:
        """Convert to dictionary for API response."""
        return {
            "report_id": self.report_id,
            "generated_at": self.generated_at.isoformat(),
            "pdf_url": self.pdf_url,
            "json_url": self.json_url,
            "file_size_bytes": self.file_size_bytes,
        }
