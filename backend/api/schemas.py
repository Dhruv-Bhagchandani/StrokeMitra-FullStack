"""API request/response schemas."""

from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator

from src.scoring.schemas import SeverityLevel
from src.risk.schemas import RiskTier
from src.reporting.schemas import SegmentAnnotation


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(..., description="API version")


class ReadinessResponse(BaseModel):
    """Readiness check response."""

    ready: bool = Field(..., description="Service ready status")
    checks: dict[str, bool] = Field(..., description="Individual component checks")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class AnalyseRequest(BaseModel):
    """Request for speech analysis (multipart form data)."""

    # Metadata (provided as form fields, audio_file as UploadFile)
    patient_age: Optional[int] = Field(None, ge=18, le=120, description="Patient age")
    onset_hours: Optional[float] = Field(
        None,
        ge=0,
        le=168,
        description="Hours since symptom onset"
    )
    language: str = Field(default="en", description="Language code (ISO 639-1)")
    return_pdf: bool = Field(default=True, description="Generate PDF report")
    return_json: bool = Field(default=True, description="Include JSON report data")

    @field_validator("language")
    @classmethod
    def validate_language(cls, v):
        """Validate language code."""
        supported = ["en"]  # Currently only English supported
        if v not in supported:
            raise ValueError(f"Unsupported language: {v}. Supported: {supported}")
        return v


class AnalyseResponse(BaseModel):
    """Response from speech analysis."""

    # Request identification
    request_id: str = Field(..., description="Unique request ID")

    # Primary results
    slurring_score: float = Field(..., description="Slurring score (0-100)")
    severity: SeverityLevel = Field(..., description="Severity level")
    risk_score: float = Field(..., description="Risk score (0-100)")
    risk_tier: RiskTier = Field(..., description="Risk tier")
    confidence: float = Field(..., description="Model confidence (0-1)")

    # Explainability
    segments: list[SegmentAnnotation] = Field(
        default_factory=list,
        description="Annotated time segments"
    )

    # Acoustic summary
    acoustic_summary: dict = Field(..., description="Key acoustic features")

    # Report URLs
    report_url: Optional[str] = Field(None, description="URL to retrieve full report (PDF)")
    json_report: Optional[dict] = Field(None, description="JSON report data (if requested)")

    # Metadata
    processing_time_ms: float = Field(..., description="Total processing time (ms)")
    model_version: str = Field(..., description="Model version used")
    api_version: str = Field(default="1.0.0", description="API version")

    # Emergency flag
    emergency_alert: bool = Field(
        default=False,
        description="Emergency alert flag (risk tier = critical)"
    )

    class Config:
        json_schema_extra = {
            "example": {
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "slurring_score": 63.4,
                "severity": "moderate",
                "risk_score": 71.2,
                "risk_tier": "high",
                "confidence": 0.87,
                "segments": [
                    {
                        "start_ms": 420,
                        "end_ms": 1150,
                        "label": "imprecise_consonants",
                        "weight": 0.82
                    }
                ],
                "acoustic_summary": {
                    "speaking_rate_syllables_per_sec": 2.8,
                    "pitch_variability_hz": 18.4,
                    "pause_ratio": 0.34,
                    "vowel_space_area": 0.61
                },
                "report_url": "https://api.example.com/v1/report/550e8400",
                "processing_time_ms": 1240,
                "model_version": "ensemble-v1.0",
                "emergency_alert": False
            }
        }


class ErrorResponse(BaseModel):
    """Error response."""

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    request_id: Optional[str] = Field(None, description="Request ID if available")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Audio file duration exceeds maximum allowed (60s)",
                "request_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-03-15T10:30:00Z"
            }
        }


class ReportRetrievalResponse(BaseModel):
    """Response when retrieving a generated report."""

    report_id: str = Field(..., description="Report ID")
    format: str = Field(..., description="Report format (pdf or json)")
    generated_at: datetime = Field(..., description="Report generation timestamp")
    file_size_bytes: Optional[int] = Field(None, description="File size in bytes")
    expires_at: Optional[datetime] = Field(None, description="Report expiration timestamp")
