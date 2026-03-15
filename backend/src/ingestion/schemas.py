"""Data schemas for audio ingestion module."""

from typing import Optional
from pathlib import Path
from datetime import datetime
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class AudioInput(BaseModel):
    """Input audio file or stream metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    file_path: Optional[Path] = Field(None, description="Path to audio file")
    file_name: str = Field(..., description="Original filename")
    file_size_bytes: int = Field(..., ge=0, description="File size in bytes")
    mime_type: str = Field(..., description="MIME type (audio/wav, audio/mp3, etc.)")

    # Audio metadata
    sample_rate: int = Field(..., gt=0, description="Original sample rate (Hz)")
    duration_sec: float = Field(..., gt=0, description="Duration in seconds")
    num_channels: int = Field(..., ge=1, le=2, description="Number of audio channels")

    # Timestamps
    uploaded_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("file_path")
    @classmethod
    def validate_file_path(cls, v):
        """Ensure file path exists if provided."""
        if v is not None and not v.exists():
            raise ValueError(f"File not found: {v}")
        return v

    @field_validator("mime_type")
    @classmethod
    def validate_mime_type(cls, v):
        """Validate supported MIME types."""
        supported = ["audio/wav", "audio/mpeg", "audio/mp3", "audio/ogg", "audio/m4a", "audio/flac"]
        if v not in supported:
            raise ValueError(f"Unsupported MIME type: {v}. Supported: {supported}")
        return v


class PreprocessedAudio(BaseModel):
    """Preprocessed audio waveform ready for feature extraction."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    waveform: np.ndarray = Field(..., description="Audio waveform array (samples,)")
    sample_rate: int = Field(..., description="Sample rate after preprocessing (Hz)")
    duration_sec: float = Field(..., description="Duration after preprocessing (seconds)")

    # Processing metadata
    original_duration_sec: float = Field(..., description="Original duration before VAD")
    resampled: bool = Field(default=False, description="Was audio resampled?")
    normalized: bool = Field(default=False, description="Was loudness normalized?")
    trimmed: bool = Field(default=False, description="Was silence trimmed?")
    vad_applied: bool = Field(default=False, description="Was VAD applied?")

    # VAD results
    speech_segments: Optional[list[tuple[float, float]]] = Field(
        None,
        description="Speech segments as (start_sec, end_sec) pairs"
    )
    speech_ratio: Optional[float] = Field(
        None,
        ge=0,
        le=1,
        description="Ratio of speech to total duration"
    )

    # Processing timestamp
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("waveform")
    @classmethod
    def validate_waveform(cls, v):
        """Validate waveform shape and type."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Waveform must be numpy array")
        if v.ndim != 1:
            raise ValueError(f"Waveform must be 1D array, got shape {v.shape}")
        if v.size == 0:
            raise ValueError("Waveform is empty")
        return v


class QualityMetrics(BaseModel):
    """Audio quality assessment metrics."""

    # Signal quality
    snr_db: Optional[float] = Field(None, description="Signal-to-noise ratio (dB)")
    clipping_ratio: float = Field(..., ge=0, le=1, description="Ratio of clipped samples")
    peak_amplitude: float = Field(..., ge=0, le=1, description="Peak amplitude (normalized)")
    rms_energy: float = Field(..., ge=0, description="RMS energy")

    # Validation flags
    is_valid: bool = Field(..., description="Overall quality validation passed")
    quality_issues: list[str] = Field(
        default_factory=list,
        description="List of quality issues detected"
    )

    # Thresholds used
    min_duration_sec: float = Field(5.0, description="Minimum duration threshold")
    max_duration_sec: float = Field(60.0, description="Maximum duration threshold")
    min_snr_db: float = Field(10.0, description="Minimum SNR threshold")
    max_clipping_ratio: float = Field(0.01, description="Maximum clipping ratio threshold")

    @field_validator("quality_issues")
    @classmethod
    def validate_quality_issues(cls, v, info):
        """Ensure quality_issues aligns with is_valid flag."""
        is_valid = info.data.get("is_valid")
        if is_valid and len(v) > 0:
            raise ValueError("is_valid=True but quality_issues not empty")
        if not is_valid and len(v) == 0:
            raise ValueError("is_valid=False but no quality_issues specified")
        return v

    def add_issue(self, issue: str):
        """Add a quality issue and mark as invalid."""
        self.quality_issues.append(issue)
        self.is_valid = False

    def get_quality_summary(self) -> str:
        """Get human-readable quality summary."""
        if self.is_valid:
            return "✓ Audio quality passed all checks"
        else:
            issues = ", ".join(self.quality_issues)
            return f"✗ Quality issues detected: {issues}"
