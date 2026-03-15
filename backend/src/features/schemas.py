"""Data schemas for acoustic feature extraction module."""

from typing import Optional
import numpy as np
from pydantic import BaseModel, Field, field_validator, ConfigDict


class MFCCFeatures(BaseModel):
    """MFCC feature representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mfcc: np.ndarray = Field(..., description="MFCC coefficients (13, time)")
    delta: np.ndarray = Field(..., description="First-order delta (13, time)")
    delta_delta: np.ndarray = Field(..., description="Second-order delta (13, time)")

    # Combined feature matrix (39, time)
    combined: Optional[np.ndarray] = Field(None, description="Concatenated MFCCs (39, time)")

    # Statistics
    mean: Optional[np.ndarray] = Field(None, description="Mean per coefficient (39,)")
    std: Optional[np.ndarray] = Field(None, description="Std per coefficient (39,)")

    @field_validator("mfcc", "delta", "delta_delta")
    @classmethod
    def validate_shape(cls, v):
        """Validate MFCC array shape."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be numpy array")
        if v.ndim != 2:
            raise ValueError(f"Must be 2D array, got shape {v.shape}")
        if v.shape[0] != 13:
            raise ValueError(f"Expected 13 MFCCs, got {v.shape[0]}")
        return v


class ProsodicFeatures(BaseModel):
    """Prosodic feature representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Fundamental frequency (F0)
    f0_contour: np.ndarray = Field(..., description="F0 contour (Hz) over time")
    f0_mean: float = Field(..., description="Mean F0 (Hz)")
    f0_std: float = Field(..., description="F0 standard deviation (Hz)")
    f0_range: float = Field(..., description="F0 range (max - min, Hz)")
    voicing_ratio: float = Field(..., ge=0, le=1, description="Ratio of voiced frames")

    # Energy
    energy_contour: np.ndarray = Field(..., description="Energy contour over time")
    energy_mean: float = Field(..., description="Mean energy")
    energy_std: float = Field(..., description="Energy standard deviation")

    # Speaking rate
    speaking_rate_syllables_per_sec: float = Field(
        ...,
        ge=0,
        description="Speaking rate (syllables/second)"
    )

    # Pauses
    pause_ratio: float = Field(..., ge=0, le=1, description="Ratio of pauses to speech")
    num_pauses: int = Field(..., ge=0, description="Number of detected pauses")
    mean_pause_duration: Optional[float] = Field(None, description="Mean pause duration (sec)")


class FormantFeatures(BaseModel):
    """Formant feature representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # F1, F2, F3 contours
    f1_contour: np.ndarray = Field(..., description="F1 contour (Hz)")
    f2_contour: np.ndarray = Field(..., description="F2 contour (Hz)")
    f3_contour: np.ndarray = Field(..., description="F3 contour (Hz)")

    # Statistics
    f1_mean: float = Field(..., description="Mean F1 (Hz)")
    f1_std: float = Field(..., description="F1 standard deviation")
    f2_mean: float = Field(..., description="Mean F2 (Hz)")
    f2_std: float = Field(..., description="F2 standard deviation")
    f3_mean: float = Field(..., description="Mean F3 (Hz)")
    f3_std: float = Field(..., description="F3 standard deviation")

    # Vowel space area (VSA)
    vowel_space_area: float = Field(..., ge=0, description="Vowel space area (Hz²)")

    # Formant dispersion
    formant_dispersion: Optional[float] = Field(None, description="Formant dispersion metric")


class EGeMAPSFeatures(BaseModel):
    """eGeMAPS (extended Geneva Minimalistic Acoustic Parameter Set) features."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 88-dimensional feature vector
    features: np.ndarray = Field(..., description="eGeMAPS feature vector (88,)")

    # Feature names (for interpretability)
    feature_names: Optional[list[str]] = Field(
        None,
        description="Names of the 88 features"
    )

    @field_validator("features")
    @classmethod
    def validate_features(cls, v):
        """Validate eGeMAPS feature vector."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be numpy array")
        if v.ndim != 1:
            raise ValueError(f"Must be 1D array, got shape {v.shape}")
        if v.shape[0] != 88:
            raise ValueError(f"Expected 88 features, got {v.shape[0]}")
        return v


class SpectrogramFeatures(BaseModel):
    """Spectrogram feature representation."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Log-mel spectrogram
    log_mel: np.ndarray = Field(..., description="Log-mel spectrogram (n_mels, time)")

    # Wavelet scalogram (optional)
    wavelet_scalogram: Optional[np.ndarray] = Field(
        None,
        description="Wavelet CWT scalogram (scales, time)"
    )

    # Stacked multi-channel representation
    stacked: Optional[np.ndarray] = Field(
        None,
        description="Stacked log-mel + scalogram (2, freq, time)"
    )

    # Metadata
    n_mels: int = Field(..., description="Number of mel bands")
    hop_length: int = Field(..., description="Hop length in samples")
    sr: int = Field(..., description="Sample rate (Hz)")

    @field_validator("log_mel")
    @classmethod
    def validate_log_mel(cls, v):
        """Validate log-mel spectrogram."""
        if not isinstance(v, np.ndarray):
            raise ValueError("Must be numpy array")
        if v.ndim != 2:
            raise ValueError(f"Must be 2D array, got shape {v.shape}")
        return v


class FeatureBundle(BaseModel):
    """Complete feature bundle aggregating all extracted features."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Raw waveform
    waveform: np.ndarray = Field(..., description="Preprocessed waveform")
    sample_rate: int = Field(..., description="Sample rate (Hz)")
    duration_sec: float = Field(..., description="Duration (seconds)")

    # Individual feature sets
    mfcc: MFCCFeatures = Field(..., description="MFCC features")
    prosody: ProsodicFeatures = Field(..., description="Prosodic features")
    formants: FormantFeatures = Field(..., description="Formant features")
    egemaps: EGeMAPSFeatures = Field(..., description="eGeMAPS features")
    spectrogram: SpectrogramFeatures = Field(..., description="Spectrogram features")

    # Fused features (optional, computed by FeatureFusion)
    fused_acoustic: Optional[np.ndarray] = Field(
        None,
        description="Fused acoustic feature vector (concatenated)"
    )

    # Metadata
    extraction_config: Optional[dict] = Field(None, description="Feature extraction config used")

    def get_acoustic_summary(self) -> dict:
        """Get summary of key acoustic features for reporting."""
        return {
            "speaking_rate_syllables_per_sec": round(
                self.prosody.speaking_rate_syllables_per_sec, 2
            ),
            "pitch_mean_hz": round(self.prosody.f0_mean, 1),
            "pitch_variability_hz": round(self.prosody.f0_std, 1),
            "pause_ratio": round(self.prosody.pause_ratio, 2),
            "vowel_space_area": round(self.formants.vowel_space_area, 2),
            "f1_mean_hz": round(self.formants.f1_mean, 1),
            "f2_mean_hz": round(self.formants.f2_mean, 1),
            "voicing_ratio": round(self.prosody.voicing_ratio, 2),
        }

    def to_dict(self) -> dict:
        """Convert to dictionary (for serialization)."""
        return {
            "duration_sec": self.duration_sec,
            "sample_rate": self.sample_rate,
            "mfcc_shape": self.mfcc.combined.shape if self.mfcc.combined is not None else None,
            "spectrogram_shape": self.spectrogram.log_mel.shape,
            "acoustic_summary": self.get_acoustic_summary(),
        }
