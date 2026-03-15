"""Audio preprocessing: resampling, normalization, silence trimming."""

import logging
import numpy as np
import librosa
import pyloudnorm as pyln

from src.ingestion.schemas import PreprocessedAudio

logger = logging.getLogger(__name__)


class AudioPreprocessor:
    """Preprocess audio for feature extraction."""

    def __init__(
        self,
        target_sr: int = 16000,
        normalize_loudness: bool = True,
        target_loudness: float = -23.0,
        trim_silence: bool = True,
        top_db: float = 30.0,
    ):
        """
        Initialize preprocessor.

        Args:
            target_sr: Target sample rate (Hz)
            normalize_loudness: Apply loudness normalization (ITU-R BS.1770)
            target_loudness: Target loudness in LUFS
            trim_silence: Trim leading/trailing silence
            top_db: Threshold for silence trimming (dB)
        """
        self.target_sr = target_sr
        self.normalize_loudness = normalize_loudness
        self.target_loudness = target_loudness
        self.trim_silence = trim_silence
        self.top_db = top_db

        # Initialize loudness meter
        if normalize_loudness:
            self.meter = pyln.Meter(target_sr)

    def process(
        self, waveform: np.ndarray, sr: int, original_duration: float
    ) -> PreprocessedAudio:
        """
        Preprocess audio waveform.

        Args:
            waveform: Input waveform
            sr: Input sample rate
            original_duration: Original duration (seconds)

        Returns:
            PreprocessedAudio with preprocessed waveform
        """
        logger.info(f"Preprocessing audio: {len(waveform)} samples at {sr}Hz")

        processed_waveform = waveform.copy()
        resampled = False
        normalized = False
        trimmed = False

        # 1. Resample to target sample rate
        if sr != self.target_sr:
            logger.debug(f"Resampling from {sr}Hz to {self.target_sr}Hz")
            processed_waveform = librosa.resample(
                processed_waveform, orig_sr=sr, target_sr=self.target_sr
            )
            sr = self.target_sr
            resampled = True

        # 2. Normalize loudness (ITU-R BS.1770)
        if self.normalize_loudness:
            try:
                # Measure loudness
                loudness = self.meter.integrated_loudness(processed_waveform)
                logger.debug(f"Original loudness: {loudness:.2f} LUFS")

                # Normalize to target loudness
                processed_waveform = pyln.normalize.loudness(
                    processed_waveform, loudness, self.target_loudness
                )
                normalized = True
                logger.debug(f"Normalized to {self.target_loudness} LUFS")
            except Exception as e:
                logger.warning(f"Loudness normalization failed: {e}. Skipping.")

        # 3. Trim silence
        if self.trim_silence:
            try:
                processed_waveform, _ = librosa.effects.trim(
                    processed_waveform, top_db=self.top_db
                )
                trimmed = True
                logger.debug(
                    f"Trimmed silence: {len(waveform)} → {len(processed_waveform)} samples"
                )
            except Exception as e:
                logger.warning(f"Silence trimming failed: {e}. Skipping.")

        # Calculate final duration
        duration_sec = len(processed_waveform) / sr

        logger.info(
            f"Preprocessing complete: {duration_sec:.2f}s "
            f"(resampled={resampled}, normalized={normalized}, trimmed={trimmed})"
        )

        return PreprocessedAudio(
            waveform=processed_waveform,
            sample_rate=sr,
            duration_sec=duration_sec,
            original_duration_sec=original_duration,
            resampled=resampled,
            normalized=normalized,
            trimmed=trimmed,
            vad_applied=False,  # VAD is applied separately
        )

    @staticmethod
    def ensure_mono(waveform: np.ndarray) -> np.ndarray:
        """Convert stereo to mono if needed."""
        if waveform.ndim == 2:
            return np.mean(waveform, axis=0)
        return waveform

    @staticmethod
    def normalize_amplitude(waveform: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
        """Normalize amplitude to target peak."""
        peak = np.abs(waveform).max()
        if peak > 0:
            return waveform * (target_peak / peak)
        return waveform
