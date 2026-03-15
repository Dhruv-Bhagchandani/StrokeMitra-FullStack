"""Voice Activity Detection using Silero VAD."""

import logging
import torch
import numpy as np
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VoiceActivityDetector:
    """Detect speech segments using Silero VAD v6."""

    def __init__(
        self,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 100,
        speech_pad_ms: int = 30,
    ):
        """
        Initialize VAD.

        Args:
            threshold: Speech probability threshold (0-1)
            sampling_rate: Audio sample rate
            min_speech_duration_ms: Minimum speech segment duration
            min_silence_duration_ms: Minimum silence between segments
            speech_pad_ms: Padding around speech segments
        """
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms

        # Load Silero VAD model
        try:
            self.model, utils = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                onnx=False,
            )
            self.get_speech_timestamps = utils[0]
            logger.info("Silero VAD model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load Silero VAD: {e}. Using fallback.")
            self.model = None

    def detect_speech(self, waveform: np.ndarray) -> Tuple[List[Tuple[float, float]], float]:
        """
        Detect speech segments in audio.

        Args:
            waveform: Audio waveform (1D numpy array)

        Returns:
            Tuple of (speech_segments, speech_ratio)
            - speech_segments: List of (start_sec, end_sec) tuples
            - speech_ratio: Ratio of speech to total duration
        """
        if self.model is None:
            logger.warning("VAD model not available, using fallback energy-based detection")
            return self._fallback_energy_vad(waveform)

        try:
            # Convert to torch tensor
            audio_tensor = torch.from_numpy(waveform).float()

            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                self.model,
                sampling_rate=self.sampling_rate,
                threshold=self.threshold,
                min_speech_duration_ms=self.min_speech_duration_ms,
                min_silence_duration_ms=self.min_silence_duration_ms,
                speech_pad_ms=self.speech_pad_ms,
            )

            # Convert to seconds
            speech_segments = [
                (ts["start"] / self.sampling_rate, ts["end"] / self.sampling_rate)
                for ts in speech_timestamps
            ]

            # Calculate speech ratio
            total_duration = len(waveform) / self.sampling_rate
            speech_duration = sum(end - start for start, end in speech_segments)
            speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

            logger.info(
                f"VAD detected {len(speech_segments)} speech segments "
                f"(ratio: {speech_ratio:.2%})"
            )

            return speech_segments, speech_ratio

        except Exception as e:
            logger.error(f"VAD failed: {e}. Using fallback.")
            return self._fallback_energy_vad(waveform)

    def _fallback_energy_vad(
        self, waveform: np.ndarray
    ) -> Tuple[List[Tuple[float, float]], float]:
        """Fallback energy-based VAD if Silero fails."""
        import librosa

        # Use librosa's split for simple energy-based VAD
        intervals = librosa.effects.split(waveform, top_db=30)

        speech_segments = [
            (start / self.sampling_rate, end / self.sampling_rate)
            for start, end in intervals
        ]

        total_duration = len(waveform) / self.sampling_rate
        speech_duration = sum(end - start for start, end in speech_segments)
        speech_ratio = speech_duration / total_duration if total_duration > 0 else 0.0

        logger.info(f"Fallback VAD: {len(speech_segments)} segments (ratio: {speech_ratio:.2%})")

        return speech_segments, speech_ratio

    def apply_vad(self, waveform: np.ndarray) -> np.ndarray:
        """
        Apply VAD and return only speech segments concatenated.

        Args:
            waveform: Input waveform

        Returns:
            Waveform with only speech segments
        """
        speech_segments, _ = self.detect_speech(waveform)

        if not speech_segments:
            logger.warning("No speech detected, returning original waveform")
            return waveform

        # Extract and concatenate speech segments
        speech_parts = []
        for start_sec, end_sec in speech_segments:
            start_sample = int(start_sec * self.sampling_rate)
            end_sample = int(end_sec * self.sampling_rate)
            speech_parts.append(waveform[start_sample:end_sample])

        speech_only = np.concatenate(speech_parts)

        logger.info(
            f"VAD applied: {len(waveform)} → {len(speech_only)} samples "
            f"({len(speech_only) / self.sampling_rate:.2f}s)"
        )

        return speech_only
