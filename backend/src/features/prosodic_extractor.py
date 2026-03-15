"""Prosodic feature extraction (F0, energy, speaking rate, pauses)."""

import logging
import numpy as np
import librosa
import torchcrepe

from src.features.schemas import ProsodicFeatures

logger = logging.getLogger(__name__)


class ProsodicExtractor:
    """Extract prosodic features using torchcrepe for F0."""

    def __init__(
        self,
        fmin: float = 50,
        fmax: float = 500,
        hop_length: int = 512,
        model_capacity: str = "full",
    ):
        """Initialize prosodic extractor."""
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.model_capacity = model_capacity

    def extract(self, waveform: np.ndarray, sr: int) -> ProsodicFeatures:
        """
        Extract prosodic features.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            ProsodicFeatures
        """
        logger.debug("Extracting prosodic features")

        # 1. Extract F0 using torchcrepe
        f0_contour, voicing_ratio = self._extract_f0_torchcrepe(waveform, sr)

        # Compute F0 statistics (only voiced frames)
        voiced_f0 = f0_contour[f0_contour > 0]
        f0_mean = float(np.mean(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_std = float(np.std(voiced_f0)) if len(voiced_f0) > 0 else 0.0
        f0_range = float(np.ptp(voiced_f0)) if len(voiced_f0) > 0 else 0.0

        # 2. Extract energy contour
        energy_contour = librosa.feature.rms(
            y=waveform, frame_length=2048, hop_length=self.hop_length
        )[0]
        energy_mean = float(np.mean(energy_contour))
        energy_std = float(np.std(energy_contour))

        # 3. Estimate speaking rate (simplified: syllable count from energy peaks)
        speaking_rate = self._estimate_speaking_rate(energy_contour, sr)

        # 4. Calculate pause ratio (simplified)
        pause_ratio, num_pauses, mean_pause_duration = self._calculate_pauses(
            energy_contour, sr
        )

        logger.info(
            f"Prosody extracted: F0={f0_mean:.1f}Hz, rate={speaking_rate:.2f} syl/s"
        )

        return ProsodicFeatures(
            f0_contour=f0_contour,
            f0_mean=f0_mean,
            f0_std=f0_std,
            f0_range=f0_range,
            voicing_ratio=voicing_ratio,
            energy_contour=energy_contour,
            energy_mean=energy_mean,
            energy_std=energy_std,
            speaking_rate_syllables_per_sec=speaking_rate,
            pause_ratio=pause_ratio,
            num_pauses=num_pauses,
            mean_pause_duration=mean_pause_duration,
        )

    def _extract_f0_torchcrepe(self, waveform: np.ndarray, sr: int) -> tuple:
        """Extract F0 using torchcrepe."""
        try:
            import torch

            audio_tensor = torch.from_numpy(waveform).unsqueeze(0).float()

            # Predict F0
            f0 = torchcrepe.predict(
                audio_tensor,
                sr,
                hop_length=self.hop_length,
                fmin=self.fmin,
                fmax=self.fmax,
                model=self.model_capacity,
                batch_size=512,
                device="cpu",  # Use CPU for compatibility
                return_periodicity=False,
            )

            f0_contour = f0.squeeze().numpy()

            # Calculate voicing ratio
            voiced_frames = np.sum(f0_contour > 0)
            voicing_ratio = voiced_frames / len(f0_contour) if len(f0_contour) > 0 else 0.0

            return f0_contour, float(voicing_ratio)

        except Exception as e:
            logger.warning(f"torchcrepe F0 extraction failed: {e}. Using librosa fallback.")
            return self._extract_f0_librosa(waveform, sr)

    def _extract_f0_librosa(self, waveform: np.ndarray, sr: int) -> tuple:
        """Fallback F0 extraction using librosa yin."""
        f0 = librosa.yin(
            waveform, fmin=self.fmin, fmax=self.fmax, sr=sr, hop_length=self.hop_length
        )

        voiced_frames = np.sum(f0 > 0)
        voicing_ratio = voiced_frames / len(f0) if len(f0) > 0 else 0.0

        return f0, float(voicing_ratio)

    def _estimate_speaking_rate(self, energy: np.ndarray, sr: int) -> float:
        """Estimate speaking rate from energy peaks (syllable count heuristic)."""
        from scipy.signal import find_peaks

        # Find peaks in energy contour
        peaks, _ = find_peaks(energy, height=np.percentile(energy, 40))

        # Estimate duration
        duration_sec = (len(energy) * self.hop_length) / sr

        # Speaking rate = peaks / duration
        speaking_rate = len(peaks) / duration_sec if duration_sec > 0 else 0.0

        return float(speaking_rate)

    def _calculate_pauses(self, energy: np.ndarray, sr: int) -> tuple:
        """Calculate pause statistics from energy."""
        # Threshold for silence
        threshold = np.percentile(energy, 20)  # Bottom 20% is considered silence

        # Find silence frames
        silence_frames = energy < threshold

        # Count pauses (consecutive silence frames)
        pauses = []
        in_pause = False
        pause_start = 0

        for i, is_silent in enumerate(silence_frames):
            if is_silent and not in_pause:
                in_pause = True
                pause_start = i
            elif not is_silent and in_pause:
                in_pause = False
                pause_duration = (i - pause_start) * self.hop_length / sr
                if pause_duration > 0.2:  # Minimum 0.2s to count as pause
                    pauses.append(pause_duration)

        num_pauses = len(pauses)
        pause_ratio = float(np.sum(silence_frames) / len(energy)) if len(energy) > 0 else 0.0
        mean_pause_duration = float(np.mean(pauses)) if pauses else None

        return pause_ratio, num_pauses, mean_pause_duration
