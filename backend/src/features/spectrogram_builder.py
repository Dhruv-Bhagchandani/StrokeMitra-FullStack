"""Spectrogram and scalogram generation."""

import logging
import numpy as np
import librosa
from scipy import signal

from src.features.schemas import SpectrogramFeatures

logger = logging.getLogger(__name__)


class SpectrogramBuilder:
    """Build log-mel spectrogram and wavelet scalogram."""

    def __init__(
        self,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        fmin: float = 0,
        fmax: float = 8000,
        use_wavelet: bool = True,
        wavelet_type: str = "morl",
        num_scales: int = 128,
    ):
        """Initialize spectrogram builder."""
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fmin = fmin
        self.fmax = fmax
        self.use_wavelet = use_wavelet
        self.wavelet_type = wavelet_type
        self.num_scales = num_scales

    def build(self, waveform: np.ndarray, sr: int) -> SpectrogramFeatures:
        """
        Build spectrogram features.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            SpectrogramFeatures with log-mel and optional wavelet scalogram
        """
        logger.debug(f"Building spectrogram: n_mels={self.n_mels}")

        # 1. Compute log-mel spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
            power=2.0,
        )

        # Convert to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # 2. Compute wavelet scalogram (optional)
        wavelet_scalogram = None
        if self.use_wavelet:
            try:
                wavelet_scalogram = self._compute_wavelet_scalogram(waveform, sr)
            except Exception as e:
                logger.warning(f"Wavelet scalogram computation failed: {e}")

        # 3. Stack log-mel + scalogram as multi-channel input
        stacked = None
        if wavelet_scalogram is not None:
            # Resize scalogram to match log-mel dimensions
            if wavelet_scalogram.shape != log_mel.shape:
                from scipy.ndimage import zoom

                zoom_factors = (
                    log_mel.shape[0] / wavelet_scalogram.shape[0],
                    log_mel.shape[1] / wavelet_scalogram.shape[1],
                )
                wavelet_scalogram = zoom(wavelet_scalogram, zoom_factors, order=1)

            # Stack as (2, freq, time)
            stacked = np.stack([log_mel, wavelet_scalogram], axis=0)
        else:
            # Use only log-mel, duplicated
            stacked = np.stack([log_mel, log_mel], axis=0)

        logger.info(f"Spectrogram built: log_mel={log_mel.shape}, stacked={stacked.shape}")

        return SpectrogramFeatures(
            log_mel=log_mel,
            wavelet_scalogram=wavelet_scalogram,
            stacked=stacked,
            n_mels=self.n_mels,
            hop_length=self.hop_length,
            sr=sr,
        )

    def _compute_wavelet_scalogram(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Compute wavelet CWT scalogram."""
        # Define scales
        scales = np.arange(1, self.num_scales + 1)

        # Compute CWT
        coefficients, frequencies = signal.cwt(
            waveform, signal.morlet2, scales, w=5.0
        )

        # Convert to power (magnitude squared)
        scalogram = np.abs(coefficients) ** 2

        # Log scale
        scalogram = np.log1p(scalogram)

        # Normalize
        scalogram = (scalogram - scalogram.min()) / (scalogram.max() - scalogram.min() + 1e-8)

        return scalogram
