"""MFCC feature extraction."""

import logging
import numpy as np
import librosa

from src.features.schemas import MFCCFeatures

logger = logging.getLogger(__name__)


class MFCCExtractor:
    """Extract MFCC features with delta and delta-delta."""

    def __init__(
        self,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0,
        fmax: float = 8000,
        delta_width: int = 9,
    ):
        """Initialize MFCC extractor."""
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.delta_width = delta_width

    def extract(self, waveform: np.ndarray, sr: int) -> MFCCFeatures:
        """
        Extract MFCC features.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            MFCCFeatures with 39-dimensional feature vectors
        """
        logger.debug(f"Extracting MFCCs: n_mfcc={self.n_mfcc}")

        # Extract MFCCs
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax,
        )

        # Compute deltas
        delta = librosa.feature.delta(mfcc, width=self.delta_width)
        delta_delta = librosa.feature.delta(mfcc, order=2, width=self.delta_width)

        # Combine: (13, time) + (13, time) + (13, time) = (39, time)
        combined = np.vstack([mfcc, delta, delta_delta])

        # Compute statistics
        mean = np.mean(combined, axis=1)
        std = np.std(combined, axis=1)

        logger.info(f"Extracted MFCCs: shape={combined.shape}")

        return MFCCFeatures(
            mfcc=mfcc,
            delta=delta,
            delta_delta=delta_delta,
            combined=combined,
            mean=mean,
            std=std,
        )
