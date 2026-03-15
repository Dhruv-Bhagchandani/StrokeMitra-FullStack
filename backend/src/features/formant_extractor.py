"""Formant feature extraction using Praat (parselmouth)."""

import logging
import numpy as np
import parselmouth

from src.features.schemas import FormantFeatures

logger = logging.getLogger(__name__)


class FormantExtractor:
    """Extract formant features (F1, F2, F3) using Praat."""

    def __init__(
        self,
        max_num_formants: int = 5,
        ceiling_hz: float = 5500,
        window_length: float = 0.025,
        pre_emphasis: float = 0.97,
    ):
        """Initialize formant extractor."""
        self.max_num_formants = max_num_formants
        self.ceiling_hz = ceiling_hz
        self.window_length = window_length
        self.pre_emphasis = pre_emphasis

    def extract(self, waveform: np.ndarray, sr: int) -> FormantFeatures:
        """
        Extract formant features.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            FormantFeatures with F1, F2, F3 statistics
        """
        logger.debug("Extracting formants using Praat")

        try:
            # Create Praat Sound object
            sound = parselmouth.Sound(waveform, sampling_frequency=sr)

            # Extract formants
            formants = sound.to_formant_burg(
                time_step=0.01,
                max_number_of_formants=self.max_num_formants,
                maximum_formant=self.ceiling_hz,
                window_length=self.window_length,
                pre_emphasis_from=50.0,
            )

            # Extract F1, F2, F3 contours
            f1_contour = []
            f2_contour = []
            f3_contour = []

            for time in np.arange(0, sound.duration, 0.01):
                f1 = formants.get_value_at_time(1, time)
                f2 = formants.get_value_at_time(2, time)
                f3 = formants.get_value_at_time(3, time)

                # Filter out undefined values
                if f1 is not None and not np.isnan(f1):
                    f1_contour.append(f1)
                if f2 is not None and not np.isnan(f2):
                    f2_contour.append(f2)
                if f3 is not None and not np.isnan(f3):
                    f3_contour.append(f3)

            # Convert to arrays
            f1_contour = np.array(f1_contour) if f1_contour else np.array([0.0])
            f2_contour = np.array(f2_contour) if f2_contour else np.array([0.0])
            f3_contour = np.array(f3_contour) if f3_contour else np.array([0.0])

            # Compute statistics
            f1_mean = float(np.mean(f1_contour))
            f1_std = float(np.std(f1_contour))
            f2_mean = float(np.mean(f2_contour))
            f2_std = float(np.std(f2_contour))
            f3_mean = float(np.mean(f3_contour))
            f3_std = float(np.std(f3_contour))

            # Compute vowel space area (VSA) - simplified using F1 and F2
            vowel_space_area = self._compute_vsa(f1_contour, f2_contour)

            # Formant dispersion
            formant_dispersion = float(np.mean([f1_mean, f2_mean, f3_mean]))

            logger.info(
                f"Formants extracted: F1={f1_mean:.0f}Hz, F2={f2_mean:.0f}Hz, VSA={vowel_space_area:.0f}"
            )

            return FormantFeatures(
                f1_contour=f1_contour,
                f2_contour=f2_contour,
                f3_contour=f3_contour,
                f1_mean=f1_mean,
                f1_std=f1_std,
                f2_mean=f2_mean,
                f2_std=f2_std,
                f3_mean=f3_mean,
                f3_std=f3_std,
                vowel_space_area=vowel_space_area,
                formant_dispersion=formant_dispersion,
            )

        except Exception as e:
            logger.error(f"Formant extraction failed: {e}")
            # Return default values
            return self._default_formants()

    def _compute_vsa(self, f1: np.ndarray, f2: np.ndarray) -> float:
        """Compute vowel space area (simplified triangle area)."""
        if len(f1) < 3 or len(f2) < 3:
            return 0.0

        # Use percentiles to get corner vowels (simplified)
        f1_low, f1_mid, f1_high = np.percentile(f1, [25, 50, 75])
        f2_low, f2_mid, f2_high = np.percentile(f2, [25, 50, 75])

        # Triangle area using Heron's formula (simplified)
        area = abs((f1_low - f1_high) * (f2_mid - f2_low) / 2.0)

        return float(area)

    def _default_formants(self) -> FormantFeatures:
        """Return default formant features on failure."""
        return FormantFeatures(
            f1_contour=np.array([500.0]),
            f2_contour=np.array([1500.0]),
            f3_contour=np.array([2500.0]),
            f1_mean=500.0,
            f1_std=0.0,
            f2_mean=1500.0,
            f2_std=0.0,
            f3_mean=2500.0,
            f3_std=0.0,
            vowel_space_area=0.0,
            formant_dispersion=1500.0,
        )
