"""Audio quality validation and assessment."""

import logging
import numpy as np

from src.ingestion.schemas import QualityMetrics

logger = logging.getLogger(__name__)


class QualityChecker:
    """Check audio quality and validate for processing."""

    def __init__(
        self,
        min_duration: float = 5.0,
        max_duration: float = 60.0,
        min_snr_db: float = 10.0,
        max_clipping_ratio: float = 0.01,
    ):
        """
        Initialize quality checker.

        Args:
            min_duration: Minimum acceptable duration (seconds)
            max_duration: Maximum acceptable duration (seconds)
            min_snr_db: Minimum signal-to-noise ratio (dB)
            max_clipping_ratio: Maximum acceptable clipping ratio
        """
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_snr_db = min_snr_db
        self.max_clipping_ratio = max_clipping_ratio

    def check(self, waveform: np.ndarray, sr: int, duration: float) -> QualityMetrics:
        """
        Assess audio quality.

        Args:
            waveform: Audio waveform
            sr: Sample rate
            duration: Duration in seconds

        Returns:
            QualityMetrics with validation results
        """
        logger.info("Checking audio quality...")

        quality_issues = []
        is_valid = True

        # 1. Check duration
        if duration < self.min_duration:
            quality_issues.append(f"Duration too short: {duration:.1f}s < {self.min_duration}s")
            is_valid = False

        if duration > self.max_duration:
            quality_issues.append(f"Duration too long: {duration:.1f}s > {self.max_duration}s")
            is_valid = False

        # 2. Estimate SNR
        snr_db = self._estimate_snr(waveform)
        if snr_db is not None and snr_db < self.min_snr_db:
            quality_issues.append(f"Low SNR: {snr_db:.1f}dB < {self.min_snr_db}dB")
            is_valid = False

        # 3. Check clipping
        clipping_ratio = self._detect_clipping(waveform)
        if clipping_ratio > self.max_clipping_ratio:
            quality_issues.append(
                f"High clipping: {clipping_ratio:.2%} > {self.max_clipping_ratio:.2%}"
            )
            is_valid = False

        # 4. Calculate peak amplitude and RMS energy
        peak_amplitude = float(np.abs(waveform).max())
        rms_energy = float(np.sqrt(np.mean(waveform**2)))

        # Check if audio is too quiet
        if peak_amplitude < 0.01:
            quality_issues.append(f"Audio too quiet: peak={peak_amplitude:.4f}")
            is_valid = False

        metrics = QualityMetrics(
            snr_db=snr_db,
            clipping_ratio=clipping_ratio,
            peak_amplitude=peak_amplitude,
            rms_energy=rms_energy,
            is_valid=is_valid,
            quality_issues=quality_issues,
            min_duration_sec=self.min_duration,
            max_duration_sec=self.max_duration,
            min_snr_db=self.min_snr_db,
            max_clipping_ratio=self.max_clipping_ratio,
        )

        if is_valid:
            logger.info("✓ Audio quality validation passed")
        else:
            logger.warning(f"✗ Audio quality issues: {', '.join(quality_issues)}")

        return metrics

    def _estimate_snr(self, waveform: np.ndarray) -> float | None:
        """
        Estimate signal-to-noise ratio (simplified method).

        Args:
            waveform: Audio waveform

        Returns:
            Estimated SNR in dB, or None if estimation fails
        """
        try:
            # Simple SNR estimation: assume top 50% energy is signal, bottom 10% is noise
            energy = waveform**2
            sorted_energy = np.sort(energy)

            # Signal: top 50% of samples by energy
            signal_threshold_idx = int(len(sorted_energy) * 0.5)
            signal_energy = sorted_energy[signal_threshold_idx:].mean()

            # Noise: bottom 10% of samples by energy
            noise_threshold_idx = int(len(sorted_energy) * 0.1)
            noise_energy = sorted_energy[:noise_threshold_idx].mean()

            if noise_energy > 0:
                snr = 10 * np.log10(signal_energy / noise_energy)
                return float(snr)
            else:
                return None

        except Exception as e:
            logger.warning(f"SNR estimation failed: {e}")
            return None

    def _detect_clipping(self, waveform: np.ndarray, threshold: float = 0.99) -> float:
        """
        Detect clipped samples.

        Args:
            waveform: Audio waveform
            threshold: Clipping threshold (absolute value)

        Returns:
            Ratio of clipped samples to total samples
        """
        clipped = np.abs(waveform) >= threshold
        clipping_ratio = float(clipped.sum() / len(waveform))
        return clipping_ratio

    def validate_or_raise(self, waveform: np.ndarray, sr: int, duration: float):
        """
        Check quality and raise ValueError if invalid.

        Args:
            waveform: Audio waveform
            sr: Sample rate
            duration: Duration in seconds

        Raises:
            ValueError: If audio quality is invalid
        """
        metrics = self.check(waveform, sr, duration)

        if not metrics.is_valid:
            raise ValueError(
                f"Audio quality validation failed: {', '.join(metrics.quality_issues)}"
            )
