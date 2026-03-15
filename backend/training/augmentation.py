"""Audio data augmentation for training."""

import numpy as np
import librosa


class AudioAugmenter:
    """Apply audio augmentations for data diversity."""

    def __init__(
        self,
        time_stretch_range=(0.8, 1.2),
        pitch_shift_range=(-2, 2),
        noise_level_range=(0.005, 0.015),
        apply_prob=0.5,
    ):
        """
        Initialize augmenter.

        Args:
            time_stretch_range: (min_rate, max_rate) for time stretching
            pitch_shift_range: (min_steps, max_steps) for pitch shifting
            noise_level_range: (min_level, max_level) for additive noise
            apply_prob: Probability of applying each augmentation
        """
        self.time_stretch_range = time_stretch_range
        self.pitch_shift_range = pitch_shift_range
        self.noise_level_range = noise_level_range
        self.apply_prob = apply_prob

    def augment(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """
        Apply random augmentations.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            Augmented waveform
        """
        # Time stretching
        if np.random.rand() < self.apply_prob:
            rate = np.random.uniform(*self.time_stretch_range)
            waveform = librosa.effects.time_stretch(waveform, rate=rate)

        # Pitch shifting
        if np.random.rand() < self.apply_prob:
            n_steps = np.random.uniform(*self.pitch_shift_range)
            waveform = librosa.effects.pitch_shift(waveform, sr=sr, n_steps=n_steps)

        # Additive noise
        if np.random.rand() < self.apply_prob:
            noise_level = np.random.uniform(*self.noise_level_range)
            noise = np.random.randn(len(waveform)) * noise_level
            waveform = waveform + noise

        # SpecAugment (applied at spectrogram level, not here)

        return waveform
