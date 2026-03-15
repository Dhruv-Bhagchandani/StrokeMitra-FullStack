"""Feature fusion and normalization."""

import logging
import numpy as np

from src.features.schemas import FeatureBundle

logger = logging.getLogger(__name__)


class FeatureFusion:
    """Fuse and normalize all extracted features."""

    def __init__(
        self,
        normalize: bool = True,
        normalization_method: str = "standard",
    ):
        """
        Initialize feature fusion.

        Args:
            normalize: Whether to normalize features
            normalization_method: 'standard' (z-score) or 'minmax'
        """
        self.normalize = normalize
        self.normalization_method = normalization_method

    def fuse(self, feature_bundle: FeatureBundle) -> FeatureBundle:
        """
        Fuse all features into a single vector.

        Args:
            feature_bundle: FeatureBundle with individual features

        Returns:
            Updated FeatureBundle with fused_acoustic field
        """
        logger.debug("Fusing acoustic features")

        # Extract individual features as vectors
        features_to_fuse = []

        # 1. MFCC statistics (39 dims: mean of combined MFCCs)
        if feature_bundle.mfcc.mean is not None:
            features_to_fuse.append(feature_bundle.mfcc.mean)

        # 2. Prosodic features (extract scalars)
        prosody_vector = np.array([
            feature_bundle.prosody.f0_mean,
            feature_bundle.prosody.f0_std,
            feature_bundle.prosody.f0_range,
            feature_bundle.prosody.voicing_ratio,
            feature_bundle.prosody.energy_mean,
            feature_bundle.prosody.energy_std,
            feature_bundle.prosody.speaking_rate_syllables_per_sec,
            feature_bundle.prosody.pause_ratio,
            feature_bundle.prosody.num_pauses,
            feature_bundle.prosody.mean_pause_duration or 0.0,
        ])
        features_to_fuse.append(prosody_vector)

        # 3. Formant features (extract scalars)
        formant_vector = np.array([
            feature_bundle.formants.f1_mean,
            feature_bundle.formants.f1_std,
            feature_bundle.formants.f2_mean,
            feature_bundle.formants.f2_std,
            feature_bundle.formants.f3_mean,
            feature_bundle.formants.f3_std,
            feature_bundle.formants.vowel_space_area,
            feature_bundle.formants.formant_dispersion or 0.0,
        ])
        features_to_fuse.append(formant_vector)

        # 4. eGeMAPS features (88 dims)
        features_to_fuse.append(feature_bundle.egemaps.features)

        # Concatenate all features
        fused_acoustic = np.concatenate(features_to_fuse)

        # Normalize if requested
        if self.normalize:
            fused_acoustic = self._normalize(fused_acoustic)

        logger.info(f"Fused acoustic features: {fused_acoustic.shape[0]} dims")

        # Update feature bundle
        feature_bundle.fused_acoustic = fused_acoustic

        return feature_bundle

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Normalize feature vector."""
        if self.normalization_method == "standard":
            # Z-score normalization
            mean = np.mean(features)
            std = np.std(features)
            if std > 0:
                return (features - mean) / std
            else:
                return features

        elif self.normalization_method == "minmax":
            # Min-max normalization to [0, 1]
            min_val = np.min(features)
            max_val = np.max(features)
            if max_val > min_val:
                return (features - min_val) / (max_val - min_val)
            else:
                return features

        else:
            return features
