"""eGeMAPS feature extraction using OpenSMILE."""

import logging
import numpy as np
import opensmile

from src.features.schemas import EGeMAPSFeatures

logger = logging.getLogger(__name__)


class EGeMAPSExtractor:
    """Extract eGeMAPS features using OpenSMILE."""

    def __init__(self, feature_set: str = "eGeMAPSv02", feature_level: str = "Functionals"):
        """
        Initialize eGeMAPS extractor.

        Args:
            feature_set: Feature set name (eGeMAPSv02, GeMAPSv01b, etc.)
            feature_level: Feature level (Functionals or LowLevelDescriptors)
        """
        self.feature_set = feature_set
        self.feature_level = feature_level

        try:
            self.smile = opensmile.Smile(
                feature_set=opensmile.FeatureSet[feature_set],
                feature_level=opensmile.FeatureLevel[feature_level],
            )
            logger.info(f"OpenSMILE initialized: {feature_set}/{feature_level}")
        except Exception as e:
            logger.warning(f"OpenSMILE initialization failed: {e}")
            self.smile = None

    def extract(self, waveform: np.ndarray, sr: int) -> EGeMAPSFeatures:
        """
        Extract eGeMAPS features.

        Args:
            waveform: Audio waveform
            sr: Sample rate

        Returns:
            EGeMAPSFeatures with 88-dimensional feature vector
        """
        logger.debug("Extracting eGeMAPS features")

        if self.smile is None:
            logger.warning("OpenSMILE not available, using placeholder features")
            return self._placeholder_features()

        try:
            # Process audio
            features_df = self.smile.process_signal(waveform, sr)

            # Extract feature values (first row, all columns)
            features = features_df.iloc[0].values.astype(np.float32)

            # Get feature names
            feature_names = list(features_df.columns)

            # Ensure 88 dimensions (eGeMAPSv02 Functionals)
            if len(features) != 88:
                logger.warning(
                    f"Expected 88 eGeMAPS features, got {len(features)}. Padding/truncating."
                )
                if len(features) < 88:
                    features = np.pad(features, (0, 88 - len(features)), mode="constant")
                else:
                    features = features[:88]

            logger.info(f"Extracted eGeMAPS features: {len(features)} dims")

            return EGeMAPSFeatures(features=features, feature_names=feature_names)

        except Exception as e:
            logger.error(f"eGeMAPS extraction failed: {e}")
            return self._placeholder_features()

    def _placeholder_features(self) -> EGeMAPSFeatures:
        """Return placeholder eGeMAPS features."""
        # Return zeros as placeholder
        features = np.zeros(88, dtype=np.float32)
        feature_names = [f"egemaps_{i}" for i in range(88)]

        logger.warning("Using placeholder eGeMAPS features (zeros)")

        return EGeMAPSFeatures(features=features, feature_names=feature_names)
