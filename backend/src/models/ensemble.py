"""Ensemble model combining HuBERT-SALR and CNN-BiLSTM branches."""

import logging
import numpy as np

logger = logging.getLogger(__name__)


class EnsembleModel:
    """
    Ensemble model for dysarthria detection.

    For MVP: Uses placeholder branch models.
    In production: Uses trained HuBERT-SALR and CNN-BiLSTM models.
    """

    def __init__(self, alpha: float = 0.6):
        """
        Initialize ensemble.

        Args:
            alpha: Weight for HuBERT branch (1-alpha for CNN branch)
        """
        self.alpha = alpha
        self.version = "ensemble-v1.0-placeholder"

        logger.info(f"EnsembleModel initialized (alpha={alpha})")

    def predict(
        self,
        waveform: np.ndarray,
        spectrogram: np.ndarray,
        acoustic_features: np.ndarray,
    ) -> dict:
        """
        Run ensemble prediction.

        Args:
            waveform: Audio waveform for HuBERT branch
            spectrogram: Spectrogram for CNN branch
            acoustic_features: Acoustic features for fusion

        Returns:
            Dictionary with logits and probabilities
        """
        logger.debug("Running ensemble prediction (placeholder)")

        # Placeholder: Generate mock predictions from both branches
        hubert_logits = self._mock_hubert_branch(waveform)
        cnn_logits = self._mock_cnn_branch(spectrogram, acoustic_features)

        # Ensemble: weighted average of logits
        ensemble_logits = self.alpha * hubert_logits + (1 - self.alpha) * cnn_logits

        # Convert to probabilities
        exp_logits = np.exp(ensemble_logits - np.max(ensemble_logits))
        probs = exp_logits / np.sum(exp_logits)

        raw_probability = float(probs[1])  # Probability of dysarthric class

        logger.info(f"Ensemble prediction: prob_dysarthric={raw_probability:.3f}")

        return {
            "logits": ensemble_logits,
            "probabilities": probs,
            "raw_probability": raw_probability,
            "hubert_logits": hubert_logits,
            "cnn_logits": cnn_logits,
            "alpha": self.alpha,
        }

    def _mock_hubert_branch(self, waveform: np.ndarray) -> np.ndarray:
        """Mock HuBERT-SALR branch prediction."""
        # Generate somewhat realistic logits
        prob = np.random.beta(2, 5)  # Bias towards healthy
        logit_healthy = np.log((1 - prob) / (prob + 1e-8))
        logit_dysarthric = np.log(prob / (1 - prob + 1e-8))

        return np.array([logit_healthy, logit_dysarthric])

    def _mock_cnn_branch(self, spectrogram: np.ndarray, acoustic_features: np.ndarray) -> np.ndarray:
        """Mock CNN-BiLSTM-Transformer branch prediction."""
        # Generate somewhat realistic logits (slightly different from HuBERT)
        prob = np.random.beta(2.5, 5.5)  # Slightly different distribution
        logit_healthy = np.log((1 - prob) / (prob + 1e-8))
        logit_dysarthric = np.log(prob / (1 - prob + 1e-8))

        return np.array([logit_healthy, logit_dysarthric])
