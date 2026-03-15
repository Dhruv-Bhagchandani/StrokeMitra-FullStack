"""Model registry for loading models from MLflow or placeholders."""

import logging
from typing import Optional
from pathlib import Path
import numpy as np
import torch

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Model registry for loading trained models or placeholders.

    For MVP: Returns placeholder models that generate mock predictions.
    In production: Load real models from MLflow.
    """

    def __init__(self, use_placeholder: bool = True):
        """
        Initialize model registry.

        Args:
            use_placeholder: If True, return placeholder models (for MVP)
        """
        self.use_placeholder = use_placeholder
        logger.info(f"ModelRegistry initialized (placeholder={use_placeholder})")

    def load_ensemble(self, version: str = "latest") -> "PlaceholderEnsemble":
        """
        Load ensemble model.

        Args:
            version: Model version

        Returns:
            Ensemble model (trained HuBERT or placeholder)
        """
        if self.use_placeholder:
            logger.info("Loading placeholder ensemble model")
            return PlaceholderEnsemble()
        else:
            logger.info("Loading trained HuBERT model")
            return TrainedHuBERTEnsemble()

    def load_calibration(self, version: str = "latest") -> "PlaceholderCalibration":
        """
        Load calibration parameters.

        Args:
            version: Calibration version

        Returns:
            Calibration object (placeholder for MVP)
        """
        if self.use_placeholder:
            logger.info("Loading placeholder calibration")
            return PlaceholderCalibration()
        else:
            logger.warning("MLflow calibration loading not implemented, using placeholder")
            return PlaceholderCalibration()


class PlaceholderEnsemble:
    """Placeholder ensemble model that returns mock predictions."""

    def __init__(self, seed: int = 42):
        """Initialize placeholder model."""
        self.seed = seed
        self.version = "placeholder-v1.0"
        np.random.seed(seed)

    def predict(self, waveform: np.ndarray, spectrogram: np.ndarray, acoustic_features: np.ndarray) -> dict:
        """
        Generate mock prediction.

        Args:
            waveform: Audio waveform (not used in placeholder)
            spectrogram: Spectrogram features (not used in placeholder)
            acoustic_features: Acoustic features (not used in placeholder)

        Returns:
            Dictionary with logits and probabilities
        """
        # Generate random but realistic-looking predictions
        # Bias towards "healthy" (non-dysarthric) for testing
        prob_dysarthric = np.random.beta(2, 5)  # Beta distribution, mean ~0.29

        logit_healthy = np.log((1 - prob_dysarthric) / (prob_dysarthric + 1e-8))
        logit_dysarthric = np.log(prob_dysarthric / (1 - prob_dysarthric + 1e-8))

        logits = np.array([logit_healthy, logit_dysarthric])
        probs = np.array([1 - prob_dysarthric, prob_dysarthric])

        logger.debug(f"Placeholder prediction: prob_dysarthric={prob_dysarthric:.3f}")

        return {
            "logits": logits,
            "probabilities": probs,
            "raw_probability": float(prob_dysarthric),
        }


class TrainedHuBERTEnsemble:
    """Real trained HuBERT model for dysarthria detection."""

    def __init__(self, checkpoint_path: str = "models/hubert_fast_best.pt"):
        """
        Initialize with trained checkpoint.

        Args:
            checkpoint_path: Path to trained model checkpoint
        """
        from training.train_hubert_fast import SimplifiedHuBERTClassifier

        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")

        # Detect device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        # Load model
        logger.info(f"Loading trained HuBERT model from {checkpoint_path} on {self.device}")
        self.model = SimplifiedHuBERTClassifier(freeze_base=True).to(self.device)

        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

        self.version = f"hubert-fast-epoch{checkpoint['epoch']}-auc{checkpoint['val_auc']:.4f}"
        logger.info(f"✓ Loaded trained model: {self.version}")

    def predict(self, waveform: np.ndarray, spectrogram: np.ndarray, acoustic_features: np.ndarray) -> dict:
        """
        Generate prediction using trained model.

        Args:
            waveform: Audio waveform (1D numpy array)
            spectrogram: Spectrogram features (not used by HuBERT)
            acoustic_features: Acoustic features (not used by HuBERT)

        Returns:
            Dictionary with logits and probabilities
        """
        # Prepare input (pad or truncate to 10 seconds)
        target_length = 16000 * 10
        if len(waveform) > target_length:
            waveform = waveform[:target_length]
        else:
            waveform = np.pad(waveform, (0, target_length - len(waveform)))

        # Convert to tensor
        waveform_tensor = torch.from_numpy(waveform).float().unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            logits = self.model(waveform_tensor)
            probs = torch.softmax(logits, dim=1)

        # Convert to numpy
        logits_np = logits.cpu().numpy()[0]
        probs_np = probs.cpu().numpy()[0]

        logger.debug(f"Trained model prediction: prob_dysarthric={probs_np[1]:.3f}")

        return {
            "logits": logits_np,
            "probabilities": probs_np,
            "raw_probability": float(probs_np[1]),
        }


class PlaceholderCalibration:
    """Placeholder calibration (identity transform for testing)."""

    def transform(self, logits: np.ndarray) -> float:
        """
        Apply calibration to logits.

        Args:
            logits: Model logits [healthy, dysarthric]

        Returns:
            Calibrated probability of dysarthria
        """
        # Simple softmax for placeholder
        exp_logits = np.exp(logits - np.max(logits))
        probs = exp_logits / np.sum(exp_logits)

        calibrated_prob = float(probs[1])  # Probability of dysarthric class

        logger.debug(f"Placeholder calibration: {calibrated_prob:.3f}")

        return calibrated_prob
