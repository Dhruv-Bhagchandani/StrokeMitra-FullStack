"""Grad-CAM for CNN spectrogram branch."""

import logging
import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Highlights which regions of the spectrogram contributed to the prediction.
    """

    def __init__(self, model, target_layer_name="cnn.conv_blocks.2"):
        """
        Initialize Grad-CAM.

        Args:
            model: PyTorch model
            target_layer_name: Name of target convolutional layer
        """
        self.model = model
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None

        # Register hooks
        self._register_hooks()

    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        # Find target layer
        target_layer = self._find_layer(self.model, self.target_layer_name)

        if target_layer is not None:
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
            logger.info(f"Grad-CAM hooks registered on: {self.target_layer_name}")
        else:
            logger.warning(f"Target layer {self.target_layer_name} not found")

    def _find_layer(self, model, layer_name):
        """Find layer by name."""
        for name, module in model.named_modules():
            if name == layer_name:
                return module
        return None

    def generate_heatmap(
        self,
        input_tensor: torch.Tensor,
        target_class: int = None,
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            input_tensor: Input spectrogram (batch, channels, freq, time)
            target_class: Target class index (if None, use predicted class)

        Returns:
            Heatmap array (freq, time) with values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # Compute weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # Global average pooling

        # Weighted combination
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, freq, time)

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        logger.info(f"Grad-CAM heatmap generated: shape={cam.shape}")

        return cam

    def overlay_heatmap(
        self,
        spectrogram: np.ndarray,
        heatmap: np.ndarray,
        alpha: float = 0.5,
    ) -> np.ndarray:
        """
        Overlay heatmap on spectrogram.

        Args:
            spectrogram: Original spectrogram (freq, time)
            heatmap: Grad-CAM heatmap (freq, time)
            alpha: Transparency (0=spectrogram only, 1=heatmap only)

        Returns:
            Overlayed visualization
        """
        # Resize heatmap to match spectrogram if needed
        if heatmap.shape != spectrogram.shape:
            from scipy.ndimage import zoom
            zoom_factors = (
                spectrogram.shape[0] / heatmap.shape[0],
                spectrogram.shape[1] / heatmap.shape[1],
            )
            heatmap = zoom(heatmap, zoom_factors, order=1)

        # Normalize spectrogram to [0, 1]
        spec_norm = (spectrogram - spectrogram.min()) / (spectrogram.max() - spectrogram.min())

        # Blend
        overlay = alpha * heatmap + (1 - alpha) * spec_norm

        return overlay
