"""Attention rollout for transformer models."""

import logging
import numpy as np
import torch

logger = logging.getLogger(__name__)


class AttentionRollout:
    """
    Attention rollout for visualizing transformer attention patterns.

    Computes the flow of information through transformer layers.
    """

    def __init__(self, model, num_layers=24, num_heads=16):
        """
        Initialize attention rollout.

        Args:
            model: Transformer model (e.g., HuBERT)
            num_layers: Number of transformer layers
            num_heads: Number of attention heads
        """
        self.model = model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.attention_maps = []

        self._register_hooks()

    def _register_hooks(self):
        """Register hooks to capture attention weights."""
        def attention_hook(module, input, output):
            # Capture attention weights
            # output format: (batch, heads, seq_len, seq_len)
            if hasattr(output, "attentions") and output.attentions is not None:
                self.attention_maps.append(output.attentions.detach())

        # Register hooks on transformer attention layers
        # This is model-specific; adjust for your architecture
        for name, module in self.model.named_modules():
            if "attention" in name.lower():
                module.register_forward_hook(attention_hook)

        logger.info("Attention hooks registered")

    def compute_rollout(
        self,
        input_tensor: torch.Tensor,
        head_fusion: str = "mean",
    ) -> np.ndarray:
        """
        Compute attention rollout.

        Args:
            input_tensor: Model input
            head_fusion: How to fuse attention heads ('mean', 'max', 'min')

        Returns:
            Attention rollout array (seq_len,) showing frame importance
        """
        self.attention_maps = []
        self.model.eval()

        with torch.no_grad():
            _ = self.model(input_tensor)

        if len(self.attention_maps) == 0:
            logger.warning("No attention maps captured. Using uniform attention.")
            seq_len = input_tensor.shape[1] if input_tensor.ndim > 1 else 100
            return np.ones(seq_len) / seq_len

        # Fuse attention heads
        fused_attentions = []
        for attn in self.attention_maps:
            # attn: (batch, heads, seq_len, seq_len)
            if head_fusion == "mean":
                fused = attn.mean(dim=1)  # (batch, seq_len, seq_len)
            elif head_fusion == "max":
                fused = attn.max(dim=1)[0]
            elif head_fusion == "min":
                fused = attn.min(dim=1)[0]
            else:
                fused = attn.mean(dim=1)

            fused_attentions.append(fused.squeeze(0).cpu().numpy())  # (seq_len, seq_len)

        # Multiply attention across layers (rollout)
        rollout = np.eye(fused_attentions[0].shape[0])  # Identity matrix
        for attn in fused_attentions:
            rollout = rollout @ attn

        # Average attention to each position
        importance = rollout.mean(axis=0)  # (seq_len,)

        # Normalize
        importance = importance / (importance.sum() + 1e-8)

        logger.info(f"Attention rollout computed: {len(fused_attentions)} layers")

        return importance

    def get_top_k_frames(
        self,
        importance: np.ndarray,
        k: int = 5,
        hop_length: int = 512,
        sr: int = 16000,
    ) -> list[tuple[float, float, float]]:
        """
        Get top-k most important time frames.

        Args:
            importance: Frame importance scores (seq_len,)
            k: Number of top frames to return
            hop_length: Hop length used in feature extraction
            sr: Sample rate

        Returns:
            List of (start_sec, end_sec, importance_score)
        """
        # Get top k indices
        top_indices = np.argsort(importance)[-k:][::-1]

        segments = []
        for idx in top_indices:
            start_sec = (idx * hop_length) / sr
            end_sec = ((idx + 1) * hop_length) / sr
            score = float(importance[idx])

            segments.append((start_sec, end_sec, score))

        return segments
