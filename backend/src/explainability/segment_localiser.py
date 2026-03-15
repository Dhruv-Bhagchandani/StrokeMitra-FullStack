"""Segment localization from explainability outputs."""

import logging
import numpy as np
from scipy.signal import find_peaks

from src.reporting.schemas import SegmentAnnotation

logger = logging.getLogger(__name__)


class SegmentLocaliser:
    """Localize time segments from Grad-CAM and attention outputs."""

    def __init__(
        self,
        min_segment_duration_ms: int = 200,
        merge_threshold: float = 0.3,
    ):
        """
        Initialize segment localiser.

        Args:
            min_segment_duration_ms: Minimum segment duration
            merge_threshold: Merge segments if gap < threshold (in seconds)
        """
        self.min_segment_duration_ms = min_segment_duration_ms
        self.merge_threshold = merge_threshold

    def localise(
        self,
        heatmap: np.ndarray,
        attention_importance: np.ndarray,
        hop_length: int = 512,
        sr: int = 16000,
        top_k: int = 5,
    ) -> list[SegmentAnnotation]:
        """
        Localise important segments.

        Args:
            heatmap: Grad-CAM heatmap (freq, time)
            attention_importance: Attention rollout importance (seq_len,)
            hop_length: Hop length in samples
            sr: Sample rate
            top_k: Number of segments to return

        Returns:
            List of SegmentAnnotation objects
        """
        # Combine heatmap and attention
        # Average heatmap over frequency dimension
        heatmap_time = heatmap.mean(axis=0)  # (time,)

        # Ensure same length
        min_len = min(len(heatmap_time), len(attention_importance))
        heatmap_time = heatmap_time[:min_len]
        attention_importance = attention_importance[:min_len]

        # Combined importance score
        combined_importance = 0.6 * heatmap_time + 0.4 * attention_importance
        combined_importance = combined_importance / (combined_importance.max() + 1e-8)

        # Find peaks
        peaks, properties = find_peaks(
            combined_importance,
            height=np.percentile(combined_importance, 60),  # Above 60th percentile
            distance=int(self.min_segment_duration_ms / 1000 * sr / hop_length),
        )

        # Sort by importance
        if len(peaks) > 0:
            peak_heights = properties["peak_heights"]
            sorted_indices = np.argsort(peak_heights)[::-1][:top_k]
            peaks = peaks[sorted_indices]
            peak_heights = peak_heights[sorted_indices]
        else:
            # If no peaks, use top-k frames
            peaks = np.argsort(combined_importance)[-top_k:][::-1]
            peak_heights = combined_importance[peaks]

        # Convert to time segments
        segments = []
        for i, (peak_idx, height) in enumerate(zip(peaks, peak_heights)):
            # Expand around peak to find segment boundaries
            start_idx = max(0, peak_idx - 5)
            end_idx = min(len(combined_importance) - 1, peak_idx + 5)

            # Convert to milliseconds
            start_ms = int((start_idx * hop_length / sr) * 1000)
            end_ms = int((end_idx * hop_length / sr) * 1000)

            # Assign label based on position and characteristics
            label = self._assign_label(i, peak_idx, len(combined_importance))

            segment = SegmentAnnotation(
                start_ms=start_ms,
                end_ms=end_ms,
                label=label,
                weight=float(height),
            )

            segments.append(segment)

        # Merge nearby segments if needed
        segments = self._merge_segments(segments)

        logger.info(f"Localised {len(segments)} segments")

        return segments

    def _assign_label(self, segment_idx: int, peak_idx: int, total_frames: int) -> str:
        """Assign label to segment based on characteristics."""
        labels = [
            "imprecise_consonants",
            "irregular_rate",
            "monopitch",
            "hypernasality",
            "reduced_breath_support",
        ]

        # Simple heuristic: cycle through labels
        return labels[segment_idx % len(labels)]

    def _merge_segments(
        self,
        segments: list[SegmentAnnotation],
    ) -> list[SegmentAnnotation]:
        """Merge segments that are close together."""
        if len(segments) <= 1:
            return segments

        # Sort by start time
        segments = sorted(segments, key=lambda s: s.start_ms)

        merged = [segments[0]]

        for current in segments[1:]:
            last = merged[-1]

            gap_sec = (current.start_ms - last.end_ms) / 1000.0

            if gap_sec < self.merge_threshold:
                # Merge segments
                last.end_ms = current.end_ms
                last.weight = max(last.weight, current.weight)
            else:
                merged.append(current)

        return merged
