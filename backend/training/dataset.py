"""PyTorch Dataset for dysarthria detection."""

import logging
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

from src.ingestion.audio_loader import AudioLoader
from src.ingestion.preprocessor import AudioPreprocessor
from src.features.mfcc_extractor import MFCCExtractor
from src.features.prosodic_extractor import ProsodicExtractor
from src.features.formant_extractor import FormantExtractor
from src.features.egemaps_extractor import EGeMAPSExtractor
from src.features.spectrogram_builder import SpectrogramBuilder
from src.features.feature_fusion import FeatureFusion
from src.features.schemas import FeatureBundle

logger = logging.getLogger(__name__)


class DysarthriaDataset(Dataset):
    """Dataset for dysarthria detection with on-the-fly feature extraction."""

    def __init__(
        self,
        manifest_path: str | Path,
        augment: bool = False,
        cache_features: bool = False,
    ):
        """
        Initialize dataset.

        Args:
            manifest_path: Path to CSV manifest (filepath, label, speaker_id, duration)
            augment: Apply data augmentation
            cache_features: Cache extracted features in memory
        """
        self.manifest = pd.read_csv(manifest_path)
        self.augment = augment
        self.cache_features = cache_features
        self.feature_cache = {} if cache_features else None

        # Initialize components
        self.audio_loader = AudioLoader()
        self.preprocessor = AudioPreprocessor(target_sr=16000)
        self.mfcc_extractor = MFCCExtractor()
        self.prosodic_extractor = ProsodicExtractor()
        self.formant_extractor = FormantExtractor()
        self.egemaps_extractor = EGeMAPSExtractor()
        self.spectrogram_builder = SpectrogramBuilder()
        self.feature_fusion = FeatureFusion()

        logger.info(f"Dataset initialized: {len(self)} samples")

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict:
        """
        Get item by index.

        Returns:
            dict with keys:
                - waveform: torch.Tensor (samples,)
                - spectrogram: torch.Tensor (2, freq, time)
                - acoustic_features: torch.Tensor (n_features,)
                - label: torch.Tensor (1,)
                - speaker_id: str
        """
        # Check cache
        if self.cache_features and idx in self.feature_cache:
            return self.feature_cache[idx]

        # Load sample info
        row = self.manifest.iloc[idx]
        audio_path = row["file_path"]  # Changed from "filepath" to "file_path"
        label = int(row["label"])
        speaker_id = row["speaker_id"]

        try:
            # Load and preprocess audio
            audio_input, waveform = self.audio_loader.load(audio_path)
            preprocessed = self.preprocessor.process(
                waveform,
                sr=audio_input.sample_rate,  # Use original SR
                original_duration=row["duration"],
            )

            waveform = preprocessed.waveform
            sr = preprocessed.sample_rate

            # Apply augmentation if training
            if self.augment:
                waveform = self._apply_augmentation(waveform, sr)

            # Extract features
            mfcc = self.mfcc_extractor.extract(waveform, sr)
            prosody = self.prosodic_extractor.extract(waveform, sr)
            formants = self.formant_extractor.extract(waveform, sr)
            egemaps = self.egemaps_extractor.extract(waveform, sr)
            spectrogram = self.spectrogram_builder.build(waveform, sr)

            # Create feature bundle
            feature_bundle = FeatureBundle(
                waveform=waveform,
                sample_rate=sr,
                duration_sec=preprocessed.duration_sec,
                mfcc=mfcc,
                prosody=prosody,
                formants=formants,
                egemaps=egemaps,
                spectrogram=spectrogram,
            )

            # Fuse acoustic features
            feature_bundle = self.feature_fusion.fuse(feature_bundle)

            # Convert to tensors
            item = {
                "waveform": torch.from_numpy(waveform).float(),
                "spectrogram": torch.from_numpy(spectrogram.stacked).float(),
                "acoustic_features": torch.from_numpy(feature_bundle.fused_acoustic).float(),
                "label": torch.tensor([label], dtype=torch.long),
                "speaker_id": speaker_id,
            }

            # Cache if enabled
            if self.cache_features:
                self.feature_cache[idx] = item

            return item

        except Exception as e:
            logger.error(f"Failed to load sample {idx} ({audio_path}): {e}")
            # Return a dummy sample
            return self._get_dummy_item(label, speaker_id)

    def _apply_augmentation(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Apply data augmentation."""
        from training.augmentation import AudioAugmenter

        augmenter = AudioAugmenter()
        return augmenter.augment(waveform, sr)

    def _get_dummy_item(self, label: int, speaker_id: str) -> dict:
        """Return a dummy item when loading fails."""
        return {
            "waveform": torch.zeros(16000 * 10),  # 10 seconds of silence
            "spectrogram": torch.zeros(2, 128, 313),
            "acoustic_features": torch.zeros(145),
            "label": torch.tensor([label], dtype=torch.long),
            "speaker_id": speaker_id,
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate function for DataLoader.

    Handles variable-length sequences by padding.
    """
    # Find max lengths
    max_waveform_len = max(item["waveform"].shape[0] for item in batch)
    max_time_frames = max(item["spectrogram"].shape[2] for item in batch)

    # Pad sequences
    waveforms = []
    spectrograms = []
    acoustic_features = []
    labels = []
    speaker_ids = []

    for item in batch:
        # Pad waveform
        waveform = item["waveform"]
        if waveform.shape[0] < max_waveform_len:
            waveform = torch.nn.functional.pad(
                waveform, (0, max_waveform_len - waveform.shape[0])
            )
        waveforms.append(waveform)

        # Pad spectrogram
        spec = item["spectrogram"]
        if spec.shape[2] < max_time_frames:
            spec = torch.nn.functional.pad(
                spec, (0, max_time_frames - spec.shape[2])
            )
        spectrograms.append(spec)

        acoustic_features.append(item["acoustic_features"])
        labels.append(item["label"])
        speaker_ids.append(item["speaker_id"])

    return {
        "waveform": torch.stack(waveforms),
        "spectrogram": torch.stack(spectrograms),
        "acoustic_features": torch.stack(acoustic_features),
        "label": torch.stack(labels),
        "speaker_id": speaker_ids,
    }
