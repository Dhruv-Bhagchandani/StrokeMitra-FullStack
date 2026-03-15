#!/usr/bin/env python3
"""Build train/val/test manifest CSV files from raw dataset."""

import logging
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import soundfile as sf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_manifests(
    data_root: Path = Path("data/raw/kaggle_dysarthria"),
    output_dir: Path = Path("data/manifests"),
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42,
):
    """
    Build stratified train/val/test manifests.

    Args:
        data_root: Root directory with audio files
        output_dir: Output directory for manifest CSVs
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        random_seed: Random seed for reproducibility
    """
    logger.info("Building dataset manifests...")

    # Find all audio files
    audio_files = list(data_root.rglob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        raise ValueError(f"No audio files found in {data_root}")

    # Build dataset info
    data = []
    for audio_path in audio_files:
        try:
            # Get audio info
            info = sf.info(audio_path)

            # Determine label from path or filename
            # Check for "Non" prefix first (for datasets like "Non_Dysarthria")
            path_str = str(audio_path).lower()
            if "non_dysarthria" in path_str or "non-dysarthria" in path_str:
                label = 0  # Healthy (Non-dysarthric)
            elif "dysarthric" in path_str or "dysarthria" in path_str:
                label = 1  # Dysarthric
            elif "healthy" in path_str or "normal" in path_str or "control" in path_str:
                label = 0  # Healthy
            else:
                # Try to infer from filename
                filename = audio_path.stem.lower()
                if "dys" in filename or "impaired" in filename:
                    label = 1
                else:
                    label = 0

            # Extract speaker ID (if available in filename)
            speaker_id = audio_path.stem.split("_")[0] if "_" in audio_path.stem else "unknown"

            data.append({
                "file_path": str(audio_path.absolute()),
                "label": label,
                "speaker_id": speaker_id,
                "duration": info.duration,
                "sample_rate": info.samplerate,
                "channels": info.channels,
            })
        except Exception as e:
            logger.warning(f"Skipping {audio_path}: {e}")

    df = pd.DataFrame(data)

    logger.info(f"Dataset statistics:")
    logger.info(f"  Total samples: {len(df)}")
    logger.info(f"  Dysarthric: {(df['label'] == 1).sum()}")
    logger.info(f"  Healthy: {(df['label'] == 0).sum()}")
    logger.info(f"  Duration range: {df['duration'].min():.1f}s - {df['duration'].max():.1f}s")

    # Stratified split: train/temp
    train_df, temp_df = train_test_split(
        df,
        train_size=train_ratio,
        stratify=df["label"],
        random_state=random_seed,
    )

    # Split temp into val/test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=val_size,
        stratify=temp_df["label"],
        random_state=random_seed,
    )

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_df)} ({len(train_df)/len(df):.1%})")
    logger.info(f"  Val:   {len(val_df)} ({len(val_df)/len(df):.1%})")
    logger.info(f"  Test:  {len(test_df)} ({len(test_df)/len(df):.1%})")

    # Save manifests
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.csv"
    val_path = output_dir / "val.csv"
    test_path = output_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info(f"\n✓ Manifests saved:")
    logger.info(f"  Train: {train_path}")
    logger.info(f"  Val:   {val_path}")
    logger.info(f"  Test:  {test_path}")

    return train_df, val_df, test_df


if __name__ == "__main__":
    try:
        build_manifests()
        logger.info("\n✓ Dataset preparation complete!")
        logger.info("Next step: Run training scripts")
    except Exception as e:
        logger.error(f"Failed to build manifests: {e}")
        raise
