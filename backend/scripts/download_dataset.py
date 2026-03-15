#!/usr/bin/env python3
"""Download Kaggle dysarthria dataset."""

import os
import sys
import logging
import hashlib
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_kaggle_dataset():
    """Download dysarthria dataset from Kaggle."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        logger.error("Kaggle API not installed. Install with: pip install kaggle")
        sys.exit(1)

    # Dataset info
    dataset_slug = "poojag718/dysarthria-and-nondysarthria-speech-dataset"
    download_path = Path("data/raw/kaggle_dysarthria")

    logger.info(f"Downloading dataset: {dataset_slug}")
    logger.info(f"Destination: {download_path}")

    # Create directory
    download_path.mkdir(parents=True, exist_ok=True)

    # Initialize Kaggle API
    api = KaggleApi()
    api.authenticate()

    # Download dataset
    logger.info("Starting download...")
    api.dataset_download_files(
        dataset_slug,
        path=str(download_path),
        unzip=True
    )

    logger.info("✓ Dataset downloaded successfully")

    # Verify files
    audio_files = list(download_path.rglob("*.wav"))
    logger.info(f"Found {len(audio_files)} audio files")

    if len(audio_files) == 0:
        logger.warning("No audio files found! Check the download.")
    else:
        logger.info("✓ Download verified")

    return download_path


def verify_integrity(file_path: Path, expected_md5: str = None):
    """Verify file integrity using MD5 hash."""
    if expected_md5 is None:
        return True

    md5_hash = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            md5_hash.update(chunk)

    actual_md5 = md5_hash.hexdigest()
    return actual_md5 == expected_md5


if __name__ == "__main__":
    # Check for Kaggle credentials
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        logger.error(
            "Kaggle credentials not found!\n\n"
            "Setup instructions:\n"
            "1. Go to https://www.kaggle.com/account\n"
            "2. Create API token (downloads kaggle.json)\n"
            "3. Move to ~/.kaggle/kaggle.json\n"
            "4. chmod 600 ~/.kaggle/kaggle.json\n"
        )
        sys.exit(1)

    try:
        download_path = download_kaggle_dataset()
        logger.info(f"\n✓ Dataset ready at: {download_path}")
        logger.info("\nNext step: Run scripts/build_manifests.py to create train/val/test splits")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        sys.exit(1)
