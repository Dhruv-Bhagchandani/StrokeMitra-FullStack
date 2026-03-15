"""
Data drift detection using Evidently AI.

Monitors:
- Feature distribution drift (acoustic features, MFCCs, prosody, formants)
- Model prediction drift
- Target drift (if labels available)

Usage:
    detector = DriftDetector(reference_data_path="data/reference_set.csv")
    report = detector.detect_drift(current_data)
    if report.has_drift:
        print("Drift detected! Consider retraining.")
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
import json
import logging
from datetime import datetime

try:
    from evidently import ColumnMapping
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset
    from evidently.metrics import (
        DataDriftTable,
        DatasetDriftMetric,
        ColumnDriftMetric,
    )
    EVIDENTLY_AVAILABLE = True
except ImportError:
    EVIDENTLY_AVAILABLE = False
    logging.warning("Evidently AI not installed. Drift detection will be disabled.")


# ══════════════════════════════════════════════════════════════════════════════
# Drift Detection
# ══════════════════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Detect data drift in production using Evidently AI.

    This class compares current production data against a reference dataset
    to identify distribution shifts that may degrade model performance.
    """

    def __init__(
        self,
        reference_data_path: Optional[Path] = None,
        reference_data: Optional[pd.DataFrame] = None,
        drift_threshold: float = 0.5,
    ):
        """
        Initialize drift detector.

        Args:
            reference_data_path: Path to reference dataset CSV
            reference_data: Reference DataFrame (alternative to path)
            drift_threshold: Drift score threshold (0-1, higher = more drift tolerance)
        """
        if not EVIDENTLY_AVAILABLE:
            raise ImportError("Evidently AI required for drift detection. Install: pip install evidently")

        self.drift_threshold = drift_threshold
        self.logger = logging.getLogger(__name__)

        # Load reference data
        if reference_data_path:
            self.reference_data = pd.read_csv(reference_data_path)
            self.logger.info(f"Loaded reference data: {len(self.reference_data)} samples")
        elif reference_data is not None:
            self.reference_data = reference_data
            self.logger.info(f"Using provided reference data: {len(self.reference_data)} samples")
        else:
            raise ValueError("Must provide either reference_data_path or reference_data")

    def detect_drift(
        self,
        current_data: pd.DataFrame,
        save_report: bool = True,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Detect drift between reference and current data.

        Args:
            current_data: Current production data
            save_report: Whether to save HTML report
            output_path: Path to save report (auto-generated if None)

        Returns:
            Dict with:
                - has_drift: bool
                - drift_score: float
                - drifted_features: list
                - report_path: Path (if saved)
        """
        self.logger.info(f"Detecting drift on {len(current_data)} samples...")

        # Create column mapping
        column_mapping = self._create_column_mapping(current_data)

        # Build report
        report = Report(metrics=[
            DatasetDriftMetric(),
            DataDriftTable(),
        ])

        report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping,
        )

        # Extract results
        report_dict = report.as_dict()
        dataset_drift = report_dict["metrics"][0]["result"]

        has_drift = dataset_drift["dataset_drift"]
        drift_score = dataset_drift.get("drift_share", 0.0)
        drifted_features = [
            col for col, metrics in dataset_drift.get("drift_by_columns", {}).items()
            if metrics.get("drift_detected", False)
        ]

        result = {
            "timestamp": datetime.utcnow().isoformat(),
            "has_drift": has_drift,
            "drift_score": drift_score,
            "drifted_features": drifted_features,
            "num_drifted_features": len(drifted_features),
            "total_features": len(current_data.columns),
            "reference_samples": len(self.reference_data),
            "current_samples": len(current_data),
        }

        # Log results
        if has_drift:
            self.logger.warning(
                f"⚠️  DRIFT DETECTED! Score: {drift_score:.4f}, "
                f"{len(drifted_features)} features drifted"
            )
            for feature in drifted_features:
                self.logger.warning(f"   - {feature}")
        else:
            self.logger.info(f"✓ No drift detected (score: {drift_score:.4f})")

        # Save report
        if save_report:
            if output_path is None:
                output_dir = Path("reports/drift")
                output_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                output_path = output_dir / f"drift_report_{timestamp}.html"

            report.save_html(str(output_path))
            result["report_path"] = str(output_path)
            self.logger.info(f"✓ Drift report saved to {output_path}")

        return result

    def _create_column_mapping(self, data: pd.DataFrame) -> ColumnMapping:
        """Create Evidently column mapping."""
        # Identify numeric features (exclude label/prediction columns)
        numeric_features = [
            col for col in data.select_dtypes(include=[np.number]).columns
            if col not in ["label", "prediction", "probability"]
        ]

        column_mapping = ColumnMapping()
        column_mapping.numerical_features = numeric_features

        if "label" in data.columns:
            column_mapping.target = "label"

        if "prediction" in data.columns:
            column_mapping.prediction = "prediction"

        return column_mapping

    def monitor_continuous(
        self,
        current_data: pd.DataFrame,
        window_size: int = 1000,
    ) -> Dict[str, Any]:
        """
        Monitor drift on a rolling window basis.

        Args:
            current_data: Recent production data
            window_size: Number of recent samples to compare

        Returns:
            Drift detection results
        """
        if len(current_data) > window_size:
            # Take most recent window
            windowed_data = current_data.tail(window_size)
        else:
            windowed_data = current_data

        return self.detect_drift(windowed_data, save_report=True)

    def compute_feature_drift_scores(
        self,
        current_data: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Compute drift score for each feature individually.

        Returns:
            Dict mapping feature name to drift score
        """
        feature_scores = {}

        for col in current_data.select_dtypes(include=[np.number]).columns:
            if col in ["label", "prediction", "probability"]:
                continue

            # Create single-feature report
            report = Report(metrics=[
                ColumnDriftMetric(column_name=col),
            ])

            # Create single-column dataframes
            ref_df = self.reference_data[[col]].copy()
            curr_df = current_data[[col]].copy()

            report.run(reference_data=ref_df, current_data=curr_df)

            # Extract drift score
            report_dict = report.as_dict()
            drift_score = report_dict["metrics"][0]["result"].get("drift_score", 0.0)
            feature_scores[col] = drift_score

        return feature_scores


# ══════════════════════════════════════════════════════════════════════════════
# Utility Functions
# ══════════════════════════════════════════════════════════════════════════════

def create_reference_dataset(
    manifest_path: Path,
    feature_extractor_func,
    num_samples: int = 500,
    output_path: Path = Path("data/reference_set.csv"),
):
    """
    Create reference dataset from training data.

    Args:
        manifest_path: Path to training manifest
        feature_extractor_func: Function to extract features from audio
        num_samples: Number of samples to include
        output_path: Where to save reference dataset
    """
    import pandas as pd

    manifest = pd.read_csv(manifest_path)

    # Sample stratified by label
    if "label" in manifest.columns:
        reference_samples = manifest.groupby("label", group_keys=False).apply(
            lambda x: x.sample(min(len(x), num_samples // 2))
        )
    else:
        reference_samples = manifest.sample(n=min(len(manifest), num_samples))

    # Extract features
    feature_rows = []
    for idx, row in reference_samples.iterrows():
        features = feature_extractor_func(row["file_path"])
        features["label"] = row.get("label", None)
        feature_rows.append(features)

    reference_df = pd.DataFrame(feature_rows)
    reference_df.to_csv(output_path, index=False)

    logging.info(f"✓ Reference dataset created: {len(reference_df)} samples → {output_path}")
    return reference_df


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    # Mock data for demonstration
    np.random.seed(42)

    # Reference data (training distribution)
    reference = pd.DataFrame({
        "mfcc_mean": np.random.normal(0, 1, 1000),
        "f0_mean": np.random.normal(150, 30, 1000),
        "energy_mean": np.random.normal(0.5, 0.1, 1000),
        "label": np.random.choice([0, 1], 1000),
    })

    # Current data (slight drift)
    current = pd.DataFrame({
        "mfcc_mean": np.random.normal(0.3, 1.2, 500),  # Shifted distribution
        "f0_mean": np.random.normal(155, 35, 500),     # Shifted distribution
        "energy_mean": np.random.normal(0.5, 0.1, 500),
        "label": np.random.choice([0, 1], 500),
    })

    # Detect drift
    detector = DriftDetector(reference_data=reference, drift_threshold=0.5)
    result = detector.detect_drift(current, save_report=True)

    print(f"\nDrift Detection Results:")
    print(f"  Has drift: {result['has_drift']}")
    print(f"  Drift score: {result['drift_score']:.4f}")
    print(f"  Drifted features: {result['drifted_features']}")
