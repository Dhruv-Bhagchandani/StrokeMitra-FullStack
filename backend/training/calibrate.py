"""
Calibrate model probabilities using Platt scaling.

This script:
1. Loads the ensemble model
2. Collects predictions on a held-out calibration set
3. Fits Platt scaling parameters (a, b) via logistic regression
4. Evaluates calibration quality (ECE, reliability diagrams)

Usage:
    python training/calibrate.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import numpy as np
from tqdm import tqdm
import yaml
import logging
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt

from training.dataset import DysarthriaDataset
from training.train_hubert_salr import HuBERTSALRModel
from training.train_cnn_bilstm import CNNBiLSTMTransformer


# ══════════════════════════════════════════════════════════════════════════════
# Calibration Metrics
# ══════════════════════════════════════════════════════════════════════════════

def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Compute Expected Calibration Error (ECE).

    ECE measures the difference between predicted confidence and actual accuracy.
    Lower ECE indicates better calibration.

    Args:
        y_true: True labels (0 or 1)
        y_prob: Predicted probabilities (0 to 1)
        n_bins: Number of bins for binning predictions

    Returns:
        ECE value
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_prob[mask].mean()
            bin_weight = mask.sum() / len(y_true)
            ece += bin_weight * np.abs(bin_acc - bin_conf)

    return ece


def reliability_curve(y_true, y_prob, n_bins=10):
    """
    Compute reliability curve data for plotting.

    Returns:
        bin_centers, bin_accuracies, bin_confidences, bin_counts
    """
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[:-1]) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_centers = []
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
            bin_accuracies.append(y_true[mask].mean())
            bin_confidences.append(y_prob[mask].mean())
            bin_counts.append(mask.sum())

    return (
        np.array(bin_centers),
        np.array(bin_accuracies),
        np.array(bin_confidences),
        np.array(bin_counts),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Model Inference
# ══════════════════════════════════════════════════════════════════════════════

def collect_predictions(hubert_model, cnn_model, dataloader, alpha, device):
    """
    Collect raw logits and probabilities from ensemble.

    Args:
        hubert_model: HuBERT-SALR model
        cnn_model: CNN-BiLSTM model
        dataloader: Data loader
        alpha: Ensemble mixing weight
        device: torch device

    Returns:
        logits, probabilities, true labels (all numpy arrays)
    """
    all_logits = []
    all_probs = []
    all_labels = []

    hubert_model.eval()
    cnn_model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Collecting predictions"):
            waveform = batch["waveform"].to(device)
            spectrogram = batch["spectrogram"].to(device)
            labels = batch["label"]

            # Ensemble logits
            hubert_logits = hubert_model(waveform)
            cnn_logits = cnn_model(spectrogram)
            ensemble_logits = alpha * hubert_logits + (1 - alpha) * cnn_logits

            # Probabilities (uncalibrated)
            probs = torch.softmax(ensemble_logits, dim=1)[:, 1]

            all_logits.extend(ensemble_logits[:, 1].cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

    return (
        np.array(all_logits),
        np.array(all_probs),
        np.array(all_labels),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Platt Scaling
# ══════════════════════════════════════════════════════════════════════════════

def fit_platt_scaling(logits, labels):
    """
    Fit Platt scaling parameters.

    Platt scaling fits:
        calibrated_prob = sigmoid(a * logit + b)

    Args:
        logits: Raw model logits (n_samples,)
        labels: True binary labels (n_samples,)

    Returns:
        a, b parameters
    """
    # Reshape for sklearn
    X = logits.reshape(-1, 1)
    y = labels

    # Fit logistic regression (no regularization)
    lr = LogisticRegression(penalty=None, solver="lbfgs", max_iter=1000)
    lr.fit(X, y)

    a = lr.coef_[0][0]
    b = lr.intercept_[0]

    return a, b


def apply_platt_scaling(logits, a, b):
    """Apply Platt scaling to logits."""
    z = a * logits + b
    calibrated_probs = 1 / (1 + np.exp(-z))
    return calibrated_probs


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_reliability_diagram(
    y_true,
    y_prob_uncal,
    y_prob_cal,
    output_path: Path,
):
    """Plot reliability diagram comparing uncalibrated vs calibrated."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, probs, title in zip(
        axes,
        [y_prob_uncal, y_prob_cal],
        ["Uncalibrated", "Calibrated"],
    ):
        centers, accs, confs, counts = reliability_curve(y_true, probs, n_bins=10)

        # Plot reliability curve
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", linewidth=2)
        ax.scatter(confs, accs, s=counts * 3, alpha=0.6, label="Model", zorder=5)
        ax.plot(confs, accs, "o-", linewidth=2, markersize=8)

        # Compute ECE
        ece = expected_calibration_error(y_true, probs)
        brier = brier_score_loss(y_true, probs)

        ax.set_xlabel("Mean Predicted Probability", fontsize=12)
        ax.set_ylabel("Fraction of Positives", fontsize=12)
        ax.set_title(f"{title}\nECE: {ece:.4f}, Brier: {brier:.4f}", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_histogram_comparison(
    y_true,
    y_prob_uncal,
    y_prob_cal,
    output_path: Path,
):
    """Plot histogram of predicted probabilities."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Split by true label
    mask_positive = y_true == 1
    mask_negative = y_true == 0

    for i, (probs, title) in enumerate(
        [(y_prob_uncal, "Uncalibrated"), (y_prob_cal, "Calibrated")]
    ):
        # Positive class
        axes[i, 0].hist(probs[mask_positive], bins=20, alpha=0.7, color="red", edgecolor="black")
        axes[i, 0].set_xlabel("Predicted Probability", fontsize=12)
        axes[i, 0].set_ylabel("Count", fontsize=12)
        axes[i, 0].set_title(f"{title} - True Dysarthric", fontsize=14)
        axes[i, 0].grid(True, alpha=0.3)

        # Negative class
        axes[i, 1].hist(probs[mask_negative], bins=20, alpha=0.7, color="blue", edgecolor="black")
        axes[i, 1].set_xlabel("Predicted Probability", fontsize=12)
        axes[i, 1].set_ylabel("Count", fontsize=12)
        axes[i, 1].set_title(f"{title} - True Healthy", fontsize=14)
        axes[i, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ──────────────────────────────────────────────────────────────────────────
    # Load Models
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Loading models...")

    hubert_checkpoint = Path("models/hubert_salr_best.pt")
    cnn_checkpoint = Path("models/cnn_bilstm_best.pt")

    hubert_model = HuBERTSALRModel()
    hubert_model.load_state_dict(torch.load(hubert_checkpoint, map_location=device)["model_state_dict"])
    hubert_model.to(device)

    cnn_model = CNNBiLSTMTransformer()
    cnn_model.load_state_dict(torch.load(cnn_checkpoint, map_location=device)["model_state_dict"])
    cnn_model.to(device)

    # Load optimal alpha
    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)
    alpha = config.get("ensemble", {}).get("alpha", 0.6)
    logger.info(f"Using ensemble alpha: {alpha}")

    # ──────────────────────────────────────────────────────────────────────────
    # Load Calibration Data (use validation set)
    # ──────────────────────────────────────────────────────────────────────────
    val_manifest = Path("data/manifests/val.csv")
    val_dataset = DysarthriaDataset(val_manifest, augmentor=None, mode="val")
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    logger.info(f"Calibration samples: {len(val_dataset)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Collect Predictions
    # ──────────────────────────────────────────────────────────────────────────
    logger.info("Collecting predictions...")

    logits, probs_uncal, labels = collect_predictions(
        hubert_model, cnn_model, val_loader, alpha, device
    )

    # ──────────────────────────────────────────────────────────────────────────
    # Fit Platt Scaling
    # ──────────────────────────────────────────────────────────────────────────
    mlflow.set_experiment("model_calibration")

    with mlflow.start_run():
        logger.info("\nFitting Platt scaling...")

        a, b = fit_platt_scaling(logits, labels)
        logger.info(f"Platt parameters: a={a:.6f}, b={b:.6f}")

        # Apply calibration
        probs_cal = apply_platt_scaling(logits, a, b)

        # ──────────────────────────────────────────────────────────────────────
        # Evaluate Calibration
        # ──────────────────────────────────────────────────────────────────────
        ece_uncal = expected_calibration_error(labels, probs_uncal)
        ece_cal = expected_calibration_error(labels, probs_cal)

        brier_uncal = brier_score_loss(labels, probs_uncal)
        brier_cal = brier_score_loss(labels, probs_cal)

        logloss_uncal = log_loss(labels, probs_uncal)
        logloss_cal = log_loss(labels, probs_cal)

        logger.info("\n" + "=" * 80)
        logger.info("CALIBRATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Expected Calibration Error (ECE):")
        logger.info(f"  Uncalibrated: {ece_uncal:.4f}")
        logger.info(f"  Calibrated:   {ece_cal:.4f} ({'↓' if ece_cal < ece_uncal else '↑'} {abs(ece_cal - ece_uncal):.4f})")
        logger.info(f"\nBrier Score:")
        logger.info(f"  Uncalibrated: {brier_uncal:.4f}")
        logger.info(f"  Calibrated:   {brier_cal:.4f} ({'↓' if brier_cal < brier_uncal else '↑'} {abs(brier_cal - brier_uncal):.4f})")
        logger.info(f"\nLog Loss:")
        logger.info(f"  Uncalibrated: {logloss_uncal:.4f}")
        logger.info(f"  Calibrated:   {logloss_cal:.4f} ({'↓' if logloss_cal < logloss_uncal else '↑'} {abs(logloss_cal - logloss_uncal):.4f})")
        logger.info("=" * 80)

        # Log to MLflow
        mlflow.log_params({
            "platt_a": a,
            "platt_b": b,
            "calibration_samples": len(labels),
        })

        mlflow.log_metrics({
            "ece_uncalibrated": ece_uncal,
            "ece_calibrated": ece_cal,
            "ece_improvement": ece_uncal - ece_cal,
            "brier_uncalibrated": brier_uncal,
            "brier_calibrated": brier_cal,
            "logloss_uncalibrated": logloss_uncal,
            "logloss_calibrated": logloss_cal,
        })

        # ──────────────────────────────────────────────────────────────────────
        # Save Results
        # ──────────────────────────────────────────────────────────────────────
        output_dir = Path("reports/calibration")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Platt parameters
        calibration_config = {
            "platt_scaling": {
                "a": float(a),
                "b": float(b),
                "ece_uncalibrated": float(ece_uncal),
                "ece_calibrated": float(ece_cal),
                "brier_uncalibrated": float(brier_uncal),
                "brier_calibrated": float(brier_cal),
            }
        }

        config_path = output_dir / "calibration_params.yaml"
        with open(config_path, "w") as f:
            yaml.dump(calibration_config, f, default_flow_style=False)
        mlflow.log_artifact(str(config_path))
        logger.info(f"\n✓ Calibration parameters saved to {config_path}")

        # Plot reliability diagram
        reliability_path = output_dir / "reliability_diagram.png"
        plot_reliability_diagram(labels, probs_uncal, probs_cal, reliability_path)
        mlflow.log_artifact(str(reliability_path))
        logger.info(f"✓ Reliability diagram saved to {reliability_path}")

        # Plot histogram comparison
        hist_path = output_dir / "probability_histograms.png"
        plot_histogram_comparison(labels, probs_uncal, probs_cal, hist_path)
        mlflow.log_artifact(str(hist_path))
        logger.info(f"✓ Probability histograms saved to {hist_path}")

        logger.info("\n✓ Calibration complete!")
        logger.info(f"  Update configs/model_config.yaml with Platt parameters:")
        logger.info(f"    a: {a:.6f}")
        logger.info(f"    b: {b:.6f}")


if __name__ == "__main__":
    main()
