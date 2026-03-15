"""
Optimize ensemble weights between HuBERT-SALR and CNN-BiLSTM models.

This script performs grid search to find the optimal alpha (mixing weight):
    ensemble_logits = alpha * hubert_logits + (1 - alpha) * cnn_logits

Usage:
    python training/train_ensemble_weights.py
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
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from training.dataset import DysarthriaDataset


# ══════════════════════════════════════════════════════════════════════════════
# Model Loading Utilities
# ══════════════════════════════════════════════════════════════════════════════

def load_hubert_salr(checkpoint_path: Path, device):
    """Load trained HuBERT-SALR model."""
    from training.train_hubert_salr import HuBERTSALRModel

    model = HuBERTSALRModel()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_cnn_bilstm(checkpoint_path: Path, device):
    """Load trained CNN-BiLSTM model."""
    from training.train_cnn_bilstm import CNNBiLSTMTransformer

    model = CNNBiLSTMTransformer()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# Ensemble Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_ensemble(
    hubert_model,
    cnn_model,
    dataloader,
    alpha: float,
    device,
):
    """
    Evaluate ensemble with given alpha weight.

    Args:
        hubert_model: HuBERT-SALR model
        cnn_model: CNN-BiLSTM model
        dataloader: Validation data
        alpha: Mixing weight (0 to 1)
        device: torch device

    Returns:
        Dict of metrics
    """
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Alpha={alpha:.2f}", leave=False):
            waveform = batch["waveform"].to(device)
            spectrogram = batch["spectrogram"].to(device)
            labels = batch["label"].to(device)

            # Get predictions from both models
            hubert_logits = hubert_model(waveform)
            cnn_logits = cnn_model(spectrogram)

            # Ensemble
            ensemble_logits = alpha * hubert_logits + (1 - alpha) * cnn_logits

            # Convert to predictions
            probs = torch.softmax(ensemble_logits, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(ensemble_logits, dim=1).cpu().numpy()

            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="binary")
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)

    # Compute sensitivity and specificity
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    return {
        "alpha": alpha,
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "confusion_matrix": cm,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Grid Search
# ══════════════════════════════════════════════════════════════════════════════

def grid_search_alpha(
    hubert_model,
    cnn_model,
    dataloader,
    device,
    alpha_range=(0.0, 1.0),
    num_points=21,
):
    """
    Perform grid search over alpha values.

    Args:
        hubert_model: HuBERT-SALR model
        cnn_model: CNN-BiLSTM model
        dataloader: Validation data
        device: torch device
        alpha_range: (min, max) alpha values
        num_points: Number of alpha values to test

    Returns:
        DataFrame with results for each alpha
    """
    alphas = np.linspace(alpha_range[0], alpha_range[1], num_points)
    results = []

    for alpha in alphas:
        metrics = evaluate_ensemble(hubert_model, cnn_model, dataloader, alpha, device)
        results.append(metrics)

    return pd.DataFrame(results)


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_alpha_search(results_df, output_path: Path):
    """Plot metrics vs alpha."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    metrics = ["accuracy", "f1", "auc", "sensitivity"]
    titles = ["Accuracy", "F1 Score", "AUC-ROC", "Sensitivity"]

    for ax, metric, title in zip(axes.flat, metrics, titles):
        ax.plot(results_df["alpha"], results_df[metric], marker="o", linewidth=2)
        ax.set_xlabel("Alpha (HuBERT weight)", fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(f"{title} vs Alpha", fontsize=14)
        ax.grid(True, alpha=0.3)

        # Mark best alpha
        best_idx = results_df[metric].idxmax()
        best_alpha = results_df.loc[best_idx, "alpha"]
        best_value = results_df.loc[best_idx, metric]
        ax.axvline(best_alpha, color="red", linestyle="--", alpha=0.5)
        ax.scatter([best_alpha], [best_value], color="red", s=100, zorder=5)
        ax.text(
            best_alpha,
            best_value,
            f"α={best_alpha:.2f}\n{best_value:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm, alpha, output_path: Path):
    """Plot confusion matrix for best alpha."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Dysarthric"],
        yticklabels=["Healthy", "Dysarthric"],
    )
    plt.title(f"Confusion Matrix (α={alpha:.2f})", fontsize=14)
    plt.ylabel("True Label", fontsize=12)
    plt.xlabel("Predicted Label", fontsize=12)
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
    hubert_checkpoint = Path("models/hubert_salr_best.pt")
    cnn_checkpoint = Path("models/cnn_bilstm_best.pt")

    if not hubert_checkpoint.exists():
        logger.error(f"HuBERT checkpoint not found: {hubert_checkpoint}")
        logger.error("Please train HuBERT-SALR first: python training/train_hubert_salr.py")
        return

    if not cnn_checkpoint.exists():
        logger.error(f"CNN-BiLSTM checkpoint not found: {cnn_checkpoint}")
        logger.error("Please train CNN-BiLSTM first: python training/train_cnn_bilstm.py")
        return

    logger.info("Loading HuBERT-SALR model...")
    hubert_model = load_hubert_salr(hubert_checkpoint, device)

    logger.info("Loading CNN-BiLSTM model...")
    cnn_model = load_cnn_bilstm(cnn_checkpoint, device)

    # ──────────────────────────────────────────────────────────────────────────
    # Load Validation Data
    # ──────────────────────────────────────────────────────────────────────────
    val_manifest = Path("data/manifests/val.csv")
    val_dataset = DysarthriaDataset(val_manifest, augmentor=None, mode="val")
    val_loader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Validation samples: {len(val_dataset)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Grid Search
    # ──────────────────────────────────────────────────────────────────────────
    mlflow.set_experiment("ensemble_weight_optimization")

    with mlflow.start_run():
        logger.info("\nStarting grid search over alpha values...")

        results_df = grid_search_alpha(
            hubert_model,
            cnn_model,
            val_loader,
            device,
            alpha_range=(0.0, 1.0),
            num_points=21,
        )

        # Find best alpha for each metric
        best_alpha_auc = results_df.loc[results_df["auc"].idxmax(), "alpha"]
        best_alpha_f1 = results_df.loc[results_df["f1"].idxmax(), "alpha"]
        best_alpha_acc = results_df.loc[results_df["accuracy"].idxmax(), "alpha"]

        logger.info("\n" + "=" * 80)
        logger.info("GRID SEARCH RESULTS")
        logger.info("=" * 80)
        logger.info(f"Best alpha (AUC):      {best_alpha_auc:.2f}")
        logger.info(f"Best alpha (F1):       {best_alpha_f1:.2f}")
        logger.info(f"Best alpha (Accuracy): {best_alpha_acc:.2f}")
        logger.info("=" * 80)

        # Use AUC as primary metric
        best_alpha = best_alpha_auc
        best_row = results_df.loc[results_df["alpha"] == best_alpha].iloc[0]

        logger.info(f"\nOptimal alpha: {best_alpha:.2f}")
        logger.info(f"  Accuracy:    {best_row['accuracy']:.4f}")
        logger.info(f"  F1 Score:    {best_row['f1']:.4f}")
        logger.info(f"  AUC:         {best_row['auc']:.4f}")
        logger.info(f"  Sensitivity: {best_row['sensitivity']:.4f}")
        logger.info(f"  Specificity: {best_row['specificity']:.4f}")

        # Log to MLflow
        mlflow.log_params({
            "num_alpha_points": 21,
            "alpha_range_min": 0.0,
            "alpha_range_max": 1.0,
        })

        mlflow.log_metrics({
            "best_alpha": best_alpha,
            "best_accuracy": best_row["accuracy"],
            "best_f1": best_row["f1"],
            "best_auc": best_row["auc"],
            "best_sensitivity": best_row["sensitivity"],
            "best_specificity": best_row["specificity"],
        })

        # Save results
        output_dir = Path("reports/ensemble_optimization")
        output_dir.mkdir(parents=True, exist_ok=True)

        results_csv = output_dir / "alpha_search_results.csv"
        results_df.to_csv(results_csv, index=False)
        mlflow.log_artifact(str(results_csv))
        logger.info(f"\n✓ Results saved to {results_csv}")

        # Plot metrics vs alpha
        plot_path = output_dir / "alpha_search_plot.png"
        plot_alpha_search(results_df, plot_path)
        mlflow.log_artifact(str(plot_path))
        logger.info(f"✓ Plots saved to {plot_path}")

        # Plot confusion matrix for best alpha
        cm_path = output_dir / "confusion_matrix_best_alpha.png"
        plot_confusion_matrix(best_row["confusion_matrix"], best_alpha, cm_path)
        mlflow.log_artifact(str(cm_path))
        logger.info(f"✓ Confusion matrix saved to {cm_path}")

        # Save optimal config
        optimal_config = {
            "ensemble": {
                "alpha": float(best_alpha),
                "hubert_weight": float(best_alpha),
                "cnn_bilstm_weight": float(1 - best_alpha),
                "validation_metrics": {
                    "accuracy": float(best_row["accuracy"]),
                    "f1": float(best_row["f1"]),
                    "auc": float(best_row["auc"]),
                    "sensitivity": float(best_row["sensitivity"]),
                    "specificity": float(best_row["specificity"]),
                },
            }
        }

        config_path = output_dir / "optimal_ensemble_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(optimal_config, f, default_flow_style=False)
        mlflow.log_artifact(str(config_path))
        logger.info(f"✓ Optimal config saved to {config_path}")

        logger.info("\n✓ Ensemble weight optimization complete!")
        logger.info(f"  Update configs/model_config.yaml with alpha={best_alpha:.2f}")


if __name__ == "__main__":
    main()
