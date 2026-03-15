"""
Comprehensive model evaluation on test set.

This script:
1. Loads trained ensemble model with calibration
2. Evaluates on held-out test set
3. Computes classification metrics (accuracy, F1, AUC, sensitivity, specificity)
4. Generates confusion matrix, ROC curve, PR curve
5. Performs error analysis

Usage:
    python training/evaluate.py
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
import logging
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
import matplotlib.pyplot as plt
import seaborn as sns

from training.dataset import DysarthriaDataset
from training.train_hubert_salr import HuBERTSALRModel
from training.train_cnn_bilstm import CNNBiLSTMTransformer


# ══════════════════════════════════════════════════════════════════════════════
# Model Inference
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_model(hubert_model, cnn_model, dataloader, alpha, platt_a, platt_b, device):
    """
    Evaluate calibrated ensemble on test set.

    Returns:
        predictions, probabilities, labels, file_paths
    """
    all_preds = []
    all_probs = []
    all_labels = []
    all_files = []

    hubert_model.eval()
    cnn_model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            waveform = batch["waveform"].to(device)
            spectrogram = batch["spectrogram"].to(device)
            labels = batch["label"]
            file_paths = batch.get("file_path", [""] * len(labels))

            # Ensemble logits
            hubert_logits = hubert_model(waveform)
            cnn_logits = cnn_model(spectrogram)
            ensemble_logits = alpha * hubert_logits + (1 - alpha) * cnn_logits

            # Apply Platt scaling
            raw_logits = ensemble_logits[:, 1].cpu().numpy()
            z = platt_a * raw_logits + platt_b
            calibrated_probs = 1 / (1 + np.exp(-z))

            # Predictions
            preds = (calibrated_probs > 0.5).astype(int)

            all_preds.extend(preds)
            all_probs.extend(calibrated_probs)
            all_labels.extend(labels.numpy())
            all_files.extend(file_paths)

    return (
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_labels),
        all_files,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Metrics Computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob):
    """Compute comprehensive classification metrics."""
    # Basic metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity and specificity
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

    # Positive and negative predictive value
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0

    return {
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
        "average_precision": ap,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "confusion_matrix": cm,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════

def plot_confusion_matrix(cm, output_path: Path):
    """Plot confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Healthy", "Dysarthric"],
        yticklabels=["Healthy", "Dysarthric"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix - Test Set", fontsize=16, fontweight="bold")
    plt.ylabel("True Label", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_roc_curve(y_true, y_prob, auc_score, output_path: Path):
    """Plot ROC curve."""
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f"Model (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random Classifier")

    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curve - Test Set", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_precision_recall_curve(y_true, y_prob, ap_score, output_path: Path):
    """Plot Precision-Recall curve."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, linewidth=2, label=f"Model (AP = {ap_score:.4f})")

    plt.xlabel("Recall", fontsize=14)
    plt.ylabel("Precision", fontsize=14)
    plt.title("Precision-Recall Curve - Test Set", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_probability_distribution(y_true, y_prob, output_path: Path):
    """Plot distribution of predicted probabilities by class."""
    plt.figure(figsize=(10, 6))

    mask_positive = y_true == 1
    mask_negative = y_true == 0

    plt.hist(
        y_prob[mask_negative],
        bins=30,
        alpha=0.6,
        color="blue",
        label="Healthy",
        edgecolor="black",
    )
    plt.hist(
        y_prob[mask_positive],
        bins=30,
        alpha=0.6,
        color="red",
        label="Dysarthric",
        edgecolor="black",
    )

    plt.axvline(0.5, color="black", linestyle="--", linewidth=2, label="Decision Threshold")

    plt.xlabel("Predicted Probability", fontsize=14)
    plt.ylabel("Count", fontsize=14)
    plt.title("Predicted Probability Distribution - Test Set", fontsize=16, fontweight="bold")
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# Error Analysis
# ══════════════════════════════════════════════════════════════════════════════

def perform_error_analysis(y_true, y_pred, y_prob, file_paths, output_path: Path):
    """Identify and save misclassified samples."""
    errors = []

    for i, (true_label, pred_label, prob, file_path) in enumerate(
        zip(y_true, y_pred, y_prob, file_paths)
    ):
        if true_label != pred_label:
            error_type = "False Positive" if pred_label == 1 else "False Negative"
            confidence = prob if pred_label == 1 else (1 - prob)

            errors.append({
                "file_path": file_path,
                "true_label": "Dysarthric" if true_label == 1 else "Healthy",
                "predicted_label": "Dysarthric" if pred_label == 1 else "Healthy",
                "probability": prob,
                "confidence": confidence,
                "error_type": error_type,
            })

    errors_df = pd.DataFrame(errors)
    errors_df = errors_df.sort_values("confidence", ascending=False)
    errors_df.to_csv(output_path, index=False)

    return errors_df


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ──────────────────────────────────────────────────────────────────────────
    # Load Configuration
    # ──────────────────────────────────────────────────────────────────────────
    with open("configs/model_config.yaml") as f:
        config = yaml.safe_load(f)

    alpha = config.get("ensemble", {}).get("alpha", 0.6)

    # Load Platt scaling parameters
    calibration_file = Path("reports/calibration/calibration_params.yaml")
    if calibration_file.exists():
        with open(calibration_file) as f:
            cal_config = yaml.safe_load(f)
        platt_a = cal_config["platt_scaling"]["a"]
        platt_b = cal_config["platt_scaling"]["b"]
        logger.info(f"Loaded Platt parameters: a={platt_a:.6f}, b={platt_b:.6f}")
    else:
        platt_a, platt_b = 1.0, 0.0
        logger.warning("Calibration parameters not found, using identity mapping")

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

    # ──────────────────────────────────────────────────────────────────────────
    # Load Test Data
    # ──────────────────────────────────────────────────────────────────────────
    test_manifest = Path("data/manifests/test.csv")
    test_dataset = DysarthriaDataset(test_manifest, augmentor=None, mode="test")
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

    logger.info(f"Test samples: {len(test_dataset)}")

    # ──────────────────────────────────────────────────────────────────────────
    # Evaluate
    # ──────────────────────────────────────────────────────────────────────────
    mlflow.set_experiment("model_evaluation")

    with mlflow.start_run():
        logger.info("\nEvaluating on test set...")

        y_pred, y_prob, y_true, file_paths = evaluate_model(
            hubert_model, cnn_model, test_loader, alpha, platt_a, platt_b, device
        )

        # Compute metrics
        metrics = compute_metrics(y_true, y_pred, y_prob)

        # ──────────────────────────────────────────────────────────────────────
        # Print Results
        # ──────────────────────────────────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("TEST SET EVALUATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"Accuracy:           {metrics['accuracy']:.4f}")
        logger.info(f"F1 Score:           {metrics['f1']:.4f}")
        logger.info(f"AUC-ROC:            {metrics['auc']:.4f}")
        logger.info(f"Average Precision:  {metrics['average_precision']:.4f}")
        logger.info(f"Sensitivity:        {metrics['sensitivity']:.4f}")
        logger.info(f"Specificity:        {metrics['specificity']:.4f}")
        logger.info(f"PPV:                {metrics['ppv']:.4f}")
        logger.info(f"NPV:                {metrics['npv']:.4f}")
        logger.info("")
        logger.info("Confusion Matrix:")
        logger.info(f"  True Negatives:   {metrics['tn']}")
        logger.info(f"  False Positives:  {metrics['fp']}")
        logger.info(f"  False Negatives:  {metrics['fn']}")
        logger.info(f"  True Positives:   {metrics['tp']}")
        logger.info("=" * 80)

        # Log to MLflow
        mlflow.log_params({
            "ensemble_alpha": alpha,
            "platt_a": platt_a,
            "platt_b": platt_b,
            "test_samples": len(y_true),
        })

        mlflow.log_metrics({
            "test_accuracy": metrics["accuracy"],
            "test_f1": metrics["f1"],
            "test_auc": metrics["auc"],
            "test_ap": metrics["average_precision"],
            "test_sensitivity": metrics["sensitivity"],
            "test_specificity": metrics["specificity"],
            "test_ppv": metrics["ppv"],
            "test_npv": metrics["npv"],
        })

        # ──────────────────────────────────────────────────────────────────────
        # Save Results
        # ──────────────────────────────────────────────────────────────────────
        output_dir = Path("reports/evaluation")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = output_dir / "test_metrics.yaml"
        with open(metrics_file, "w") as f:
            # Convert numpy types to Python types
            metrics_to_save = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
            yaml.dump(metrics_to_save, f, default_flow_style=False)
        mlflow.log_artifact(str(metrics_file))
        logger.info(f"\n✓ Metrics saved to {metrics_file}")

        # Classification report
        report = classification_report(
            y_true,
            y_pred,
            target_names=["Healthy", "Dysarthric"],
            digits=4,
        )
        report_file = output_dir / "classification_report.txt"
        with open(report_file, "w") as f:
            f.write(report)
        mlflow.log_artifact(str(report_file))
        logger.info(f"✓ Classification report saved to {report_file}")

        # Confusion matrix
        cm_path = output_dir / "confusion_matrix.png"
        plot_confusion_matrix(metrics["confusion_matrix"], cm_path)
        mlflow.log_artifact(str(cm_path))
        logger.info(f"✓ Confusion matrix plot saved to {cm_path}")

        # ROC curve
        roc_path = output_dir / "roc_curve.png"
        plot_roc_curve(y_true, y_prob, metrics["auc"], roc_path)
        mlflow.log_artifact(str(roc_path))
        logger.info(f"✓ ROC curve saved to {roc_path}")

        # Precision-Recall curve
        pr_path = output_dir / "precision_recall_curve.png"
        plot_precision_recall_curve(y_true, y_prob, metrics["average_precision"], pr_path)
        mlflow.log_artifact(str(pr_path))
        logger.info(f"✓ Precision-Recall curve saved to {pr_path}")

        # Probability distribution
        prob_dist_path = output_dir / "probability_distribution.png"
        plot_probability_distribution(y_true, y_prob, prob_dist_path)
        mlflow.log_artifact(str(prob_dist_path))
        logger.info(f"✓ Probability distribution saved to {prob_dist_path}")

        # Error analysis
        errors_file = output_dir / "misclassified_samples.csv"
        errors_df = perform_error_analysis(y_true, y_pred, y_prob, file_paths, errors_file)
        mlflow.log_artifact(str(errors_file))
        logger.info(f"✓ Error analysis saved to {errors_file}")
        logger.info(f"  Total errors: {len(errors_df)}")
        logger.info(f"  False Positives: {len(errors_df[errors_df['error_type'] == 'False Positive'])}")
        logger.info(f"  False Negatives: {len(errors_df[errors_df['error_type'] == 'False Negative'])}")

        logger.info("\n✓ Evaluation complete!")


if __name__ == "__main__":
    main()
