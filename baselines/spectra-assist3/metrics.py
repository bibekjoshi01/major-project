"""
Metric helpers

Label convention: 1 = bonafide (real), 0 = spoof (fake).
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_eer(y_true, y_scores):
    """
    Returns (eer_percent, optimal_threshold) using the FPR/FNR crossover.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    optimal_threshold = thresholds[idx]
    return eer * 100, optimal_threshold


def compute_classification_metrics(y_true, y_scores, threshold):
    """Hard-decision metrics at a given score threshold."""
    y_pred = (y_scores >= threshold).astype(int)

    return {
        "accuracy": accuracy_score(y_true, y_pred) * 100,
        "precision": precision_score(y_true, y_pred, zero_division=0) * 100,
        "recall": recall_score(y_true, y_pred, zero_division=0) * 100,
        "f1": f1_score(y_true, y_pred, zero_division=0) * 100,
    }


def evaluate_all(y_true, y_scores):
    """Runs EER + hard-decision metrics together and returns one dict."""
    eer, threshold = compute_eer(y_true, y_scores)
    metrics = compute_classification_metrics(y_true, y_scores, threshold)
    metrics["eer"] = eer
    metrics["threshold"] = float(threshold)
    return metrics


def print_report(metrics, n_bonafide, n_spoof):
    print("\n" + "=" * 50)
    print("     SPEECH ANTI-SPOOFING MODEL PERFORMANCE")
    print("=" * 50)
    print(f"Bonafide (Real) files     : {n_bonafide:,}")
    print(f"Spoof (Fake) files        : {n_spoof:,}")
    print(f"Accuracy                  : {metrics['accuracy']:.2f}%")
    print(f"Precision                 : {metrics['precision']:.2f}%")
    print(f"Recall (Sensitivity)      : {metrics['recall']:.2f}%")
    print(f"F1-Score                  : {metrics['f1']:.2f}%")
    print(f"Equal Error Rate (EER)    : {metrics['eer']:.2f}%")
    print(f"Calculated Target Cut-off : {metrics['threshold']:.4f}")
    print("=" * 50)


def plot_eer_curve(y_true, y_scores, save_path=None):
    """
    Plot EER curve and saves to `save_path` if given, otherwise shows interactively.
    """
    import matplotlib

    if save_path:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer_point = fpr[idx] * 100
    eer_threshold = thresholds[idx]

    plt.figure(figsize=(9, 6))
    plt.plot(
        thresholds,
        fpr * 100,
        label="False Alarm Rate (FPR)",
        color="#ff4d4d",
        linewidth=2,
    )
    plt.plot(
        thresholds, fnr * 100, label="Miss Rate (FNR)", color="#1e90ff", linewidth=2
    )
    plt.plot(
        eer_threshold,
        eer_point,
        marker="*",
        color="gold",
        markersize=14,
        markeredgecolor="black",
        label=f"EER Sweet Spot ({eer_point:.2f}%)",
    )

    plt.title(
        "Detection Threshold Optimization (EER)", fontsize=14, fontweight="bold", pad=15
    )
    plt.xlabel("Model Score Threshold")
    plt.ylabel("Error Rate Percentage (%)")

    clean_scores = [s for s in y_scores if not np.isinf(s)]
    if clean_scores:
        plt.xlim(min(clean_scores) - 2, max(clean_scores) + 2)
    plt.ylim(-2, 102)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="upper center", fontsize=10, frameon=True, shadow=True)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"EER plot saved to {save_path}")
    else:
        plt.show()
