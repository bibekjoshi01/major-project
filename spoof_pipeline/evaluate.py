"""
Evaluation utilities for speech anti-spoofing models.

This module is completely independent of the underlying model.
Whether the model is PyTorch, Scikit-learn, AASIST, RawNet2,
Random Forest, or SVM, the evaluation procedure remains the same.

Metrics
-------
- Accuracy
- Precision
- Recall (Sensitivity)
- F1-Score
- Equal Error Rate (EER)
"""

import numpy as np

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)


def compute_eer(y_true, y_scores):
    """
    Compute the Equal Error Rate (EER).

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth labels.

    y_scores : numpy.ndarray
        Bonafide confidence scores.

    Returns
    -------
    eer : float
        Equal Error Rate (percentage).

    threshold : float
        Decision threshold corresponding to the EER.
    """

    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_scores,
        pos_label=1,
    )

    fnr = 1 - tpr

    idx = np.nanargmin(np.abs(fpr - fnr))

    eer = (fpr[idx] + fnr[idx]) / 2

    return eer * 100, thresholds[idx]


def compute_metrics(y_true, y_scores):
    """
    Compute evaluation metrics.

    Parameters
    ----------
    y_true : numpy.ndarray
        Ground-truth labels.

    y_scores : numpy.ndarray
        Bonafide confidence scores.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """

    eer, threshold = compute_eer(
        y_true,
        y_scores,
    )

    y_pred = (y_scores >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        )
        * 100,
        "precision": precision_score(
            y_true,
            y_pred,
        )
        * 100,
        "recall": recall_score(
            y_true,
            y_pred,
        )
        * 100,
        "f1": f1_score(
            y_true,
            y_pred,
        )
        * 100,
        "eer": eer,
        "threshold": threshold,
    }

    return metrics


def print_report(
    metrics,
    title="SPEECH ANTI-SPOOFING MODEL PERFORMANCE",
):
    """
    Print a formatted evaluation report.
    """

    print("\n" + "=" * 60)
    print(title.center(60))
    print("=" * 60)

    print(f"Accuracy                  : {metrics['accuracy']:.2f}%")
    print(f"Precision                 : {metrics['precision']:.2f}%")
    print(f"Recall (Sensitivity)      : {metrics['recall']:.2f}%")
    print(f"F1-Score                  : {metrics['f1']:.2f}%")
    print(f"Equal Error Rate (EER)    : {metrics['eer']:.2f}%")
    print(f"Decision Threshold        : {metrics['threshold']:.4f}")

    print("=" * 60)
