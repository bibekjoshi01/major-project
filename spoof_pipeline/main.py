"""
Main entry point for the speech anti-spoofing pipeline.

Usage
-----
    python main.py

To switch datasets or models, edit `config.py`.
No changes to this file are required.
"""

from config import Config
from data.dataset import build_dataloaders
from evaluate import compute_metrics, print_report
from models.loader import build_model


def run(config: Config):
    """
    Execute the complete spoof detection pipeline.

    Steps
    -----
    1. Build dataloaders.
    2. Build the selected model.
    3. Train the model (if required).
    4. Predict bonafide confidence scores.
    5. Compute evaluation metrics.
    6. Print the performance report.

    Parameters
    ----------
    config : Config
        Configuration object.

    Returns
    -------
    dict
        Evaluation metrics.
    """

    print("=" * 70)
    print(f"Model  : {config.model_name}")
    print(f"Class  : {config.model_module}.{config.model_class}")
    print(f"Device : {config.device}")
    print("=" * 70)

    # Build datasets and dataloaders
    train_loader, val_loader, eval_loader = build_dataloaders(config)

    # Build model
    model = build_model(config)

    # Train (or skip if pretrained)
    model.fit(train_loader, val_loader)

    # Evaluate
    y_true, y_scores = model.predict_scores(eval_loader)

    metrics = compute_metrics(
        y_true,
        y_scores,
    )

    print_report(
        metrics,
        title=f"{config.model_name.upper()} PERFORMANCE",
    )

    return metrics


if __name__ == "__main__":
    config = Config()
    run(config)
