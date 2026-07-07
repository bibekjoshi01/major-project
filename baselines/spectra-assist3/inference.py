"""
Runs the model over the evaluation set and collects per-utterance scores.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader


def build_dataloader(dataset, batch_size=64, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )


@torch.no_grad()
def run_inference(model, data_loader, device):
    """
    Executes batched inference and returns parallel lists/arrays:
        utt_ids, attack_labels, y_true, y_scores

    y_scores is the bonafide-class score.
    """
    model.eval()

    utt_ids = []
    attack_labels = []
    y_true = []
    y_scores = []

    n_batches = len(data_loader)
    print("Inference started...")

    for batch_idx, batch in enumerate(data_loader):
        waveform = batch["waveform"].to(device, non_blocking=True)

        logits = model(waveform)
        scores = logits[:, 1].detach().cpu().numpy()

        utt_ids.extend(batch["utt_id"])
        attack_labels.extend(batch["attack_label"])
        y_true.extend(batch["label"].numpy().tolist())
        y_scores.extend(scores.tolist())

        if (batch_idx + 1) % 20 == 0 or (batch_idx + 1) == n_batches:
            done = min(
                (batch_idx + 1) * data_loader.batch_size, len(data_loader.dataset)
            )
            print(f"Processed {done:,} / {len(data_loader.dataset):,} utterances...")

    return utt_ids, attack_labels, np.array(y_true), np.array(y_scores)


def write_score_file(save_path, utt_ids, y_scores):
    """Writes a standard CM score file: '<utt_id> <score>' per line."""
    with open(save_path, "w") as fh:
        for utt_id, score in zip(utt_ids, y_scores):
            fh.write(f"{utt_id} {score}\n")
    print(f"Scores saved to {save_path}")
