import sys

import torch
from torch import nn
import torch.nn.functional as F

# Module Imports
from utils import compute_eer


def validate(dev_loader, model, device):
    model.eval()

    num_correct = 0.0
    num_total = 0.0

    scores_all = []
    labels_all = []

    with torch.no_grad():
        for batch in dev_loader:
            batch_x = batch["waveform"]
            batch_y = batch["label"]

            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()

            batch_size = batch_x.size(0)
            num_total += batch_size

            _, batch_out = model(batch_x)

            # PREDICTION
            # =========================
            _, batch_pred = batch_out.max(dim=1)

            num_correct += (batch_pred == batch_y).sum().item()

            # =========================
            # SCORES FOR EER
            # =========================
            score = F.softmax(batch_out, dim=1)[:, 1]

            scores_all.append(score.detach().cpu())
            labels_all.append(batch_y.detach().cpu())

        # SAFE EER COMPUTATION
        # =========================
        scores_all = torch.cat(scores_all).numpy()
        labels_all = torch.cat(labels_all).numpy()

        pos_scores = scores_all[labels_all == 1]
        neg_scores = scores_all[labels_all == 0]

        assert set(labels_all.tolist()) <= {0, 1}, "Invalid label space"

        if len(pos_scores) == 0 or len(neg_scores) == 0:
            val_eer = 0.0
        else:
            val_eer = compute_eer(pos_scores, neg_scores)[0]

        val_accuracy = (num_correct / num_total) * 100

        return val_accuracy, val_eer * 100


def train_epoch(train_loader, model, optimizer, device):
    """Training Loop"""

    running_loss = 0
    num_correct = 0.0
    num_total = 0.0
    ii = 0

    model.train()

    # Set Objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)

    for batch in train_loader:
        batch_x = batch["waveform"]
        batch_y = batch["label"]

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device).long()

        batch_size = batch_x.size(0)
        num_total += batch_size
        ii += 1

        _, batch_out = model(batch_x)
        batch_loss = criterion(batch_out, batch_y)

        _, batch_pred = batch_out.max(dim=1)

        num_correct += (batch_pred == batch_y).sum().item()
        running_loss += batch_loss.item() * batch_size

        if ii % 10 == 0:
            sys.stdout.write(f"\r\t Acc: {(num_correct / num_total) * 100:.2f}")

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 5.0
        )  # prevent exploding gradients
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()

    running_loss /= num_total
    train_accuracy = (num_correct / num_total) * 100
    return running_loss, train_accuracy
