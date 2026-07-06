"""
Generic wrapper around any PyTorch nn.Module that takes raw waveforms and
outputs 2-class logits (index 1 = bonafide).

AASIST3, Next-TDNN, or any future deep learning architecture all use
this same wrapper—the only thing that changes is the `net` passed in.
This makes swapping models a one-line change.
"""

import numpy as np
import torch
import torch.nn as nn

from .base import BaseSpoofModel


class TorchSpoofModel(BaseSpoofModel):
    """
    Generic wrapper for PyTorch spoof detection models.
    """

    def __init__(
        self,
        net,
        device,
        pretrained=False,
        epochs=10,
        lr=1e-4,
        optimizer_cls=torch.optim.Adam,
        criterion=None,
    ):
        self.net = net.to(device)
        self.device = device

        # If True, fit() becomes a no-op.
        self.pretrained = pretrained

        self.epochs = epochs
        self.lr = lr

        self.optimizer_cls = optimizer_cls
        self.criterion = criterion or nn.CrossEntropyLoss()

    def fit(self, train_loader, val_loader=None):
        """
        Train the model unless it is marked as pretrained.
        """

        if self.pretrained:
            print(
                f"[{type(self.net).__name__}] "
                "pretrained=True -> skipping training."
            )
            return

        if train_loader is None:
            raise ValueError(
                "train_loader is required to train a non-pretrained model. "
                "Set config.val_split > 0 or provide a training split."
            )

        optimizer = self.optimizer_cls(
            self.net.parameters(),
            lr=self.lr,
        )

        for epoch in range(1, self.epochs + 1):

            self.net.train()

            running_loss = 0.0

            for audios, labels in train_loader:

                audios = audios.to(
                    self.device,
                    non_blocking=True,
                )

                labels = labels.to(
                    self.device,
                    non_blocking=True,
                )

                optimizer.zero_grad()

                logits = self.net(audios)

                loss = self.criterion(
                    logits,
                    labels,
                )

                loss.backward()

                optimizer.step()

                running_loss += (
                    loss.item() * audios.size(0)
                )

            avg_loss = (
                running_loss / len(train_loader.dataset)
            )

            message = (
                f"Epoch {epoch}/{self.epochs} "
                f"- train_loss: {avg_loss:.4f}"
            )

            if val_loader is not None:

                y_true, y_scores = self.predict_scores(
                    val_loader
                )

                val_predictions = (
                    y_scores >= 0.5
                ).astype(int)

                val_accuracy = (
                    val_predictions == y_true
                ).mean()

                message += (
                    f" - val_acc: {val_accuracy:.4f}"
                )

            print(message)

    def predict_scores(self, loader):
        """
        Predict bonafide confidence scores.

        Returns
        -------
        y_true : numpy.ndarray
            Ground-truth labels.

        y_scores : numpy.ndarray
            Probability of the bonafide class.
        """

        self.net.eval()

        y_true = []
        y_scores = []

        with torch.no_grad():

            for audios, labels in loader:

                audios = audios.to(
                    self.device,
                    non_blocking=True,
                )

                logits = self.net(audios)

                scores = torch.softmax(
                    logits,
                    dim=1,
                )[:, 1]

                y_scores.extend(
                    scores.cpu().numpy()
                )

                if torch.is_tensor(labels):
                    y_true.extend(labels.numpy())
                else:
                    y_true.extend(labels)

        return (
            np.array(y_true),
            np.array(y_scores),
        )

    def save(self, path):
        """
        Save model parameters.
        """
        torch.save(
            self.net.state_dict(),
            path,
        )

    def load(self, path):
        """
        Load model parameters.
        """
        self.net.load_state_dict(
            torch.load(
                path,
                map_location=self.device,
            )
        )