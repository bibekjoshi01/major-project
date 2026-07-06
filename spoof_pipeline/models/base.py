"""
Base interface for all spoof detection models.

Every model—whether it is AASIST, Next-TDNN, a CNN, SVM,
Random Forest, XGBoost, or any future model—must implement
this interface so that `main.py` can interact with all models
in exactly the same way.

Required methods
----------------
model.fit(train_loader, val_loader=None)
    Train the model. Pretrained models may simply do nothing.

model.predict_scores(loader)
    Return:
        y_true   : NumPy array of ground-truth labels
        y_scores : NumPy array of bonafide confidence scores
"""

from abc import ABC, abstractmethod


class BaseSpoofModel(ABC):
    """Abstract base class for all spoof detection models."""

    @abstractmethod
    def fit(self, train_loader, val_loader=None):
        """
        Train the model.

        Parameters
        ----------
        train_loader : DataLoader
            Training data.

        val_loader : DataLoader, optional
            Validation data.
        """
        pass

    @abstractmethod
    def predict_scores(self, loader):
        """
        Predict bonafide confidence scores.

        Parameters
        ----------
        loader : DataLoader

        Returns
        -------
        tuple
            (y_true, y_scores)

            y_true   : numpy.ndarray
                Ground-truth labels.

            y_scores : numpy.ndarray
                Bonafide confidence scores.
        """
        pass

    def save(self, path):
        """
        Save model weights or parameters.
        Override this method in subclasses if saving is supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement save()."
        )

    def load(self, path):
        """
        Load model weights or parameters.
        Override this method in subclasses if loading is supported.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement load()."
        )