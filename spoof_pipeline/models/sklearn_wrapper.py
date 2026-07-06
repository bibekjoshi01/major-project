"""
Generic wrapper for classical machine learning models.

This wrapper works with any estimator that follows the Scikit-learn API:

    estimator.fit(X, y)
    estimator.predict_proba(X)

Examples
--------
- SVC (probability=True)
- RandomForestClassifier
- XGBoost
- LightGBM
- LogisticRegression
- Any custom estimator implementing the same interface.
"""

import numpy as np

from .base import BaseSpoofModel


class SklearnSpoofModel(BaseSpoofModel):
    """
    Wrapper for Scikit-learn compatible spoof detection models.
    """

    def __init__(
        self,
        estimator,
        feature_fn,
        sample_rate=16000,
    ):
        self.estimator = estimator
        self.feature_fn = feature_fn
        self.sample_rate = sample_rate

    def _to_feature_matrix(self, loader):
        """
        Convert a DataLoader into a feature matrix.

        Parameters
        ----------
        loader : DataLoader

        Returns
        -------
        X : numpy.ndarray
            Feature matrix.

        y : numpy.ndarray
            Ground-truth labels.
        """

        X = []
        y = []

        for audios, labels in loader:

            waveforms = audios.numpy()

            if hasattr(labels, "numpy"):
                labels = labels.numpy()

            for waveform, label in zip(waveforms, labels):

                features = self.feature_fn(
                    waveform,
                    sample_rate=self.sample_rate,
                )

                X.append(features)
                y.append(label)

        return np.array(X), np.array(y)

    def fit(self, train_loader, val_loader=None):
        """
        Train the estimator.
        """

        if train_loader is None:
            raise ValueError(
                "train_loader is required to train a classical ML model."
            )

        X_train, y_train = self._to_feature_matrix(
            train_loader
        )

        self.estimator.fit(X_train, y_train)

        if val_loader is not None:

            y_true, y_scores = self.predict_scores(
                val_loader
            )

            predictions = (
                y_scores >= 0.5
            ).astype(int)

            val_accuracy = (
                predictions == y_true
            ).mean()

            print(
                f"[{type(self.estimator).__name__}] "
                f"Validation Accuracy: {val_accuracy:.4f}"
            )

    def predict_scores(self, loader):
        """
        Predict bonafide confidence scores.

        Returns
        -------
        y_true : numpy.ndarray

        y_scores : numpy.ndarray
            Probability of the bonafide class.
        """

        X, y = self._to_feature_matrix(loader)

        scores = self.estimator.predict_proba(X)[:, 1]

        return np.array(y), scores

    def save(self, path):
        """
        Save the trained estimator.
        """

        import joblib

        joblib.dump(self.estimator, path)

    def load(self, path):
        """
        Load a previously saved estimator.
        """

        import joblib

        self.estimator = joblib.load(path)