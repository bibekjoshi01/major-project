"""
Default feature extraction functions for classical machine learning models.

These functions are used when `config.model_type == "sklearn"`.

Any feature extraction function must follow the interface:

    feature_fn(
        waveform: np.ndarray,
        sample_rate: int
    ) -> np.ndarray
"""

import numpy as np


def extract_mfcc_features(
    waveform,
    sample_rate=16000,
    n_mfcc=20,
):
    """
    Extract MFCC-based features from an audio waveform.

    The returned feature vector consists of:
        - Mean of each MFCC coefficient
        - Standard deviation of each MFCC coefficient

    Parameters
    ----------
    waveform : numpy.ndarray
        Audio waveform.

    sample_rate : int, default=16000
        Sampling rate of the waveform.

    n_mfcc : int, default=20
        Number of MFCC coefficients.

    Returns
    -------
    numpy.ndarray
        Feature vector of length 2 * n_mfcc.
    """

    import librosa

    mfcc = librosa.feature.mfcc(
        y=waveform,
        sr=sample_rate,
        n_mfcc=n_mfcc,
    )

    mfcc_mean = mfcc.mean(axis=1)
    mfcc_std = mfcc.std(axis=1)

    return np.concatenate(
        [
            mfcc_mean,
            mfcc_std,
        ]
    )