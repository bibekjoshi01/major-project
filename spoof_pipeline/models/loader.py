"""
Model loader.

This is the only model-specific loading code in the pipeline.

Responsibilities
----------------
1. Import the model class specified in the configuration.
2. Instantiate the model.
3. Optionally download the model module if required.
4. Wrap the model in either:
    - TorchSpoofModel
    - SklearnSpoofModel

Because of this wrapper, `main.py` never needs to know which
model is actually being used.
"""

import importlib
import os
import shutil
import urllib.error
import urllib.request

from .sklearn_wrapper import SklearnSpoofModel
from .torch_wrapper import TorchSpoofModel


def _maybe_download(url, dest_filename):
    """
    Download a Python module if it does not already exist.

    Hugging Face URLs are downloaded using huggingface_hub when
    available. Other URLs use urllib.
    """

    if not url or os.path.exists(dest_filename):
        return

    print(f"Downloading '{dest_filename}' from:\n{url}")

    # ---------------------------------------------------------
    # Prefer Hugging Face Hub for HF URLs.
    # ---------------------------------------------------------
    if "huggingface.co" in url:
        try:
            from huggingface_hub import hf_hub_download

            # Example:
            # https://huggingface.co/user/repo/raw/main/model.py
            parts = url.split("huggingface.co/", 1)[1].split("/")

            repo_id = "/".join(parts[:2])

            # Skip: repo/raw/revision/
            filename = "/".join(parts[4:])

            cached_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
            )

            shutil.copy(cached_path, dest_filename)
            return

        except Exception as e:
            print(
                f"HuggingFace download failed ({e}). "
                "Falling back to HTTP..."
            )

    # ---------------------------------------------------------
    # Standard HTTP download
    # ---------------------------------------------------------
    try:
        urllib.request.urlretrieve(url, dest_filename)

    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
    ) as e:

        raise RuntimeError(
            f"Failed to download '{dest_filename}'.\n\n"
            f"URL: {url}\n\n"
            f"Reason:\n{e}\n\n"
            "Solutions:\n"
            "  • Check your internet connection.\n"
            "  • Verify access to huggingface.co.\n"
            "  • Download the file manually.\n"
            "  • Set pretrained_module_url=None if the "
            "file already exists."
        ) from e


def build_model(config):
    """
    Build the spoof detection model defined in config.
    """

    # ==========================================================
    # PyTorch models
    # ==========================================================
    if config.model_type == "torch":

        if config.pretrained_module_url:
            _maybe_download(
                config.pretrained_module_url,
                f"{config.model_module}.py",
            )

        module = importlib.import_module(
            config.model_module
        )

        model_class = getattr(
            module,
            config.model_class,
        )

        if (
            config.pretrained
            and config.pretrained_source
        ):
            network = model_class.from_pretrained(
                config.pretrained_source
            )
        else:
            network = model_class(
                **config.model_kwargs
            )

        return TorchSpoofModel(
            net=network,
            device=config.device,
            pretrained=config.pretrained,
            epochs=config.epochs,
            lr=config.lr,
        )

    # ==========================================================
    # Scikit-Learn models
    # ==========================================================
    elif config.model_type == "sklearn":

        module = importlib.import_module(
            config.model_module
        )

        estimator_class = getattr(
            module,
            config.model_class,
        )

        estimator = estimator_class(
            **config.model_kwargs
        )

        feature_module_name, feature_function_name = (
            config.feature_fn.rsplit(".", 1)
        )

        feature_module = importlib.import_module(
            feature_module_name
        )

        feature_function = getattr(
            feature_module,
            feature_function_name,
        )

        return SklearnSpoofModel(
            estimator=estimator,
            feature_fn=feature_function,
            sample_rate=config.sample_rate,
        )

    # ==========================================================
    # Unsupported model type
    # ==========================================================
    else:
        raise ValueError(
            f"Unknown model_type '{config.model_type}'. "
            "Expected 'torch' or 'sklearn'."
        )