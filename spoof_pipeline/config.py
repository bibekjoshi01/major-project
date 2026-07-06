"""
Central configuration.

To switch dataset -> change `audio_dir` / `protocol_file`.
To switch model   -> change `model_module` / `model_class` (and `model_type`,
                      `pretrained`) to point at whatever class you wrote.

Everything else in the pipeline (data loading, training loop, evaluation,
metrics) stays exactly the same regardless of what you pick here.
"""

import torch
from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------
    # 1) DATASET  -- change these two lines to point at a new dataset
    # ------------------------------------------------------------------
    audio_dir: str = "./asvspoof5_audio"
    protocol_file: str = "./asvspoof5_protocols/ASVspoof5.eval.track_1.tsv"

    # column layout of the protocol file (ASVspoof-style tsv/txt).
    # change these if a new dataset's protocol file has a different layout.
    file_id_col: int = 1
    label_col: int = 8
    bonafide_tag: str = "bonafide"

    # ------------------------------------------------------------------
    # 2) MODEL -- point this at YOUR model, nothing else needs to change
    # ------------------------------------------------------------------
    model_name: str = "aasist3"           # just a label, used for the printed report title
    model_type: str = "torch"             # "torch" (nn.Module) or "sklearn" (.fit/.predict_proba)
    model_module: str = "model"           # python import path, e.g. "model" or "my_models.next_tdnn"
    model_class: str = "SpectraAASIST3"   # class name inside that module

    pretrained: bool = True               # True -> skip training, just evaluate
    pretrained_source: str | None = "lab260/Spectra-AASIST3"   # arg passed to cls.from_pretrained(...)
    # if the module file itself needs downloading first (like the AASIST3 model.py), set this;
    # leave as None for models you already have on disk (e.g. your own next_tdnn.py)
    pretrained_module_url: str | None = "https://huggingface.co/lab260/Spectra-AASIST3/raw/main/model.py"

    model_kwargs: dict = field(default_factory=dict)   # extra kwargs forwarded to cls(**model_kwargs)

    # only used when model_type == "sklearn": dotted path to a feature_fn(waveform, sample_rate) -> np.array
    feature_fn: str = "sklearn_features.extract_mfcc_features"

    # ------------------------------------------------------------------
    # 3) DATA / TRAINING knobs (rarely need to change these)
    # ------------------------------------------------------------------
    max_files: int | None = None      # cap total files loaded, None = no cap
    val_split: float = 0.1            # fraction of data held out for validation (0 = no split)
    target_length: int = 64600        # samples per clip (pad/trim to this length)
    sample_rate: int = 16000

    batch_size: int = 64
    num_workers: int = 2
    epochs: int = 10
    lr: float = 1e-4
    seed: int = 42

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # where to save/load trained model weights
    checkpoint_dir: str = "./checkpoints"