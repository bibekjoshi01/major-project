"""
Dataset-agnostic loading.

This file should NOT need to change when you switch models.
It only needs to change if a brand-new dataset has a genuinely
different protocol/label format. In that case, write a new
`parse_*_protocol()` function and point `build_dataloaders()`
to it.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split


def parse_asvspoof_protocol(
    protocol_file,
    file_id_col=1,
    label_col=8,
    bonafide_tag="bonafide",
):
    """
    Parse an ASVspoof-style protocol file.

    Parameters
    ----------
    protocol_file : str
        Path to the protocol file.

    file_id_col : int
        Column containing the audio file ID.

    label_col : int
        Column containing the label.

    bonafide_tag : str
        Label representing genuine speech.

    Returns
    -------
    dict
        Dictionary mapping:
            file_id -> 1 (bonafide)
            file_id -> 0 (spoof)
    """
    labels_dict = {}

    with open(protocol_file, "r") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) > max(file_id_col, label_col):
                file_id = parts[file_id_col].strip()
                label = parts[label_col].strip()

                labels_dict[file_id] = (
                    1 if label == bonafide_tag else 0
                )

    return labels_dict


class SpoofDataset(Dataset):
    """
    Generic dataset for spoof detection.

    Works for any directory containing .flac/.wav files,
    provided a dictionary:

        {file_id : label}
    """

    def __init__(
        self,
        audio_dir,
        labels_dict,
        max_files=None,
        target_length=64600,
        extensions=("flac", "wav"),
    ):
        self.audio_dir = audio_dir
        self.labels_dict = labels_dict
        self.target_length = target_length

        # --------------------------------------------------
        # Find all audio files
        # --------------------------------------------------
        all_paths = []

        for ext in extensions:
            pattern = os.path.join(audio_dir, "**", f"*.{ext}")
            all_paths.extend(glob.glob(pattern, recursive=True))

        self.valid_files = []
        self.actual_bonafide = 0
        self.actual_spoof = 0

        for file_path in all_paths:
            file_id = os.path.splitext(
                os.path.basename(file_path)
            )[0]

            if file_id not in labels_dict:
                continue

            self.valid_files.append((file_path, file_id))

            if labels_dict[file_id] == 1:
                self.actual_bonafide += 1
            else:
                self.actual_spoof += 1

            if (
                max_files is not None
                and len(self.valid_files) >= max_files
            ):
                break

        print(
            f"[SpoofDataset] {len(self.valid_files):,} files "
            f"(bonafide={self.actual_bonafide:,}, "
            f"spoof={self.actual_spoof:,}) "
            f"from {audio_dir}"
        )

    def __len__(self):
        return len(self.valid_files)

    @staticmethod
    def _pad_trim(audio, target_length):
        """
        Pad short audio by repeating it.
        Trim long audio.
        """
        if len(audio) < target_length:
            repeats = int(np.ceil(target_length / len(audio)))
            audio = np.tile(audio, repeats)

        return audio[:target_length]

    def __getitem__(self, index):
        import soundfile as sf

        file_path, file_id = self.valid_files[index]

        audio, _ = sf.read(file_path, dtype="float32")

        audio = self._pad_trim(audio, self.target_length)

        return (
            torch.FloatTensor(audio),
            self.labels_dict[file_id],
        )


def build_dataloaders(config):
    """
    Build train, validation and evaluation dataloaders.

    Returns
    -------
    train_loader
    val_loader
    eval_loader
    """

    labels_dict = parse_asvspoof_protocol(
        protocol_file=config.protocol_file,
        file_id_col=config.file_id_col,
        label_col=config.label_col,
        bonafide_tag=config.bonafide_tag,
    )

    full_dataset = SpoofDataset(
        audio_dir=config.audio_dir,
        labels_dict=labels_dict,
        max_files=config.max_files,
        target_length=config.target_length,
    )

    eval_loader = DataLoader(
        full_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    train_loader = None
    val_loader = None

    if config.val_split and config.val_split > 0:

        n_val = int(len(full_dataset) * config.val_split)
        n_train = len(full_dataset) - n_val

        generator = torch.Generator().manual_seed(config.seed)

        train_dataset, val_dataset = random_split(
            full_dataset,
            [n_train, n_val],
            generator=generator,
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, eval_loader