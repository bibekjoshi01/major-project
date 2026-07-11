import os
import json
import random
import warnings
import torch
import torchaudio
from torch.utils.data import Dataset

from datasets.channel_profiles import build_default_profiles
from datasets.rtc_augment import RTCAudioSimulator


class DatasetBase(Dataset):
    def __init__(
        self,
        manifest_path,
        base_data_dir,
        max_len=16000,
        is_training=True,
        sample_rate=16000,
        rtc_config=None,
    ):
        """
        Step 1: Manifest Parsing & Verification Layout

        Args:
            manifest_path (str): Path to train.jsonl, dev.jsonl, or eval.jsonl
            base_data_dir (str): Path to the corresponding audio folder (e.g., data/asvspoof5/audio/train)
            sample_rate (int): Target sample rate; files loaded at a different
                native rate are resampled to this before augmentation/padding.
            rtc_config (dict): Optional overrides for RTCAudioSimulator (enabled,
                clean_prob, active_profiles, degrade_eval, ffmpeg_bin, ffmpeg_timeout).
        """
        self.base_data_dir = base_data_dir
        self.is_training = is_training
        self.max_len = max_len
        self.target_sample_rate = sample_rate

        # Slicing keys for detailed RTC analysis down the road
        self.file_paths = []
        self.labels = []
        self.utterance_ids = []
        self.attack_labels = []

        print(f"Reading manifest from: {manifest_path}...")

        # Instantiate our RTC module
        rtc_config = rtc_config or {}
        profiles = build_default_profiles(database_path=base_data_dir)
        active_profiles = rtc_config.get("active_profiles")

        if active_profiles:
            profiles = {k: v for k, v in profiles.items() if k in active_profiles}

        self.rtc_simulator = RTCAudioSimulator(
            sample_rate=sample_rate,
            profiles=profiles,
            clean_prob=float(rtc_config.get("clean_prob", 0.18)),
            enabled=bool(rtc_config.get("enabled", True)),
            degrade_eval=bool(rtc_config.get("degrade_eval", False)),
            ffmpeg_bin=rtc_config.get("ffmpeg_bin", "ffmpeg"),
            ffmpeg_timeout=float(rtc_config.get("ffmpeg_timeout", 8.0)),
        )

        # Build a set of files actually on disk in one directory walk. A
        # per-line os.path.isfile() would cost one stat() per manifest row
        # (hundreds of thousands for the full protocol) even though a
        # partial Colab download only has a small fraction of those files;
        # scanning once and doing in-memory set lookups instead turns that
        # into O(files on disk) + O(manifest rows) with no syscalls in the loop.
        audio_root = os.path.join(self.base_data_dir, "audio")
        existing_files = set()
        for dirpath, _, filenames in os.walk(audio_root):
            for fname in filenames:
                existing_files.add(os.path.normpath(os.path.join(dirpath, fname)))

        # 1. Read jsonl lines safely
        skipped_missing = 0
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)

                    # 1. Combine the base dir with the relative JSON path
                    # e.g., "data/asvspoof5/" + "audio/dev/D_0000000001.flac"
                    full_path = os.path.normpath(
                        os.path.join(self.base_data_dir, item["audio_path"])
                    )

                    # Real-time Colab downloads can be interrupted mid-extraction,
                    # leaving manifest entries without a matching file under
                    # audio/train or audio/dev on disk. Only index entries whose
                    # audio is actually present instead of loading the full
                    # jsonl and discovering gaps later.
                    if full_path not in existing_files:
                        skipped_missing += 1
                        continue

                    self.file_paths.append(full_path)

                    # 2. Extract safe items
                    self.labels.append(int(item["label"]))
                    self.utterance_ids.append(item["utt_id"])
                    self.attack_labels.append(item["attack_label"])

        if skipped_missing:
            print(
                f"Skipped {skipped_missing} manifest entries with no audio file on disk."
            )

        if not self.file_paths:
            raise RuntimeError(
                f"No audio files found under '{audio_root}' matching any entry in "
                f"{manifest_path}. Run download_audio.py for this split before "
                f"training/evaluating on it, e.g.: "
                f"python data/asvspoof5/download_audio.py --split "
                f"{os.path.basename(manifest_path).split('.')[0]}"
            )

        print(f"Successfully processed {len(self.file_paths)} track records.")
        print(f"--> Target Path 0: {self.file_paths[0]}")
        print(f"--> Target Label 0: {self.labels[0]} (Type: {type(self.labels[0])})")

    def __len__(self):
        return len(self.file_paths)

    def _pad_or_truncate(self, waveform):
        """
        Enforces a uniform sample shape dimension across variable-length utterances.
        """
        num_samples = waveform.shape[1]

        # Scenario A: Audio is longer than target length -> Truncate
        if num_samples >= self.max_len:
            return waveform[:, : self.max_len]

        # Scenario B: Audio is shorter -> Repeat audio cyclically to fill the buffer
        else:
            repeats = (self.max_len // num_samples) + 1
            waveform_repeated = waveform.repeat(1, repeats)
            return waveform_repeated[:, : self.max_len]

    def __getitem__(self, idx, remove_channel=True, _retries=5):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 1. Load audio
        # Real-time Colab downloads can be interrupted mid-extraction, leaving
        # some manifest entries without a matching file on disk. Skip those
        # instead of killing an entire training run hours in.
        try:
            waveform, native_sr = torchaudio.load(file_path)
        except Exception as e:
            if _retries <= 0:
                raise RuntimeError(
                    f"Repeated audio load failures; last error on {file_path}: {e}"
                ) from e
            warnings.warn(f"Skipping unreadable/missing audio file: {file_path} ({e})")
            fallback_idx = random.randrange(len(self.file_paths))
            return self.__getitem__(
                fallback_idx, remove_channel=remove_channel, _retries=_retries - 1
            )

        # 2. Resample to the target rate if the file's native rate differs.
        # Downstream RTC channel simulation needs a genuinely correct input
        # rate for its codec/resample math, not a silently assumed one.
        if native_sr != self.target_sample_rate:
            waveform = torchaudio.functional.resample(
                waveform, native_sr, self.target_sample_rate
            )

        # 3. Convert to mono BEFORE augmentation: real phones capture/encode
        # mono, so a channel simulation should run on the same signal a real
        # VoIP call would actually carry, not per-channel-then-averaged.
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 4. RTC/VoIP channel augmentation
        waveform_rtc, channel_profile = self.rtc_simulator.process(
            waveform, sample_rate=self.target_sample_rate, is_training=self.is_training
        )

        # 5. Pad / truncate
        waveform_fixed = self._pad_or_truncate(waveform_rtc)

        # 6. Remove channel ONLY if your models expect [T]
        if remove_channel:
            waveform_fixed = waveform_fixed.squeeze(0)

        # 7. FINAL SHAPE
        waveform_fixed = waveform_fixed.squeeze()
        assert waveform_fixed.dim() == 1, f"Expected [T], got {waveform_fixed.shape}"

        return {
            "waveform": waveform_fixed,
            "label": torch.tensor(label, dtype=torch.long),
            "utt_id": self.utterance_ids[idx],
            # Bonafide records store attack_label=None in the manifest; the
            # default collate_fn crashes on a batch whose first item is None,
            # which real training would hit as soon as a batch starts with a
            # bonafide sample. Use a sentinel string instead.
            "attack_label": (
                self.attack_labels[idx]
                if self.attack_labels[idx] is not None
                else "bonafide"
            ),
            "channel_profile": channel_profile,
        }
