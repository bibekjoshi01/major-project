import os
import json
import torch
import torchaudio
from torch.utils.data import Dataset

from baselines.datasets.rtc_augment import RTCAudioSimulator


class DatasetBase(Dataset):
    def __init__(self, manifest_path, base_data_dir, max_len=64000, is_training=True):
        """
        Step 1: Manifest Parsing & Verification Layout

        Args:
            manifest_path (str): Path to train.jsonl, dev.jsonl, or eval.jsonl
            base_data_dir (str): Path to the corresponding audio folder (e.g., data/asvspoof5/audio/train)
        """
        self.base_data_dir = base_data_dir
        self.is_training = is_training
        self.max_len = max_len

        # Slicing keys for detailed RTC analysis down the road
        self.file_paths = []
        self.labels = []
        self.utterance_ids = []
        self.attack_labels = []

        print(f"Reading manifest from: {manifest_path}...")

        # Instantiate our RTC module
        self.rtc_simulator = RTCAudioSimulator()

        # 1. Read jsonl lines safely
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)

                    # 1. Combine the base dir with the relative JSON path
                    # e.g., "data/asvspoof5/" + "audio/dev/D_0000000001.flac"
                    full_path = os.path.normpath(
                        os.path.join(self.base_data_dir, item["audio_path"])
                    )
                    self.file_paths.append(full_path)

                    # 2. Extract safe items
                    self.labels.append(int(item["label"]))
                    self.utterance_ids.append(item["utt_id"])
                    self.attack_labels.append(item["attack_label"])

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

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]

        # 1. Load audio
        waveform, sr = torchaudio.load(file_path)

        # 2. Convert to mono FIRST 
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 3. Remove channel ONLY if your models expect [T]
        waveform = waveform.squeeze(0)

        # 4. Pad / truncate
        waveform_fixed = self._pad_or_truncate(waveform)

        # 5. Augmentation 
        waveform_rtc = self.rtc_simulator.process(
            waveform_fixed, is_training=self.is_training
        )

        # 6. FINAL SHAPE 
        waveform_rtc = waveform_rtc.squeeze()
        assert waveform_rtc.dim() == 1, f"Expected [T], got {waveform_rtc.shape}"

        return {
            "waveform": waveform_rtc,
            "label": torch.tensor(label, dtype=torch.long),
            "utt_id": self.utterance_ids[idx],
            "attack_label": self.attack_labels[idx],
        }
