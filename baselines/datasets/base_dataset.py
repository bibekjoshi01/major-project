import json
from torch.utils.data import Dataset


class ASVspoof5Dataset(Dataset):
    def __init__(self, manifest_path, audio_dir):
        """
        Step 1: Manifest Parsing & Verification Layout

        Args:
            manifest_path (str): Path to train.jsonl, dev.jsonl, or eval.jsonl
            audio_dir (str): Path to the corresponding audio folder (e.g., data/asvspoof5/audio/train)
        """
        self.audio_dir = audio_dir
        self.samples = []

        print(f"Reading manifest from: {manifest_path}...")

        # 1. Read jsonl lines safely
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.samples.append(json.loads(line))

        print(f"Loaded {len(self.samples)} sample records from metadata.")

        # 2. Peek at the first entry to confirm internal keys
        if len(self.samples) > 0:
            print("\n--- Manifest Sample Structure Peek ---")
            print(json.dumps(self.samples[0], indent=2))
            print("---------------------------------------\n")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # We will add audio loading and transformations here next.
        pass
