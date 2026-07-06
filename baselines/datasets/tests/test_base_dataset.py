from baselines.datasets.base_dataset import DatasetBase
from torch.utils.data import DataLoader

ds = DatasetBase(
    manifest_path="data/asvspoof5/manifests/dev.jsonl", base_data_dir="data/asvspoof5"
)

dl = DataLoader(ds, batch_size=4, shuffle=True)

# Fetch a single batch sample stack
for batch in dl:
    print("Waveform Batch Shape:", batch["waveform"].shape)
    print("Waveform :", batch["waveform"])
    print("Labels Batch Tensor:", batch["label"])
    break

# RUN: python -m baselines.datasets.tests.test_base_dataset