from baselines.datasets.base_dataset import ASVspoof5Dataset

ds = ASVspoof5Dataset(
    manifest_path="data/asvspoof5/manifests/eval.jsonl",
    audio_dir="data/asvspoof5/audio/dev",
)
