import os

import torch

from datasets.channel_profiles import build_default_profiles
from datasets.rtc_augment import RTCAudioSimulator


def test_rtc_profiles_preserve_shape():
    """Exercises ffmpeg_codec_round_trip + GilbertElliottLossSimulator for every
    profile on a synthetic tone -- no real ASVspoof5 audio required."""
    sample_rate = 16000
    t = torch.linspace(0, 1.0, sample_rate)
    tone = (0.3 * torch.sin(2 * torch.pi * 220 * t)).unsqueeze(0)

    profiles = build_default_profiles()
    for name, profile in profiles.items():
        sim = RTCAudioSimulator(sample_rate=sample_rate, profiles={name: profile}, clean_prob=0.0)
        out, applied = sim.process(tone, sample_rate=sample_rate, is_training=True)
        assert out.shape == tone.shape, f"{name}: shape mismatch {out.shape} vs {tone.shape}"
        assert applied == name
        assert torch.isfinite(out).all(), f"{name}: non-finite samples in output"
    print(f"OK: {len(profiles)} channel profiles preserve shape: {list(profiles)}")


def test_rtc_fallback_without_ffmpeg():
    """ffmpeg must never crash a training run: point at a nonexistent binary
    and confirm the simulator still returns a valid tensor."""
    sample_rate = 16000
    tone = torch.sin(torch.linspace(0, 20 * torch.pi, sample_rate)).unsqueeze(0)

    sim = RTCAudioSimulator(
        sample_rate=sample_rate,
        clean_prob=0.0,
        ffmpeg_bin="ffmpeg-does-not-exist",
    )
    out, applied = sim.process(tone, sample_rate=sample_rate, is_training=True)
    assert out.shape == tone.shape
    assert torch.isfinite(out).all()
    print(f"OK: ffmpeg-unavailable fallback returned a valid tensor (profile={applied})")


def test_dataset_with_real_manifest():
    """Manual smoke test against the real ASVspoof5 manifest/audio, if present."""
    manifest_path = "data/asvspoof5/manifests/dev.jsonl"
    if not os.path.exists(manifest_path):
        print(f"Skipping real-manifest smoke test: {manifest_path} not found.")
        return

    from datasets.base_dataset import DatasetBase
    from torch.utils.data import DataLoader

    ds = DatasetBase(manifest_path=manifest_path, base_data_dir="data/asvspoof5")
    dl = DataLoader(ds, batch_size=4, shuffle=True)

    for batch in dl:
        print("Waveform Batch Shape:", batch["waveform"].shape)
        print("Labels Batch Tensor:", batch["label"])
        print("Channel Profiles:", batch["channel_profile"])
        break


if __name__ == "__main__":
    test_rtc_profiles_preserve_shape()
    test_rtc_fallback_without_ffmpeg()
    test_dataset_with_real_manifest()

# RUN: python -m datasets.tests.test_base_dataset
