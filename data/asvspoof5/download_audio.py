import argparse
import tarfile
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "jungjee/asvspoof5"

# Available archives on HF
ARCHIVES = {
    "train": [
        "flac_T_aa.tar",
        "flac_T_ab.tar",
        "flac_T_ac.tar",
        "flac_T_ad.tar",
        "flac_T_ae.tar",
    ],
    "dev": [
        "flac_D_aa.tar",
        "flac_D_ab.tar",
        "flac_D_ac.tar",
    ],
    "eval": [
        "flac_E_aa.tar",
        "flac_E_ab.tar",
        "flac_E_ac.tar",
        "flac_E_ad.tar",
        "flac_E_ae.tar",
        "flac_E_af.tar",
        "flac_E_ag.tar",
        "flac_E_ah.tar",
        "flac_E_ai.tar",
        "flac_E_aj.tar",
    ],
}


def download_archive(archive_name: str, destination: Path):
    print(f"\nDownloading {archive_name}")

    archive_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=archive_name,
    )

    print(f"Extracting -> {destination}")

    destination.mkdir(parents=True, exist_ok=True)

    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(destination)

    print(f"Finished {archive_name}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--split",
        required=True,
        choices=["train", "dev", "eval"],
        help="Dataset split",
    )

    parser.add_argument(
        "--archive",
        default=None,
        help="Specific archive name (optional)",
    )

    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    audio_root = root / "audio"
    destination = audio_root / args.split

    if args.archive:
        archives = [args.archive]
    else:
        archives = ARCHIVES[args.split]

    for archive in archives:
        download_archive(archive, destination)

    print("\nAll downloads completed.")


if __name__ == "__main__":
    main()


"""
## Running commands

1. Download all training audio
    python data/asvspoof5/download_audio.py --split train

2. Download development audio
    python data/asvspoof5/download_audio.py --split dev

3. Download all evaluation audio
    python data/asvspoof5/download_audio.py --split eval

4. Download only one archive
    python data/asvspoof5/download_audio.py --split eval --archive flac_E_aa.tar
    python data/asvspoof5/download_audio.py --split train --archive flac_T_ab.tar
"""
