import argparse
import os
import shutil
import tarfile
from pathlib import Path, PurePosixPath
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import hf_hub_download
from tqdm import tqdm

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = (
    "1"  # make sure this is set before any hf_hub_download call
)

REPO_ID = "jungjee/asvspoof5"

ARCHIVES = {
    "train": [
        "flac_T_aa.tar",
        "flac_T_ab.tar",
        "flac_T_ac.tar",
        "flac_T_ad.tar",
        "flac_T_ae.tar",
    ],
    "dev": ["flac_D_aa.tar", "flac_D_ab.tar", "flac_D_ac.tar"],
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


def _extract_member(
    tar: tarfile.TarFile, member: tarfile.TarInfo, destination: Path
) -> None:
    member_path = PurePosixPath(member.name)
    parts = member_path.parts
    rel_path = Path(*parts[1:]) if len(parts) > 1 else Path(parts[0])
    if not rel_path.parts:
        return

    target_path = destination / rel_path

    if member.isdir():
        target_path.mkdir(parents=True, exist_ok=True)
        return

    if member.isreg():
        target_path.parent.mkdir(parents=True, exist_ok=True)
        if target_path.exists() or target_path.is_symlink():
            if target_path.is_dir() and not target_path.is_symlink():
                shutil.rmtree(target_path)
            else:
                target_path.unlink()
        with tar.extractfile(member) as src, open(target_path, "wb") as dst:
            if src is None:
                raise RuntimeError(f"Unable to read archive member: {member.name}")
            shutil.copyfileobj(src, dst)


def download_archive(archive_name: str, destination: Path):
    # Download only — extraction happens separately so extraction of one
    # archive isn't blocked waiting on another archive's download.
    archive_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=archive_name
    )
    return archive_name, archive_path


def extract_archive(archive_name: str, archive_path: str, destination: Path):
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "r") as tar:
        members = [m for m in tar.getmembers() if m.name not in {".", "./"}]
        for member in tqdm(
            members, desc=f"Extracting {archive_name}", unit="file", leave=False
        ):
            _extract_member(tar, member, destination)


def download_and_extract(archive_name: str, destination: Path):
    name, path = download_archive(archive_name, destination)
    print(f"Downloaded {name}, extracting...")
    extract_archive(name, path, destination)
    print(f"Finished {name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["train", "dev", "eval"])
    parser.add_argument("--archive", default=None)
    parser.add_argument(
        "--workers", type=int, default=4, help="Parallel download workers"
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    destination = root / "audio" / args.split
    archives = [args.archive] if args.archive else ARCHIVES[args.split]

    print(
        f"Downloading {len(archives)} archive(s) with {args.workers} parallel workers"
    )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_and_extract, a, destination): a for a in archives
        }
        for future in as_completed(futures):
            archive = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Failed on {archive}: {e}")

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
