import os
import tarfile
import subprocess
from huggingface_hub import hf_hub_download


def download_and_extract(
    repo_id: str,
    archive_name: str,
    audio_dir: str = "./asvspoof5_audio",
    repo_type: str = "dataset",
):
    """
    Download an archive from Hugging Face Hub and extract it.

    Args:
        repo_id (str):
            Hugging Face repository ID.
            Example: "jungjee/asvspoof5"

        archive_name (str):
            Name of the archive file to download.
            Example: "flac_E_aa.tar"

        audio_dir (str, optional):
            Directory where the archive will be extracted.
            Default: "./asvspoof5_audio"

        repo_type (str, optional):
            Hugging Face repository type.
            Default: "dataset"
    """

    print("=== Step 1: Downloading archive from Hugging Face Hub ===")
    print(f"Downloading '{archive_name}'...")

    archive_path = hf_hub_download(
        repo_id=repo_id,
        filename=archive_name,
        repo_type=repo_type,
    )

    print("\n=== Step 2: Extracting archive ===")
    print(f"Extracting into '{audio_dir}'...")

    os.makedirs(audio_dir, exist_ok=True)

    with tarfile.open(archive_path, "r") as tar:
        tar.extractall(path=audio_dir)

    print("\nDataset setup completed successfully!")


if __name__ == "__main__":
    try:
        import huggingface_hub
    except ImportError:
        print("huggingface_hub library not found. Installing it now...")
        subprocess.check_call(["pip", "install", "huggingface_hub"])

    # Example usage
    download_and_extract(
        repo_id="jungjee/asvspoof5",
        archive_name="flac_E_aa.tar",
        audio_dir="./asvspoof5_audio",
    )