import json
from pathlib import Path


def convert_split(tsv_path: Path, split: str, output_dir: Path):
    """
    Convert ASVspoof5 TSV metadata to JSONL manifest.

    Output format:
    {
        "utt_id": "...",
        "speaker_id": "...",
        "gender": "...",
        "audio_path": "...",
        "codec": null,
        "codec_q": null,
        "codec_seed": null,
        "attack_tag": "...",
        "attack_label": "...",
        "label": 0/1
    }
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{split}.jsonl"

    with open(tsv_path, "r", encoding="utf-8") as fin, open(
        output_path, "w", encoding="utf-8"
    ) as fout:

        for line in fin:
            line = line.strip()
            if not line:
                continue

            fields = line.split()

            speaker_id = fields[0]
            utt_id = fields[1]
            gender = fields[2]

            attack_tag = None if fields[6] == "-" else fields[6]
            attack_label = None if fields[7] == "bonafide" else fields[7]

            # spoof -> 1, bonafide -> 0
            label = 1 if fields[8] == "spoof" else 0

            record = {
                "utt_id": utt_id,
                "speaker_id": speaker_id,
                "gender": gender,
                "audio_path": f"audio/{split}/{utt_id}.flac",
                "codec": None,
                "codec_q": None,
                "codec_seed": None,
                "attack_tag": attack_tag,
                "attack_label": attack_label,
                "label": label,
            }

            fout.write(json.dumps(record) + "\n")

    print(f"✓ Saved {output_path}")


def main():
    root = Path(".")
    ROOT = Path(__file__).resolve().parent

    PROTOCOL_DIR = ROOT / "protocols"
    MANIFEST_DIR = ROOT / "manifests"

    splits = {
        "train": PROTOCOL_DIR / "train.tsv",
        "dev": PROTOCOL_DIR / "dev.tsv",
        "eval": PROTOCOL_DIR / "eval.tsv",
    }

    for split, tsv in splits.items():
        convert_split(tsv, split, MANIFEST_DIR)


if __name__ == "__main__":
    main()


# RUN Command: python data/asvspoof5/convert_to_manifest.py