"""
Evaluate a Spectra-AASIST3 checkpoint on ASVspoof5 using the RTC-aware
DatasetBase pipeline (manifest.jsonl -> waveform, label, utt_id, attack_label).

Examples
--------

From the HF hub:
    python evaluate.py \
        --database_path data/asvspoof5 \
        --manifest data/asvspoof5/manifests/eval.jsonl \
        --hf_repo lab260/Spectra-AASIST3 \
        --eval_output scores/eval_scores.txt \
        --metrics_output scores/metrics.json \
        --plot_path scores/eer_curve.png \
        --batch_size 64 \
        --num_workers 2 \
        --max_len 16000 \
        --max_eval_files 2000

Local checkpoint:
    python evaluate.py \
        --database_path data/asvspoof5 \
        --manifest data/asvspoof5/manifests/eval.jsonl \
        --model_path checkpoints/spectra_aasist3.pth \
        --eval_output scores/eval_scores.txt \
        --metrics_output scores/metrics.json \
        --plot_path scores/eer_curve.png \
        --batch_size 64 \
        --num_workers 2 \
        --max_len 16000 \
        --max_eval_files 2000
"""

import argparse
import json
import os

import torch
from torch.utils.data import Subset

from datasets.base_dataset import DatasetBase
from .inference import (
    build_dataloader,
    run_inference,
    write_score_file,
)
from .metrics import evaluate_all
from .model_utils import load_model
from .metrics import plot_eer_curve, print_report


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate Spectra-AASIST3 on ASVspoof5"
    )

    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="Root directory containing the dataset.",
    )

    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Path to eval.jsonl. Defaults to " "<database_path>/manifests/eval.jsonl",
    )

    parser.add_argument(
        "--hf_repo",
        type=str,
        default=None,
        help="Hugging Face repo id, e.g. lab260/Spectra-AASIST3",
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Local checkpoint (.pth).",
    )

    parser.add_argument(
        "--eval_output",
        type=str,
        required=True,
        help="Output score file (<utt_id> <score>).",
    )

    parser.add_argument(
        "--metrics_output",
        type=str,
        default=None,
        help="Optional JSON file to save evaluation metrics.",
    )

    parser.add_argument(
        "--plot_path",
        type=str,
        default=None,
        help="Optional output path for the EER curve.",
    )

    parser.add_argument(
        "--max_eval_files",
        type=int,
        default=None,
        help="Evaluate only the first N files.",
    )

    parser.add_argument(
        "--max_len",
        type=int,
        default=16000,
        help="Maximum waveform length.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""

    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()

    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load model
    model = load_model(
        device=device,
        model_path=args.model_path,
        hf_repo=args.hf_repo,
    )

    # Dataset
    manifest_path = (
        args.manifest
        if args.manifest is not None
        else os.path.join(args.database_path, "manifests", "eval.jsonl")
    )

    eval_set = DatasetBase(
        manifest_path=manifest_path,
        base_data_dir=args.database_path,
        max_len=args.max_len,
        is_training=False,
    )

    print(f"Evaluation utterances: {len(eval_set):,}")

    if args.max_eval_files is not None and args.max_eval_files < len(eval_set):
        print(f"Capping evaluation to " f"{args.max_eval_files:,} utterances.")

        eval_set = Subset(
            eval_set,
            range(args.max_eval_files),
        )

    # DataLoader
    data_loader = build_dataloader(
        eval_set,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Inference
    print("Running inference...")

    utt_ids, attack_labels, y_true, y_scores = run_inference(
        model=model,
        data_loader=data_loader,
        device=device,
    )

    # Save scores
    os.makedirs(
        os.path.dirname(args.eval_output) or ".",
        exist_ok=True,
    )

    write_score_file(
        args.eval_output,
        utt_ids,
        y_scores,
    )

    print(f"Scores written to: {args.eval_output}")

    # Compute metrics
    metrics = evaluate_all(
        y_true,
        y_scores,
    )

    n_bonafide = int((y_true == 1).sum())
    n_spoof = int((y_true == 0).sum())

    print_report(
        metrics,
        n_bonafide,
        n_spoof,
    )

    # Save metrics
    if args.metrics_output is not None:

        os.makedirs(
            os.path.dirname(args.metrics_output) or ".",
            exist_ok=True,
        )

        with open(args.metrics_output, "w") as f:
            json.dump(
                metrics,
                f,
                indent=2,
            )

        print(f"Metrics saved to: {args.metrics_output}")

    # Plot EER curve
    if args.plot_path is not None:
        plot_eer_curve(
            y_true,
            y_scores,
            save_path=args.plot_path,
        )

        print(f"EER curve saved to: {args.plot_path}")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()
