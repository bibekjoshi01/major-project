import argparse
from pathlib import Path

from baselines.aasist import model as aasist_model
from baselines.aasist.trainer import forward_logits
from baselines.aasist.utils import (
    build_loader,
    collect_scores,
    get_device,
    load_config,
    load_model_state,
    set_seed,
    write_scores,
)
from baselines.common import compute_eer


def build_model(config):
    model_cls = getattr(aasist_model, "AASIST", getattr(aasist_model, "Model", None))
    if model_cls is None:
        raise AttributeError("Expected baselines.aasist.model to define AASIST or Model")
    return model_cls(config["model_config"])


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate AASIST baseline")
    parser.add_argument("--config", type=str, default="baselines/aasist/aasist.conf")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="eval", choices=["train", "dev", "eval"])
    parser.add_argument("--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    seed = int(config.get("seed", 1234))
    set_seed(
        seed,
        config.get("cudnn_deterministic_toggle", True),
        config.get("cudnn_benchmark_toggle", False),
    )
    config["training"] = {**config.get("training", {}), "freq_aug": False}

    device = get_device(config)
    model = build_model(config).to(device)
    checkpoint = load_model_state(model, args.checkpoint, device)
    if checkpoint.get("epoch") is not None:
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']}: {args.checkpoint}")
    else:
        print(f"Loaded checkpoint: {args.checkpoint}")

    loader = build_loader(config, args.split, is_training=False, seed=seed)
    scores, labels, preds, utt_ids = collect_scores(
        loader, model, device, forward_logits, config
    )

    output_path = Path(args.output or config.get("eval_output", f"{args.split}_scores.txt"))
    write_scores(output_path, utt_ids, scores, labels, preds)

    accuracy = 100.0 * (preds == labels).mean()
    eer = compute_eer(
        scores, labels, positive_label=int(config.get("metrics", {}).get("positive_label", 1))
    )
    print(f"Scores: {output_path}")
    print(f"{args.split} accuracy={accuracy:.2f} eer={eer:.3f}")


if __name__ == "__main__":
    main()
