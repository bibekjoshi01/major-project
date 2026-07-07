import argparse
import os

import torch
from torch.utils.data import DataLoader

from datasets.base_dataset import DatasetBase
from model import RawNet2
from startup_config import set_random_seed


def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=64, shuffle=False, drop_last=False)
    model.eval()

    for batch_x, utt_id in data_loader:
        fname_list = []
        score_list = []
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)

        _, batch_out = model(batch_x, is_test=True)
        batch_score = (batch_out[:, 1]).data.cpu().numpy().ravel()

        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_score.tolist())

        with open(save_path, "a+") as fh:
            for f, cm in zip(fname_list, score_list):
                fh.write("{} {}\n".format(f, cm))

        fh.close()

    print("Scores saved to {}".format(save_path))


def parse_args():
    parser = argparse.ArgumentParser(description="RawNet2 Evaluation")

    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="Path to dataset root.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--eval_output",
        type=str,
        required=True,
        help="Output score file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="Random seed.",
    )

    parser.add_argument(
        "--cudnn-deterministic-toggle",
        action="store_false",
        default=True,
        help="Use cudnn deterministic.",
    )
    parser.add_argument(
        "--cudnn-benchmark-toggle",
        action="store_true",
        default=False,
        help="Use cudnn benchmark.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    # reproducibility
    set_random_seed(args.seed, args)

    # device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # model
    model = RawNet2(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    print(f"Loaded checkpoint: {args.model_path}")

    # evaluation protocol
    eval_set = DatasetBase(
        manifest_path=os.path.join(args.database_path, "manifests/eval.jsonl"),
        base_data_dir=args.database_path,
        is_training=False,
    )

    # remove existing output if present
    if os.path.exists(args.eval_output):
        os.remove(args.eval_output)

    produce_evaluation_file(
        dataset=eval_set,
        model=model,
        device=device,
        save_path=args.eval_output,
    )

    print(f"Evaluation scores saved to {args.eval_output}")


if __name__ == "__main__":
    main()
