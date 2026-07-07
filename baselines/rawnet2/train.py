import argparse
import os
import sys
import logging

import torch
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# Module Imports
from baselines.rawnet2.model import RawNet2
from baselines.rawnet2.trainer import train_epoch, validate
from startup_config import set_random_seed
from datasets.base_dataset import DatasetBase

log_format = "%(asctime)s %(message)s"
logging.basicConfig(
    stream=sys.stdout,
    level=logging.INFO,
    format=log_format,
    datefmt="%m/%d %I:%M:%S %p",
)


def parse_args():
    parser = argparse.ArgumentParser(description="RawNet2 Training")

    # Dataset
    parser.add_argument(
        "--database_path",
        type=str,
        required=True,
        help="Path to dataset root.",
    )

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    # Model / Experiment
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output_dir", type=str, default="./RawNet2_Baseline")
    parser.add_argument("--comment", type=str, default="RawNet2_baseline_exp")

    # Backend
    parser.add_argument(
        "--cudnn-deterministic-toggle", action="store_false", default=True
    )
    parser.add_argument("--cudnn-benchmark-toggle", action="store_true", default=False)

    return parser.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    set_random_seed(args.seed, args)

    # Model save path
    model_tag = f"model_{args.comment}_{args.num_epochs}_{args.batch_size}_{args.lr}"
    model_save_path = os.path.join(args.output_dir, model_tag)
    os.makedirs(model_save_path, exist_ok=True)

    # Handle Loggin
    fh = logging.FileHandler(os.path.join(args.output_dir, "log.txt"))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model = RawNet2(device).to(device)

    print(f"nb_params: {sum(p.numel() for p in model.parameters())}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # =========================
    # TRAIN DATA
    # =========================
    train_set = DatasetBase(
        manifest_path=os.path.join(args.database_path, "manifests/train.jsonl"),
        base_data_dir=args.database_path,
        is_training=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=True,
    )

    # =========================
    # DEV DATA
    # =========================
    dev_set = DatasetBase(
        manifest_path=os.path.join(args.database_path, "manifests/dev.jsonl"),
        base_data_dir=args.database_path,
        is_training=False,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=16,
    )

    # =========================
    # TRAIN LOOP
    # =========================
    writer = SummaryWriter(f"logs/{model_tag}")

    for epoch in range(args.num_epochs):

        train_loss, train_acc = train_epoch(train_loader, model, optimizer, device)
        val_acc, val_eer = validate(dev_loader, model, device)

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_accuracy", train_acc, epoch)
        writer.add_scalar("val_accuracy", val_acc, epoch)
        writer.add_scalar("val_eer", val_eer, epoch)

        print(
            f"{epoch} | loss={train_loss:.4f} | train_acc={train_acc:.2f} | val_acc={val_acc:.2f} | val_eer={val_eer:.2f}"
        )
        logging.info(
            f"[{epoch}] loss={train_loss} train_acc={train_acc} val_acc={val_acc} val_eer={val_eer}"
        )
        torch.save(
            model.state_dict(),
            os.path.join(model_save_path, f"epoch_{epoch}.pth"),
        )


if __name__ == "__main__":
    main()
