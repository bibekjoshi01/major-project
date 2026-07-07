import argparse

from baselines.aasist import model as aasist_model
from baselines.aasist.trainer import train_epoch, validate
from baselines.aasist.utils import (
    build_criterion,
    build_loader,
    build_optimizer,
    build_scheduler,
    get_device,
    get_summary_writer,
    load_config,
    make_experiment_dir,
    save_checkpoint,
    set_seed,
)


def build_model(config):
    model_cls = getattr(aasist_model, "AASIST", getattr(aasist_model, "Model", None))
    if model_cls is None:
        raise AttributeError("Expected baselines.aasist.model to define AASIST or Model")
    return model_cls(config["model_config"])


def parse_args():
    parser = argparse.ArgumentParser(description="Train AASIST baseline")
    parser.add_argument("--config", type=str, default="baselines/aasist/aasist.conf")
    parser.add_argument("--output_dir", type=str, default=None)
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

    device = get_device(config)
    exp_dir = make_experiment_dir(config, args.config, args.output_dir)
    print(f"Device: {device}")
    print(f"Experiment: {exp_dir}")

    model = build_model(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_loader = build_loader(config, "train", is_training=True, seed=seed)
    dev_loader = build_loader(config, "dev", is_training=False, seed=seed)
    criterion = build_criterion(config, device)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config, steps_per_epoch=len(train_loader))
    writer = get_summary_writer(exp_dir / "tensorboard")

    best_eer = float("inf")
    num_epochs = int(config.get("training", {}).get("num_epochs", 1))
    log_path = exp_dir / "train.log"

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_epoch(
            train_loader, model, optimizer, scheduler, device, criterion, config
        )
        dev_metrics = validate(dev_loader, model, device, criterion, config)

        writer.add_scalar("loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("accuracy/train", train_metrics["accuracy"], epoch)
        writer.add_scalar("loss/dev", dev_metrics["loss"], epoch)
        writer.add_scalar("accuracy/dev", dev_metrics["accuracy"], epoch)
        writer.add_scalar("eer/dev", dev_metrics["eer"], epoch)

        line = (
            f"epoch={epoch:03d} "
            f"train_loss={train_metrics['loss']:.5f} "
            f"train_acc={train_metrics['accuracy']:.2f} "
            f"dev_loss={dev_metrics['loss']:.5f} "
            f"dev_acc={dev_metrics['accuracy']:.2f} "
            f"dev_eer={dev_metrics['eer']:.3f}"
        )
        print(line)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

        metrics = {"train": train_metrics, "dev": dev_metrics}
        save_checkpoint(
            exp_dir / "checkpoints" / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            metrics,
            config,
        )
        if dev_metrics["eer"] <= best_eer:
            best_eer = dev_metrics["eer"]
            save_checkpoint(
                exp_dir / "checkpoints" / "best.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                metrics,
                config,
            )
            print(f"Saved new best checkpoint at epoch {epoch} (EER={best_eer:.3f})")

    writer.close()
    print(f"Done. Best dev EER: {best_eer:.3f}. Checkpoints: {exp_dir / 'checkpoints'}")


if __name__ == "__main__":
    main()
