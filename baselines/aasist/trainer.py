import torch

from baselines.aasist.utils import str_to_bool
from baselines.common import run_epoch


def forward_logits(model: torch.nn.Module, waveforms: torch.Tensor, config: dict) -> torch.Tensor:
    freq_aug = str_to_bool(config.get("training", {}).get("freq_aug", False))
    _, logits = model(waveforms, Freq_aug=freq_aug)
    return logits


def train_epoch(train_loader, model, optimizer, scheduler, device, criterion, config):
    return run_epoch(
        train_loader,
        model,
        device,
        forward_logits,
        criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        desc="Train",
    )


def validate(dev_loader, model, device, criterion, config):
    eval_config = dict(config)
    eval_config["training"] = {**config.get("training", {}), "freq_aug": False}
    return run_epoch(
        dev_loader,
        model,
        device,
        forward_logits,
        criterion,
        optimizer=None,
        scheduler=None,
        config=eval_config,
        desc="Dev",
    )
