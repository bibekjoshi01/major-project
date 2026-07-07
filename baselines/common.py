from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_config(config: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)


def str_to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"y", "yes", "t", "true", "on", "1"}:
        return True
    if value in {"n", "no", "f", "false", "off", "0"}:
        return False
    raise ValueError(f"Invalid boolean value: {value}")


def set_seed(seed: int, cudnn_deterministic=True, cudnn_benchmark=False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = str_to_bool(cudnn_deterministic)
        torch.backends.cudnn.benchmark = str_to_bool(cudnn_benchmark)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_device(config: dict) -> torch.device:
    requested = config.get("device", "auto")
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(requested)


def make_experiment_dir(config: dict, config_path: str, output_dir: Optional[str]) -> Path:
    train_config = config.get("training", {})
    root = Path(output_dir or config.get("output_dir", "baselines/runs"))
    exp_name = train_config.get("experiment_name")
    if not exp_name:
        stem = Path(config_path).stem
        exp_name = f"{stem}_ep{train_config.get('num_epochs', 1)}_bs{train_config.get('batch_size', 1)}"

    exp_dir = root / exp_name
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (exp_dir / "scores").mkdir(parents=True, exist_ok=True)
    save_config(config, exp_dir / "config.conf")
    return exp_dir


def build_dataset(config: dict, split: str, is_training: bool) -> DatasetBase:
    from datasets.base_dataset import DatasetBase

    data_config = config["data"]
    manifest_name = data_config.get("manifests", {}).get(split, f"{split}.jsonl")
    manifest_path = Path(data_config["database_path"]) / "manifests" / manifest_name
    return DatasetBase(
        manifest_path=str(manifest_path),
        base_data_dir=data_config["database_path"],
        max_len=int(data_config.get("max_len", 64600)),
        is_training=is_training,
    )


def build_loader(config: dict, split: str, is_training: bool, seed: int) -> DataLoader:
    train_config = config.get("training", {})
    dataset = build_dataset(config, split, is_training)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=int(train_config.get("batch_size", 32)),
        shuffle=is_training,
        num_workers=int(train_config.get("num_workers", 4)),
        drop_last=is_training and str_to_bool(train_config.get("drop_last", True)),
        pin_memory=str_to_bool(train_config.get("pin_memory", True)),
        worker_init_fn=seed_worker,
        generator=generator if is_training else None,
    )


def build_optimizer(model: torch.nn.Module, config: dict) -> torch.optim.Optimizer:
    optim_config = config.get("optimizer", {})
    name = optim_config.get("name", "adam").lower()
    lr = float(optim_config.get("lr", optim_config.get("base_lr", 1e-4)))
    weight_decay = float(optim_config.get("weight_decay", 0.0))

    if name == "adam":
        return torch.optim.Adam(
            model.parameters(),
            lr=lr,
            betas=tuple(optim_config.get("betas", [0.9, 0.999])),
            weight_decay=weight_decay,
            amsgrad=str_to_bool(optim_config.get("amsgrad", False)),
        )
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=float(optim_config.get("momentum", 0.9)),
            weight_decay=weight_decay,
            nesterov=str_to_bool(optim_config.get("nesterov", False)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def build_scheduler(
    optimizer: torch.optim.Optimizer, config: dict, steps_per_epoch: int
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    optim_config = config.get("optimizer", {})
    scheduler_name = optim_config.get("scheduler", "none").lower()
    epochs = int(config.get("training", {}).get("num_epochs", 1))

    if scheduler_name in {"none", "null", ""}:
        return None
    if scheduler_name == "cosine":
        total_steps = max(1, epochs * steps_per_epoch)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=float(optim_config.get("lr_min", 0.0)),
        )
    if scheduler_name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(optim_config.get("step_size", 10)),
            gamma=float(optim_config.get("gamma", 0.5)),
        )
    raise ValueError(f"Unsupported scheduler: {scheduler_name}")


def build_criterion(config: dict, device: torch.device) -> nn.Module:
    loss_config = config.get("loss", {})
    weights = loss_config.get("class_weights")
    if weights is not None:
        weights = torch.tensor(weights, dtype=torch.float32, device=device)
    return nn.CrossEntropyLoss(weight=weights)


def compute_eer(scores: np.ndarray, labels: np.ndarray, positive_label: int = 1) -> float:
    target_scores = scores[labels == positive_label]
    nontarget_scores = scores[labels != positive_label]
    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return 0.0

    all_scores = np.concatenate([target_scores, nontarget_scores])
    trial_labels = np.concatenate(
        [np.ones(target_scores.size), np.zeros(nontarget_scores.size)]
    )
    order = np.argsort(all_scores, kind="mergesort")
    trial_labels = trial_labels[order]

    target_cumsum = np.cumsum(trial_labels)
    nontarget_cumsum = nontarget_scores.size - (
        np.arange(1, len(all_scores) + 1) - target_cumsum
    )
    frr = np.concatenate([[0.0], target_cumsum / target_scores.size])
    far = np.concatenate([[1.0], nontarget_cumsum / nontarget_scores.size])
    idx = np.nanargmin(np.abs(frr - far))
    return float(np.mean([frr[idx], far[idx]]) * 100.0)


def run_epoch(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    forward_fn: Callable,
    criterion: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    config: Optional[dict] = None,
    desc: str = "Train",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0
    all_scores = []
    all_labels = []

    grad_clip = float((config or {}).get("training", {}).get("grad_clip", 5.0))
    positive_label = int((config or {}).get("metrics", {}).get("positive_label", 1))

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc, leave=False):
            waveforms = batch["waveform"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True).long().view(-1)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            logits = forward_fn(model, waveforms, config or {})
            loss = criterion(logits, labels)

            if is_train:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            batch_size = labels.size(0)
            total += batch_size
            total_loss += loss.item() * batch_size
            total_correct += (preds == labels).sum().item()
            all_scores.append(probs[:, positive_label].detach().cpu())
            all_labels.append(labels.detach().cpu())

    scores = torch.cat(all_scores).numpy()
    labels_np = torch.cat(all_labels).numpy()
    return {
        "loss": total_loss / max(1, total),
        "accuracy": 100.0 * total_correct / max(1, total),
        "eer": compute_eer(scores, labels_np, positive_label=positive_label),
    }


def collect_scores(
    loader: DataLoader,
    model: torch.nn.Module,
    device: torch.device,
    forward_fn: Callable,
    config: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Iterable[str]]:
    model.eval()
    positive_label = int(config.get("metrics", {}).get("positive_label", 1))
    scores, labels, preds, utt_ids = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluate", leave=False):
            waveforms = batch["waveform"].to(device, non_blocking=True)
            logits = forward_fn(model, waveforms, config)
            probs = F.softmax(logits, dim=1)

            scores.append(probs[:, positive_label].cpu())
            preds.append(torch.argmax(logits, dim=1).cpu())
            labels.append(batch["label"].long().view(-1).cpu())
            utt_ids.extend(batch["utt_id"])

    return (
        torch.cat(scores).numpy(),
        torch.cat(labels).numpy(),
        torch.cat(preds).numpy(),
        utt_ids,
    )


def write_scores(
    output_path: Path,
    utt_ids: Iterable[str],
    scores: np.ndarray,
    labels: np.ndarray,
    preds: np.ndarray,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("utt_id score label prediction\n")
        for utt_id, score, label, pred in zip(utt_ids, scores, labels, preds):
            f.write(f"{utt_id} {score:.8f} {int(label)} {int(pred)}\n")


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    epoch: int,
    metrics: dict,
    config: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def load_model_state(model: torch.nn.Module, checkpoint_path: str, device: torch.device) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    return checkpoint if isinstance(checkpoint, dict) else {}


class NullSummaryWriter:
    def add_scalar(self, *args, **kwargs):
        return None

    def close(self):
        return None


def get_summary_writer(log_dir: Path):
    try:
        from torch.utils.tensorboard import SummaryWriter

        return SummaryWriter(str(log_dir))
    except Exception:
        print("TensorBoard is not available; continuing without event logging.")
        return NullSummaryWriter()
