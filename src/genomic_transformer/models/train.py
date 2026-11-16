"""Train the small transformer window classifier.

Reads the prepared windows.npz through WindowDataset, trains the pre-LN
encoder from transformer.py with AdamW + linear-warmup/cosine scheduling,
early stops on validation AUROC, and reports AUROC/AUPRC/accuracy at 0.5 on
the chromosome-held-out validation and test splits.

Artifacts are written into paths.output_dir:
- <out-name>_metrics.json (same schema shape as baseline_metrics.json)
- <out-name>_model.pt (best state_dict plus model config for reuse)
- <out-name>_checkpoint.pt (last-epoch state for --resume, atomic)
- <out-name>_training_log.jsonl (one JSON object per epoch, append-only)
- <out-name>_status.json (atomically overwritten per-epoch pulse)

All trainer output is flushed and prefixed with an [HH:MM:SS] elapsed
timestamp so redirected logs stay live during long runs.
"""

import argparse
import contextlib
import dataclasses
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import sklearn
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from genomic_transformer.config import load_config
from genomic_transformer.data.dataset import WindowDataset
from genomic_transformer.models.transformer import (
    SmallTransformer,
    TransformerConfig,
    count_parameters,
)

_SPLITS = ("train", "validation", "test")
_METRICS_SUFFIX = "_metrics.json"
_MODEL_SUFFIX = "_model.pt"
_CHECKPOINT_SUFFIX = "_checkpoint.pt"
_TRAINING_LOG_SUFFIX = "_training_log.jsonl"
_STATUS_SUFFIX = "_status.json"
_HEARTBEAT_ENV = "PYTORCH_MPS_HIGH_WATERMARK_RATIO"
_MPS_LOW_WATERMARK_ENV = "PYTORCH_MPS_LOW_WATERMARK_RATIO"


def _apply_mps_watermark(ratio: Optional[float]) -> None:
    # Cap the MPS allocator pool before any tensor allocation; a preset env var wins.
    # torch requires a low watermark <= the high one (its default 1.4 would
    # invalidate any configured ratio below that), so derive one from the
    # effective high value unless the env presets it too.
    if ratio is None:
        return
    if _HEARTBEAT_ENV not in os.environ:
        os.environ[_HEARTBEAT_ENV] = str(ratio)
    if _MPS_LOW_WATERMARK_ENV not in os.environ:
        high = float(os.environ[_HEARTBEAT_ENV])
        os.environ[_MPS_LOW_WATERMARK_ENV] = str(max(0.1, high - 0.05))


def _log(start: float, message: str) -> None:
    """Print a flushed line prefixed with [HH:MM:SS] elapsed since run start."""
    elapsed = int(time.monotonic() - start)
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"[{hours:02d}:{minutes:02d}:{seconds:02d}] {message}", flush=True)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    tmp_path = path.parent / (path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(tmp_path, path)


class WindowTensorDataset(Dataset):
    """Torch view of one split: sequences as long tokens, labels as float."""

    def __init__(self, window_dataset: WindowDataset, split: str) -> None:
        indices = window_dataset.split_indices(split)
        sequences = np.stack(
            [row["sequence"] for row in window_dataset.iter_split(split)]
        )
        self.sequences = torch.from_numpy(sequences).long()
        self.labels = torch.from_numpy(window_dataset.labels[indices]).float()

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[index], self.labels[index]


def resolve_device(choice: str) -> torch.device:
    """Map --device auto|mps|cuda|cpu onto a torch.device.

    auto prefers MPS, then CUDA, then CPU; explicit mps/cuda raise when the
    corresponding backend is unavailable.
    """
    if choice == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("--device mps requested but MPS is not available")
        return torch.device("mps")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {choice!r}; expected auto|mps|cuda|cpu")


def resolve_precision(precision: Optional[str], device: torch.device) -> str:
    """Resolve the configured precision against the device.

    bf16 autocast is only used on mps/cuda; every other device (including
    CPU, which keeps the tests deterministic) forces fp32.
    """
    value = str(precision or "fp32").lower()
    if value not in ("fp32", "bf16"):
        raise ValueError(f"unknown precision {precision!r}; expected fp32|bf16")
    if value == "bf16" and device.type not in ("mps", "cuda"):
        return "fp32"
    return value


def _autocast(device: torch.device, precision: str):
    """Forward-pass autocast context; no-op for fp32."""
    if precision == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def _lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to zero, as a LambdaLR multiplier."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return lr_lambda


def _make_loader(
    dataset: WindowTensorDataset,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed) if shuffle else None
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator)


@torch.no_grad()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    precision: str = "fp32",
) -> dict[str, float]:
    model.eval()
    logits, labels = [], []
    for sequences, targets in loader:
        with _autocast(device, precision):
            batch_logits = model(sequences.to(device))
        logits.append(batch_logits.float().cpu())
        labels.append(targets)
    probabilities = torch.sigmoid(torch.cat(logits)).numpy()
    labels_np = torch.cat(labels).numpy()
    return {
        "auroc": float(roc_auc_score(labels_np, probabilities)),
        "auprc": float(average_precision_score(labels_np, probabilities)),
        "accuracy_at_0.5": float(np.mean((probabilities >= 0.5) == labels_np)),
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val_auroc: float,
    best_epoch: int,
    epochs_without_improvement: int,
    best_state: dict[str, torch.Tensor],
    best_val_metrics: dict[str, float],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
        "epochs_without_improvement": epochs_without_improvement,
        "best_state_dict": best_state,
        "best_val_metrics": best_val_metrics,
        "model_config": model_config,
        "training_config": dict(training_config),
    }
    tmp_path = path.parent / (path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def _status_payload(
    epoch: int,
    total_epochs: int,
    best_val_auroc: float,
    best_epoch: int,
    epochs_without_improvement: int,
    train_loss: Optional[float],
    last_epoch_seconds: Optional[float],
    finished: bool,
    final_metrics: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "best_val_auroc": best_val_auroc,
        "best_epoch": best_epoch,
        "epochs_since_improvement": epochs_without_improvement,
        "train_loss": train_loss,
        "last_epoch_seconds": last_epoch_seconds,
        "finished": finished,
        "final_metrics": final_metrics,
    }


def _load_initial_weights(model: nn.Module, init_from: Path, start: float) -> None:
    """Load weights from a classifier or MLM pretrain checkpoint into model.

    Pretrain checkpoints (models/pretrain.py) save the encoder under an
    "encoder." prefix with an extra mlm_head; both are handled here: the
    prefix is stripped so encoder keys map onto SmallTransformer's names,
    and non-matching keys (mlm_head) are reported rather than loaded.
    Raises ValueError when no key matches or a matched key has a bad shape.
    """
    checkpoint = torch.load(init_from, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"init-from checkpoint at {init_from} is not a mapping")
    source = checkpoint.get("state_dict")
    if source is None:
        source = checkpoint.get("model_state_dict")
    if not isinstance(source, dict):
        raise ValueError(
            f"init-from checkpoint at {init_from} has neither "
            "'state_dict' nor 'model_state_dict'"
        )
    target_keys = set(model.state_dict().keys())
    remapped = {}
    unmatched_source = []
    for key, value in source.items():
        stripped = key[len("encoder.") :] if key.startswith("encoder.") else key
        if stripped in target_keys:
            remapped[stripped] = value
        else:
            unmatched_source.append(key)
    if not remapped:
        raise ValueError(f"no keys in {init_from} match the model parameters")
    try:
        missing, unexpected = model.load_state_dict(remapped, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            f"init-from checkpoint at {init_from} does not match model shapes: {exc}"
        ) from exc
    _log(
        start,
        f"init-from {init_from}: matched={len(remapped)} "
        f"ignored_source_keys={sorted(unmatched_source) or 'none'} "
        f"left_at_init={sorted(missing) or 'none'} "
        f"unexpected={sorted(unexpected) or 'none'}",
    )


def _freeze_encoder(model: nn.Module) -> tuple[int, int]:
    """Freeze every parameter except the head; returns (trainable, frozen) counts."""
    trainable = frozen = 0
    for name, parameter in model.named_parameters():
        if name.startswith("head."):
            trainable += parameter.numel()
        else:
            parameter.requires_grad_(False)
            frozen += parameter.numel()
    return trainable, frozen


def run_training(
    data_config_path: Union[str, Path],
    model_config_path: Union[str, Path],
    device: str = "auto",
    epochs: Optional[int] = None,
    out_name: str = "transformer",
    resume: bool = False,
    init_from: Optional[Union[str, Path]] = None,
    freeze_encoder: bool = False,
) -> dict[str, Any]:
    """Train, evaluate, and persist artifacts; returns the metrics document."""
    start_time = time.monotonic()
    data_config_path = Path(data_config_path)
    model_config_path = Path(model_config_path)
    data_config = load_config(data_config_path)
    model_config_file = load_config(model_config_path)
    model_dict = dict(model_config_file.get("model") or {})
    training = dict(model_config_file.get("training") or {})
    if epochs is not None:
        training["epochs"] = epochs
    training["freeze_encoder"] = bool(freeze_encoder)

    seed = int(training.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)  # MPS is not bitwise deterministic; CPU is used for deterministic tests.

    device = resolve_device(device)
    _log(start_time, f"device: {device}")
    if device.type == "mps":
        _apply_mps_watermark(training.get("mps_high_watermark_ratio"))
    requested_precision = str(training.get("precision", "fp32")).lower()
    precision = resolve_precision(training.get("precision"), device)
    if precision == requested_precision:
        _log(start_time, f"precision: {precision}")
    else:
        _log(
            start_time,
            f"precision: {precision} "
            f"(requested {requested_precision} downgraded on {device.type})",
        )

    dataset_path = Path(data_config["paths"]["dataset"])
    output_dir = Path(data_config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    batch_size = int(training.get("batch_size", 128))
    n_epochs = int(training.get("epochs", 40))
    patience = int(training.get("patience", 6))
    grad_clip = float(training.get("grad_clip", 1.0))
    grad_accum = max(1, int(training.get("grad_accum_steps", 1)))
    heartbeat_every = max(1, int(training.get("heartbeat_steps", 200)))

    with WindowDataset(dataset_path) as window_dataset:
        loaders = {
            split: _make_loader(
                WindowTensorDataset(window_dataset, split),
                batch_size,
                shuffle=split == "train",
                seed=seed,
            )
            for split in _SPLITS
        }
        counts = window_dataset.counts()
        class_balance = {split: window_dataset.counts(split) for split in _SPLITS}

    config = TransformerConfig.from_dict(model_dict)
    model = SmallTransformer(config).to(device)
    if init_from is not None:
        _load_initial_weights(model, Path(init_from), start_time)
    if freeze_encoder:
        trainable, frozen = _freeze_encoder(model)
        _log(
            start_time,
            f"freeze-encoder: {trainable} trainable (head) / {frozen} frozen parameters",
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training.get("lr", 3.0e-4)),
        weight_decay=float(training.get("weight_decay", 0.01)),
    )
    micro_batches = len(loaders["train"])
    steps_per_epoch = max(1, math.ceil(micro_batches / grad_accum))
    total_steps = max(1, n_epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(int(training.get("warmup_steps", 0)), total_steps)
    )
    loss_fn = nn.BCEWithLogitsLoss()

    checkpoint_path = output_dir / f"{out_name}{_CHECKPOINT_SUFFIX}"
    log_path = output_dir / f"{out_name}{_TRAINING_LOG_SUFFIX}"
    status_path = output_dir / f"{out_name}{_STATUS_SUFFIX}"

    best_state: Optional[dict[str, torch.Tensor]] = None
    best_val_metrics: Optional[dict[str, float]] = None
    best_val_auroc = 0.0
    best_epoch = 0
    epochs_without_improvement = 0
    start_epoch = 1

    if resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        checkpoint_model_config = checkpoint.get("model_config")
        if checkpoint_model_config != dataclasses.asdict(config):
            raise ValueError(
                f"checkpoint model config {checkpoint_model_config} does not match "
                f"current config {dataclasses.asdict(config)}"
            )
        checkpoint_precision = str(
            (checkpoint.get("training_config") or {}).get("precision", "fp32")
        ).lower()
        if checkpoint_precision != requested_precision:
            raise ValueError(
                f"checkpoint precision {checkpoint_precision} does not match "
                f"current config precision {requested_precision}"
            )
        checkpoint_freeze_encoder = bool(
            (checkpoint.get("training_config") or {}).get("freeze_encoder", False)
        )
        if checkpoint_freeze_encoder != freeze_encoder:
            raise ValueError(
                f"checkpoint freeze_encoder {checkpoint_freeze_encoder} does not match "
                f"current freeze_encoder {freeze_encoder}"
            )
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except RuntimeError as exc:
            raise ValueError(
                f"checkpoint at {checkpoint_path} does not match model shape: {exc}"
            ) from exc
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        for state in optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_auroc = float(checkpoint["best_val_auroc"])
        best_epoch = int(checkpoint["best_epoch"])
        epochs_without_improvement = int(checkpoint.get("epochs_without_improvement", 0))
        best_val_metrics = checkpoint.get("best_val_metrics")
        best_state = checkpoint.get("best_state_dict")
        _log(
            start_time,
            f"resumed from {checkpoint_path} at epoch {start_epoch} "
            f"(best val_auroc={best_val_auroc:.4f}, best epoch {best_epoch})",
        )
    elif resume:
        _log(start_time, f"no checkpoint at {checkpoint_path}; starting from scratch")

    last_epoch = 0
    last_train_loss: Optional[float] = None
    last_epoch_seconds: Optional[float] = None
    log_mode = "a" if (resume and checkpoint_path.exists()) else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_stream:
        for epoch in range(start_epoch, n_epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            optimizer.zero_grad()
            running_loss = 0.0
            seen = 0
            optimizer_step = 0
            last_micro_loss = 0.0
            for micro_index, (sequences, targets) in enumerate(loaders["train"]):
                sequences = sequences.to(device)
                targets = targets.to(device)
                with _autocast(device, precision):
                    logits = model(sequences)
                loss = loss_fn(logits.float(), targets)
                (loss / grad_accum).backward()
                last_micro_loss = loss.item()
                running_loss += last_micro_loss * sequences.shape[0]
                seen += sequences.shape[0]
                is_optimizer_step = (micro_index + 1) % grad_accum == 0 or (
                    micro_index + 1
                ) == micro_batches
                if is_optimizer_step:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    optimizer_step += 1
                    if optimizer_step % heartbeat_every == 0:
                        _log(
                            start_time,
                            f"epoch {epoch} step {optimizer_step}/{steps_per_epoch} "
                            f"loss={last_micro_loss:.4f}",
                        )
            train_loss = running_loss / max(1, seen)
            epoch_seconds = time.monotonic() - epoch_start
            last_epoch = epoch
            last_train_loss = train_loss
            last_epoch_seconds = epoch_seconds

            val_metrics = _evaluate(model, loaders["validation"], device, precision)
            improved = best_val_metrics is None or (
                val_metrics["auroc"] > best_val_metrics["auroc"]
            )
            if improved:
                best_val_metrics = val_metrics
                best_val_auroc = val_metrics["auroc"]
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            marker = "*" if improved else ""
            _log(
                start_time,
                f"epoch {epoch}/{n_epochs} train_loss={train_loss:.4f} "
                f"val_auroc={val_metrics['auroc']:.4f} best={best_val_auroc:.4f}{marker} "
                f"epoch_seconds={epoch_seconds:.1f}",
            )
            log_stream.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "val_auroc": val_metrics["auroc"],
                        "val_auprc": val_metrics["auprc"],
                        "val_accuracy": val_metrics["accuracy_at_0.5"],
                        "lr": float(scheduler.get_last_lr()[0]),
                        "epoch_seconds": epoch_seconds,
                        "best_val_auroc": best_val_auroc,
                        "epochs_since_improvement": epochs_without_improvement,
                    }
                )
                + "\n"
            )
            log_stream.flush()
            _write_json_atomic(
                status_path,
                _status_payload(
                    epoch,
                    n_epochs,
                    best_val_auroc,
                    best_epoch,
                    epochs_without_improvement,
                    train_loss,
                    epoch_seconds,
                    finished=False,
                    final_metrics=None,
                ),
            )
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                epoch,
                best_val_auroc,
                best_epoch,
                epochs_without_improvement,
                best_state,
                best_val_metrics,
                dataclasses.asdict(config),
                training,
            )
            if not improved and epochs_without_improvement >= patience:
                _log(start_time, f"early stop after epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is None or best_val_metrics is None:  # pragma: no cover
        raise RuntimeError("no epoch produced a validation AUROC")
    model.load_state_dict(best_state)
    test_metrics = _evaluate(model, loaders["test"], device, precision)
    final_metrics = {"validation": best_val_metrics, "test": test_metrics}
    _write_json_atomic(
        status_path,
        _status_payload(
            last_epoch,
            n_epochs,
            best_val_auroc,
            best_epoch,
            epochs_without_improvement,
            last_train_loss,
            last_epoch_seconds,
            finished=True,
            final_metrics=final_metrics,
        ),
    )

    splits_config = data_config.get("splits") or {}
    metrics_doc: dict[str, Any] = {
        "config": {
            "path": str(model_config_path),
            "data_config": str(data_config_path),
            "dataset": str(dataset_path),
            "validation_chroms": [str(c) for c in splits_config.get("validation", [])],
            "test_chroms": [str(c) for c in splits_config.get("test", [])],
            "model": dataclasses.asdict(config),
            "training": training,
            "n_parameters": count_parameters(model),
            "best_epoch": best_epoch,
            "init_from": str(init_from) if init_from is not None else None,
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "counts": counts,
        "class_balance": {
            split: {"negatives": balance[0], "positives": balance[1]}
            for split, balance in class_balance.items()
        },
        "metrics": final_metrics,
    }

    metrics_path = output_dir / f"{out_name}{_METRICS_SUFFIX}"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics_doc, stream, indent=2, sort_keys=True)
        stream.write("\n")
    model_path = output_dir / f"{out_name}{_MODEL_SUFFIX}"
    torch.save(
        {"state_dict": best_state, "model_config": dataclasses.asdict(config)},
        model_path,
    )

    _print_summary(metrics_doc, metrics_path, model_path, start_time)
    return metrics_doc


def _print_summary(
    metrics: dict[str, Any], metrics_path: Path, model_path: Path, start: float
) -> None:
    config = metrics["config"]
    _log(
        start,
        f"small transformer: {config['n_parameters']} params, "
        f"dataset={config['dataset']}, device={metrics['versions']['device']}",
    )
    print(
        f"{'split':<12}{'windows':>10}{'neg/pos':>12}{'AUROC':>10}{'AUPRC':>10}{'ACC@0.5':>10}",
        flush=True,
    )
    for split in _SPLITS:
        balance = metrics["class_balance"][split]
        line = (
            f"{split:<12}{metrics['counts'][split]:>10}"
            f"{balance['negatives']:>6}/{balance['positives']:<6}"
        )
        if split in metrics["metrics"]:
            values = metrics["metrics"][split]
            line += (
                f"{values['auroc']:>10.4f}{values['auprc']:>10.4f}"
                f"{values['accuracy_at_0.5']:>10.4f}"
            )
        print(line, flush=True)
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {model_path}", flush=True)


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the small transformer window classifier"
    )
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--model-config", required=True, type=Path)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--out-name", default="transformer")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <out-name>_checkpoint.pt if it exists (epochs are total, not additional)",
    )
    parser.add_argument(
        "--init-from",
        default=None,
        type=Path,
        help="optional checkpoint (classifier or MLM pretrain format) whose encoder "
        "weights initialize the model before training",
    )
    parser.add_argument(
        "--freeze-encoder",
        action="store_true",
        help="freeze all parameters except the classifier head before training",
    )
    args = parser.parse_args(argv)
    run_training(
        args.data_config,
        args.model_config,
        device=args.device,
        epochs=args.epochs,
        out_name=args.out_name,
        resume=args.resume,
        init_from=args.init_from,
        freeze_encoder=args.freeze_encoder,
    )


if __name__ == "__main__":
    main()
