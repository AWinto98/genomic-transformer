"""Masked-language-model pretraining for the small genomic transformer.

Wraps the SmallTransformer encoder stack (transformer.py) with a per-position
head and trains it to predict masked tokens: 15% of positions are selected,
of which 80% are replaced with N (token 4, which doubles as MASK since the
alphabet has no separate mask symbol), 10% with a random base, and 10% left
unchanged. Loss is masked-token cross-entropy (ignore_index=-100).

The corpus is built directly from a FASTA reference: non-overlapping windows
(stride=window) over the given chromosomes, skipping windows whose N-fraction
exceeds the limit, with an optional seeded subsample for pilot-scale runs.

Encoder weights transfer cleanly to the classifier: MLMModel keeps the
encoder as a submodule named `encoder`, so its state-dict keys are exactly
SmallTransformer's keys prefixed with "encoder." (mlm_head.* is extra).
train.py --init-from strips that prefix and loads with strict=False.

Artifacts are written to --output-dir with the same conventions as train.py:
- <out-name>_pretrain_metrics.json (final metrics document)
- <out-name>_pretrain_model.pt (state_dict + model_config + mlm_config)
- <out-name>_pretrain_checkpoint.pt (last-epoch state for --resume, atomic)
- <out-name>_pretrain_training_log.jsonl (one JSON object per epoch)
- <out-name>_pretrain_status.json (atomically overwritten per-epoch pulse)

All trainer output is flushed and prefixed with an [HH:MM:SS] elapsed
timestamp so redirected logs stay live during long runs.
"""

import argparse
import dataclasses
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pyfaidx
import torch
from torch import nn

from genomic_transformer.config import load_config
from genomic_transformer.data.prepare_ctcf import encode_sequence
from genomic_transformer.models.train import (
    _apply_mps_watermark,
    _autocast,
    _log,
    _lr_lambda,
    _write_json_atomic,
    resolve_device,
    resolve_precision,
)
from genomic_transformer.models.transformer import (
    VOCAB_SIZE,
    SmallTransformer,
    TransformerConfig,
    count_parameters,
)

MASK_RATE = 0.15
MASK_TOKEN = 4  # N doubles as the mask token; the alphabet has no separate MASK

_METRICS_SUFFIX = "_pretrain_metrics.json"
_MODEL_SUFFIX = "_pretrain_model.pt"
_CHECKPOINT_SUFFIX = "_pretrain_checkpoint.pt"
_TRAINING_LOG_SUFFIX = "_pretrain_training_log.jsonl"
_STATUS_SUFFIX = "_pretrain_status.json"

_ENCODING = {
    "alphabet": "ACGTN",
    "tokens": {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4},
    "input_case": "upper",
    "unknown_character": "N",
    "dtype": "uint8",
}


class MLMModel(nn.Module):
    """SmallTransformer encoder stack plus a per-position MLM head.

    forward(input_ids [B, L]) -> logits [B, L, vocab]. The encoder is kept
    as a submodule so encoder weights share SmallTransformer's state-dict
    names under an "encoder." prefix; only mlm_head is extra.
    """

    def __init__(self, config: Union[dict[str, Any], TransformerConfig]) -> None:
        super().__init__()
        self.config = TransformerConfig.from_dict(config)
        self.encoder = SmallTransformer(self.config)
        # The classification head is unused for MLM and never enters the
        # forward path; dropping it keeps the state dict a clean prefix map
        # onto SmallTransformer (every encoder.* key has a classifier namesake
        # except head, which stays at random init on --init-from).
        del self.encoder.head
        self.mlm_head = nn.Linear(self.config.d_model, VOCAB_SIZE)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be [batch, length], got shape {tuple(input_ids.shape)}"
            )
        length = input_ids.shape[1]
        if length > self.config.max_len:
            raise ValueError(
                f"sequence length {length} exceeds max_len {self.config.max_len}"
            )
        positions = torch.arange(length, device=input_ids.device)
        x = self.encoder.token_embed(input_ids) + self.encoder.pos_embed(positions)[None, :, :]
        x = self.encoder.dropout(x)
        for block in self.encoder.blocks:
            x = block(x)
        x = self.encoder.norm(x)
        return self.mlm_head(x)


def build_corpus(
    fasta_path: Union[str, Path],
    chroms: Optional[list[str]],
    window: int,
    stride: int,
    max_n_fraction: float = 0.05,
    limit: Optional[int] = None,
    seed: int = 42,
) -> np.ndarray:
    """Tokenize a FASTA reference into [N, window] uint8 windows.

    Windows start at 0, stride, 2*stride, ... while a full window fits.
    Windows whose N-fraction (token 4, including unknown characters) exceeds
    max_n_fraction are dropped. When limit < available, a seeded permutation
    selects the subset (deterministic for a given seed); otherwise the
    genome order (chroms order, then position) is kept.
    """
    if window <= 0 or stride <= 0:
        raise ValueError(f"window and stride must be positive, got {window}/{stride}")
    fasta = pyfaidx.Fasta(str(fasta_path))
    try:
        names = list(chroms) if chroms is not None else list(fasta.keys())
        chunks = []
        for name in names:
            if name not in fasta.keys():
                raise ValueError(f"chromosome {name!r} not present in {fasta_path}")
            sequence = str(fasta[name][:]).upper()
            if len(sequence) < window:
                continue
            starts = range(0, len(sequence) - window + 1, stride)
            windows = np.stack(
                [encode_sequence(sequence[start : start + window], _ENCODING) for start in starts]
            )
            n_fraction = (windows == MASK_TOKEN).mean(axis=1)
            chunks.append(windows[n_fraction <= max_n_fraction])
    finally:
        fasta.close()
    if not chunks:
        return np.empty((0, window), dtype=np.uint8)
    corpus = np.concatenate(chunks)
    if limit is not None and limit < len(corpus):
        rng = np.random.default_rng(seed)
        corpus = corpus[rng.permutation(len(corpus))[:limit]]
    return np.ascontiguousarray(corpus)


def mask_batch(
    tokens: np.ndarray, seed_rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply MLM masking to [B, L] uint8 tokens.

    Returns (inputs, targets): inputs is the masked uint8 array; targets is
    an int64 array holding the original token at masked positions and -100
    elsewhere (nn.CrossEntropyLoss ignore_index). Of the masked positions,
    80% become N(4), 10% a random base 0-3, 10% stay unchanged.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 2:
        raise ValueError(f"tokens must be [batch, length], got shape {tokens.shape}")
    draw = seed_rng.random(tokens.shape)
    masked = draw < MASK_RATE
    policy = seed_rng.random(tokens.shape)
    random_base = seed_rng.integers(0, 4, size=tokens.shape).astype(np.uint8)
    inputs = tokens.copy()
    to_mask = masked & (policy < 0.8)
    inputs[to_mask] = MASK_TOKEN
    to_random = masked & (policy >= 0.8) & (policy < 0.9)
    inputs[to_random] = random_base[to_random]
    targets = np.full(tokens.shape, -100, dtype=np.int64)
    targets[masked] = tokens[masked]
    return inputs, targets


def _status_payload(
    epoch: int,
    total_epochs: int,
    train_loss: Optional[float],
    last_epoch_seconds: Optional[float],
    finished: bool,
    final_metrics: Optional[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "epoch": epoch,
        "total_epochs": total_epochs,
        "train_loss": train_loss,
        "last_epoch_seconds": last_epoch_seconds,
        "finished": finished,
        "final_metrics": final_metrics,
    }


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    epoch: int,
    rng_state: dict[str, Any],
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    mlm_config: dict[str, Any],
) -> None:
    payload = {
        "model_state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "epoch": epoch,
        "rng_state": rng_state,
        "model_config": model_config,
        "training_config": dict(training_config),
        "mlm_config": dict(mlm_config),
    }
    tmp_path = path.parent / (path.name + ".tmp")
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)


def run_pretraining(
    fasta_path: Union[str, Path],
    model_config_path: Union[str, Path],
    out_name: str = "mlm",
    output_dir: Union[str, Path] = "data/processed/pretrain",
    windows: int = 1_500_000,
    epochs: int = 1,
    batch_size: int = 64,
    lr: float = 3.0e-4,
    seed: int = 42,
    device: str = "auto",
    resume: bool = False,
    chroms: Optional[list[str]] = None,
    window: int = 1000,
    stride: Optional[int] = None,
    max_n_fraction: float = 0.05,
) -> dict[str, Any]:
    """Build a corpus from FASTA, run MLM pretraining, persist artifacts."""
    start_time = time.monotonic()
    fasta_path = Path(fasta_path)
    model_config_path = Path(model_config_path)
    model_config_file = load_config(model_config_path)
    model_dict = dict(model_config_file.get("model") or {})
    configured_training = dict(model_config_file.get("training") or {})
    stride = window if stride is None else stride

    torch.manual_seed(seed)
    np.random.seed(seed)  # CPU is used for deterministic tests.

    device = resolve_device(device)
    _log(start_time, f"device: {device}")
    if device.type == "mps":
        _apply_mps_watermark(configured_training.get("mps_high_watermark_ratio"))
    requested_precision = str(configured_training.get("precision", "fp32")).lower()
    precision = resolve_precision(configured_training.get("precision"), device)
    if precision == requested_precision:
        _log(start_time, f"precision: {precision}")
    else:
        _log(
            start_time,
            f"precision: {precision} "
            f"(requested {requested_precision} downgraded on {device.type})",
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _log(
        start_time,
        f"building corpus: fasta={fasta_path} window={window} stride={stride} "
        f"max_n_fraction={max_n_fraction} limit={windows} "
        f"chroms={chroms if chroms is not None else 'all'}",
    )
    corpus = build_corpus(
        fasta_path,
        chroms,
        window,
        stride,
        max_n_fraction=max_n_fraction,
        limit=windows,
        seed=seed,
    )
    if len(corpus) == 0:
        raise ValueError(f"corpus is empty: {fasta_path} yielded no usable windows")
    _log(start_time, f"corpus: {len(corpus)} windows of {window} tokens")

    config = TransformerConfig.from_dict(model_dict)
    if window > config.max_len:
        raise ValueError(
            f"window {window} exceeds model max_len {config.max_len}; "
            "the model config and corpus window must agree"
        )
    model = MLMModel(config).to(device)
    training = {
        "precision": requested_precision,
        "lr": float(lr),
        "batch_size": int(batch_size),
        "weight_decay": float(configured_training.get("weight_decay", 0.01)),
        "warmup_steps": int(configured_training.get("warmup_steps", 0)),
        "grad_clip": float(configured_training.get("grad_clip", 1.0)),
        "heartbeat_steps": max(1, int(configured_training.get("heartbeat_steps", 200))),
        "seed": int(seed),
    }
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training["lr"],
        weight_decay=training["weight_decay"],
    )
    steps_per_epoch = max(1, math.ceil(len(corpus) / training["batch_size"]))
    total_steps = max(1, epochs * steps_per_epoch)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(training["warmup_steps"], total_steps)
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    _log(start_time, f"steps/epoch: {steps_per_epoch} (batch {training['batch_size']})")
    _log(start_time, f"mlm model: {count_parameters(model)} params")

    mlm_config: dict[str, Any] = {
        "mask_rate": MASK_RATE,
        "corpus_windows": int(len(corpus)),
        "window": int(window),
        "stride": int(stride),
        "max_n_fraction": float(max_n_fraction),
        "chroms": [str(c) for c in (chroms if chroms is not None else [])],
        "epochs": int(epochs),
        "batch_size": training["batch_size"],
        "steps_per_epoch": steps_per_epoch,
        "seed": int(seed),
    }

    checkpoint_path = output_dir / f"{out_name}{_CHECKPOINT_SUFFIX}"
    log_path = output_dir / f"{out_name}{_TRAINING_LOG_SUFFIX}"
    status_path = output_dir / f"{out_name}{_STATUS_SUFFIX}"

    rng = np.random.default_rng(seed)
    start_epoch = 1
    optimizer_steps_done = 0

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
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        rng.bit_generator.state = checkpoint["rng_state"]
        start_epoch = int(checkpoint["epoch"]) + 1
        optimizer_steps_done = int(checkpoint.get("mlm_config", {}).get("optimizer_steps", 0))
        _log(start_time, f"resumed from {checkpoint_path} at epoch {start_epoch}")
    elif resume:
        _log(start_time, f"no checkpoint at {checkpoint_path}; starting from scratch")

    last_epoch = 0
    last_train_loss: Optional[float] = None
    last_epoch_seconds: Optional[float] = None
    initial_train_loss: Optional[float] = None
    log_mode = "a" if (resume and checkpoint_path.exists()) else "w"
    with log_path.open(log_mode, encoding="utf-8") as log_stream:
        for epoch in range(start_epoch, epochs + 1):
            epoch_start = time.monotonic()
            model.train()
            running_loss = 0.0
            seen = 0
            optimizer_step = 0
            last_batch_loss = 0.0
            order = rng.permutation(len(corpus))
            for batch_number, start in enumerate(range(0, len(corpus), training["batch_size"])):
                batch_tokens = corpus[order[start : start + training["batch_size"]]]
                inputs_np, targets_np = mask_batch(batch_tokens, rng)
                inputs = torch.from_numpy(inputs_np).long().to(device)
                targets = torch.from_numpy(targets_np).long().to(device)
                with _autocast(device, precision):
                    logits = model(inputs)
                loss = loss_fn(logits.float().view(-1, VOCAB_SIZE), targets.view(-1))
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), training["grad_clip"])
                optimizer.step()
                scheduler.step()
                optimizer_step += 1
                optimizer_steps_done += 1
                last_batch_loss = loss.item()
                running_loss += last_batch_loss * len(batch_tokens)
                seen += len(batch_tokens)
                if optimizer_step % training["heartbeat_steps"] == 0:
                    _log(
                        start_time,
                        f"epoch {epoch} step {optimizer_step}/{steps_per_epoch} "
                        f"loss={last_batch_loss:.4f}",
                    )
            train_loss = running_loss / max(1, seen)
            if initial_train_loss is None:
                initial_train_loss = train_loss
            epoch_seconds = time.monotonic() - epoch_start
            last_epoch = epoch
            last_train_loss = train_loss
            last_epoch_seconds = epoch_seconds
            _log(
                start_time,
                f"epoch {epoch}/{epochs} train_loss={train_loss:.4f} "
                f"lr={float(scheduler.get_last_lr()[0]):.3g} "
                f"epoch_seconds={epoch_seconds:.1f}",
            )
            log_stream.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "train_loss": train_loss,
                        "lr": float(scheduler.get_last_lr()[0]),
                        "epoch_seconds": epoch_seconds,
                        "optimizer_steps": optimizer_steps_done,
                    }
                )
                + "\n"
            )
            log_stream.flush()
            _write_json_atomic(
                status_path,
                _status_payload(
                    epoch,
                    epochs,
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
                scheduler,
                epoch,
                dict(rng.bit_generator.state),
                dataclasses.asdict(config),
                training,
                mlm_config,
            )

    mlm_config["optimizer_steps"] = optimizer_steps_done
    final_metrics: dict[str, Any] = {
        "train_loss": last_train_loss,
        "initial_train_loss": initial_train_loss,
        "optimizer_steps": optimizer_steps_done,
    }
    _write_json_atomic(
        status_path,
        _status_payload(
            last_epoch,
            epochs,
            last_train_loss,
            last_epoch_seconds,
            finished=True,
            final_metrics=final_metrics,
        ),
    )

    metrics_doc: dict[str, Any] = {
        "config": {
            "path": str(model_config_path),
            "fasta": str(fasta_path),
            "out_name": out_name,
            "model": dataclasses.asdict(config),
            "training": training,
            "mlm": mlm_config,
            "n_parameters": count_parameters(model),
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "device": str(device),
            "mps_available": bool(torch.backends.mps.is_available()),
        },
        "corpus": {
            "windows": int(len(corpus)),
            "window": int(window),
            "stride": int(stride),
            "max_n_fraction": float(max_n_fraction),
            "chroms": [str(c) for c in (chroms if chroms is not None else [])],
        },
        "metrics": final_metrics,
    }

    metrics_path = output_dir / f"{out_name}{_METRICS_SUFFIX}"
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics_doc, stream, indent=2, sort_keys=True)
        stream.write("\n")
    model_path = output_dir / f"{out_name}{_MODEL_SUFFIX}"
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": dataclasses.asdict(config),
            "mlm_config": dict(mlm_config),
        },
        model_path,
    )

    _log(start_time, f"train_loss: {initial_train_loss:.4f} -> {last_train_loss:.4f}")
    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {model_path}", flush=True)
    return metrics_doc


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Masked-language-model pretraining of the small genomic transformer"
    )
    parser.add_argument("--fasta", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--out-name", default="mlm")
    parser.add_argument("--output-dir", default=Path("data/processed/pretrain"), type=Path)
    parser.add_argument(
        "--windows",
        type=int,
        default=1_500_000,
        help="corpus limit: seeded subsample when below the available window count",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3.0e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "mps", "cuda", "cpu"]
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from <out-name>_pretrain_checkpoint.pt if it exists "
        "(epochs are total, not additional)",
    )
    parser.add_argument(
        "--chroms",
        default=None,
        help="comma-separated chromosome list; default is every sequence in the FASTA",
    )
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument(
        "--stride",
        type=int,
        default=None,
        help="window start spacing; default is --window (non-overlapping)",
    )
    parser.add_argument("--max-n-fraction", type=float, default=0.05)
    args = parser.parse_args(argv)
    chroms = [name.strip() for name in args.chroms.split(",") if name.strip()] if args.chroms else None
    run_pretraining(
        args.fasta,
        args.config,
        out_name=args.out_name,
        output_dir=args.output_dir,
        windows=args.windows,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        resume=args.resume,
        chroms=chroms,
        window=args.window,
        stride=args.stride,
        max_n_fraction=args.max_n_fraction,
    )


if __name__ == "__main__":
    main()
