"""Extract an importance motif from a trained window classifier.

Two methods are available:

- ``saliency`` (default): gradient×input saliency of the positive-class
  logit over the positive windows of a prepared split.
- ``rollout``: attention-rollout positional importance — per-head attention
  weights captured by replaying each block with need_weights=True, averaged
  per layer, residual folded in as 0.5*(A + I), composed across layers, and
  reduced to the attention received per key position.

Both accumulate the per-position importance into a position×base motif
matrix and render a sequence-logo-style PNG of the window center. On a
well-trained model the known CTCF motif should emerge around the window
centers.

Checkpoint contract matches train.py's <slug>_model.pt payload:
{"state_dict": <best state dict>, "model_config": <TransformerConfig asdict>},
loaded with torch.load(..., weights_only=True).

CLI:
    python -m genomic_transformer.interpret \
        --data-config config/data/ctcf_gm12878.yaml \
        --model data/processed/ctcf_gm12878/transformer_cloud-004_model.pt \
        --split validation --n-windows 2000 \
        --out-prefix data/processed/ctcf_gm12878/transformer_cloud-004

    # attention rollout instead of saliency:
    #     ... same flags ... --method rollout
    # writes <prefix>_rollout.{npy,json,png} instead of <prefix>_motif.*
"""

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from genomic_transformer.config import load_config
from genomic_transformer.data.dataset import WindowDataset
from genomic_transformer.models.transformer import SmallTransformer, TransformerConfig

N_TOKENS = 4  # A=0, C=1, G=2, T=3; token 4 (N) is excluded from the motif
_BASES = ("A", "C", "G", "T")
_BASE_COLORS = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
_CENTER_PROFILE_BP = 200
_LOGO_HALF_WIDTH = 40


def resolve_device(choice: str) -> torch.device:
    """Map --device auto|cuda|mps|cpu onto a torch.device.

    Unlike train.py's laptop-first order, auto here prefers CUDA, then MPS,
    then CPU: interpretation runs are expected on the cloud GPU. Explicit
    cuda/mps raise when the corresponding backend is unavailable.
    """
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("--device mps requested but MPS is not available")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {choice!r}; expected auto|cuda|mps|cpu")


def load_model(
    checkpoint_path: Union[str, Path], device: Union[str, torch.device]
) -> Tuple[SmallTransformer, TransformerConfig]:
    """Build a SmallTransformer from a train.py model checkpoint.

    Returns the model in eval mode on the device, plus the stored
    TransformerConfig.
    """
    checkpoint_path = Path(checkpoint_path)
    device = torch.device(device)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload or "model_config" not in payload:
        raise ValueError(
            f"{checkpoint_path} is not a train.py model checkpoint "
            f"(expected a mapping with 'state_dict' and 'model_config' keys)"
        )
    model_config = TransformerConfig.from_dict(payload["model_config"])
    model = SmallTransformer(model_config)
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        raise ValueError(
            f"checkpoint at {checkpoint_path} does not match model shape: {exc}"
        ) from exc
    model.to(device)
    model.eval()
    return model, model_config


def _token_embedding(model: nn.Module) -> nn.Embedding:
    """Locate the token embedding module, robust to attribute renaming."""
    embed = getattr(model, "token_embed", None)
    if isinstance(embed, nn.Embedding):
        return embed
    for name, module in model.named_modules():
        lowered = name.lower()
        if isinstance(module, nn.Embedding) and ("token" in lowered or "embed" in lowered):
            return module
    raise ValueError("could not locate the token embedding module on the model")


def _forward_from_embeddings(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Replay the SmallTransformer forward pass given token embeddings.

    Mirrors transformer.py's forward from the embedding step onward, so the
    gradient w.r.t. the injected embedding tensor is the gradient w.r.t. the
    token embeddings.
    """
    required = ("pos_embed", "dropout", "blocks", "norm", "head")
    missing = [name for name in required if getattr(model, name, None) is None]
    if missing:
        raise ValueError(
            f"model is missing modules required for saliency replay: {missing}"
        )
    if getattr(model, "pooling", "mean") != "mean":
        raise ValueError(
            f"pooling must be 'mean' for saliency replay, got {model.pooling!r}"
        )
    length = x.shape[1]
    positions = torch.arange(length, device=x.device)
    x = x + model.pos_embed(positions)[None, :, :]
    x = model.dropout(x)
    for block in model.blocks:
        x = block(x)
    x = model.norm(x)
    return model.head(x.mean(dim=1)).squeeze(-1)


def _forward_with_attention(
    model: nn.Module, input_ids: torch.Tensor
) -> Tuple[torch.Tensor, list[torch.Tensor]]:
    """Replay SmallTransformer.forward while capturing per-head attention.

    Mirrors transformer.py's forward exactly — token + positional embedding,
    dropout, pre-LN blocks, final norm, mean-pool, head — except each block's
    attention is invoked as ``attn(normed, normed, normed, need_weights=True,
    average_attn_weights=False)`` so the raw per-head probability matrices
    [batch, heads, length, length] are returned alongside the logit. In eval
    mode the returned logit matches ``model(input_ids)`` to float tolerance;
    tests/test_interpret.py pins that parity on CPU.
    """
    required = ("token_embed", "pos_embed", "dropout", "blocks", "norm", "head")
    missing = [name for name in required if getattr(model, name, None) is None]
    if missing:
        raise ValueError(
            f"model is missing modules required for attention capture: {missing}"
        )
    if getattr(model, "pooling", "mean") != "mean":
        raise ValueError(
            f"pooling must be 'mean' for attention capture, got {model.pooling!r}"
        )
    length = input_ids.shape[1]
    positions = torch.arange(length, device=input_ids.device)
    x = model.token_embed(input_ids) + model.pos_embed(positions)[None, :, :]
    x = model.dropout(x)
    attn_weights: list[torch.Tensor] = []
    for block in model.blocks:
        normed = block.norm1(x)
        attended, weights = block.attn(
            normed,
            normed,
            normed,
            need_weights=True,
            average_attn_weights=False,
        )
        attn_weights.append(weights)
        x = x + block.dropout(attended)
        x = x + block.dropout(block.ff(block.norm2(x)))
    x = model.norm(x)
    return model.head(x.mean(dim=1)).squeeze(-1), attn_weights


def saliency_motif(
    model: nn.Module,
    sequences: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device],
    batch_size: int = 64,
) -> np.ndarray:
    """Accumulate a [L, 4] motif matrix from gradient×input saliency.

    For each batch: re-embed the long [B, L] token windows through a leaf
    copy of the model's token embedding table, replay the rest of the model
    to logits, backprop the summed logit, and take the absolute
    gradient×input as per-position importance. Each window's importance is
    scattered into the row of its observed base; N positions (token 4) are
    skipped. The returned matrix is the raw, unnormalized accumulation over
    all windows. Deterministic and batch-size independent (the summed logit
    decomposes per window).
    """
    device = torch.device(device)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences)
    sequences = sequences.long()
    if sequences.dim() != 2:
        raise ValueError(
            f"sequences must be [windows, length], got shape {tuple(sequences.shape)}"
        )
    n_windows, length = sequences.shape
    if n_windows == 0:
        raise ValueError("sequences is empty; nothing to score")
    if int(sequences.min()) < 0 or int(sequences.max()) > 4:
        raise ValueError("sequences must contain tokens 0-4 (A/C/G/T/N encoding)")

    embedding = _token_embedding(model)
    matrix = np.zeros((length, N_TOKENS), dtype=np.float64)
    was_training = model.training
    model.eval()
    try:
        for start in range(0, n_windows, batch_size):
            batch = sequences[start : start + batch_size].to(device)
            hidden = embedding(batch)  # [batch, length, d_model]
            logits = _forward_from_embeddings(model, hidden)
            # Gradient w.r.t. the gathered embeddings is per-position, so the
            # result does not depend on how windows are grouped into batches.
            (grad,) = torch.autograd.grad(logits.sum(), hidden)
            saliency = (grad * hidden.detach()).abs()  # [batch, length, d_model]
            importance = saliency.sum(dim=-1)  # [batch, length]
            onehot = F.one_hot(batch.clamp(max=N_TOKENS - 1), num_classes=N_TOKENS)
            onehot = onehot.to(torch.float64) * (batch < N_TOKENS).unsqueeze(-1)
            matrix += (importance.to(torch.float64).unsqueeze(-1) * onehot).sum(dim=0).cpu().numpy()
    finally:
        if was_training:
            model.train()
    return matrix


def _joint_rollout(attn_weights: list[torch.Tensor]) -> torch.Tensor:
    """Compose captured per-layer attention into the joint rollout matrix.

    Each entry is the raw per-head output of nn.MultiheadAttention with
    average_attn_weights=False, shape [batch, heads, L, L]. Per layer the
    heads are averaged to A_l [batch, L, L] and the residual connection is
    folded in as Â_l = 0.5 * (A_l + I), keeping rows stochastic. The joint
    rollout R = Â_1 @ Â_2 @ ... @ Â_n is built left to right with batched
    matmul. Returns [batch, L, L].
    """
    rollout: Optional[torch.Tensor] = None
    for weights in attn_weights:
        layer_attn = weights.mean(dim=1)  # [batch, L, L]
        _, length, _ = layer_attn.shape
        identity = torch.eye(
            length, device=layer_attn.device, dtype=layer_attn.dtype
        )[None, :, :]
        layer_attn = 0.5 * (layer_attn + identity)
        rollout = layer_attn if rollout is None else torch.matmul(rollout, layer_attn)
    if rollout is None:
        raise ValueError("attn_weights is empty; model has no attention layers")
    return rollout


def _rollout_importance(attn_weights: list[torch.Tensor]) -> torch.Tensor:
    """Per-position importance profile from captured attention weights.

    The joint rollout R [batch, L, L] holds attention from query position
    (dim 1) to key position (dim 2); the importance of a key position is the
    attention it receives, R.sum(dim=1), giving a [batch, L] profile that is
    L2-normalized per window. Non-negative and finite by construction.
    """
    rollout = _joint_rollout(attn_weights)
    importance = rollout.sum(dim=1)  # [batch, L] attention received per key
    norms = importance.norm(dim=-1, keepdim=True)
    return importance / norms.clamp(min=1e-12)


def rollout_motif(
    model: nn.Module,
    sequences: Union[np.ndarray, torch.Tensor],
    device: Union[str, torch.device],
    batch_size: int = 64,
) -> np.ndarray:
    """Accumulate a [L, 4] motif matrix from attention-rollout importance.

    For each batch: replay the model with per-head attention capture
    (_forward_with_attention, no gradients), combine the captured weights
    into an L2-normalized per-window importance profile (_rollout_importance),
    and scatter each window's importance into the row of its observed base.
    N positions (token 4) are skipped, exactly as in saliency_motif. The
    returned matrix is the raw, unnormalized accumulation over all windows.
    Deterministic and batch-size independent (attention is computed per
    window).
    """
    device = torch.device(device)
    if batch_size < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    if isinstance(sequences, np.ndarray):
        sequences = torch.from_numpy(sequences)
    sequences = sequences.long()
    if sequences.dim() != 2:
        raise ValueError(
            f"sequences must be [windows, length], got shape {tuple(sequences.shape)}"
        )
    n_windows, length = sequences.shape
    if n_windows == 0:
        raise ValueError("sequences is empty; nothing to score")
    if int(sequences.min()) < 0 or int(sequences.max()) > 4:
        raise ValueError("sequences must contain tokens 0-4 (A/C/G/T/N encoding)")

    matrix = np.zeros((length, N_TOKENS), dtype=np.float64)
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for start in range(0, n_windows, batch_size):
                batch = sequences[start : start + batch_size].to(device)
                _, attn_weights = _forward_with_attention(model, batch)
                importance = _rollout_importance(attn_weights)  # [batch, length]
                onehot = F.one_hot(batch.clamp(max=N_TOKENS - 1), num_classes=N_TOKENS)
                onehot = onehot.to(torch.float64) * (batch < N_TOKENS).unsqueeze(-1)
                matrix += (importance.to(torch.float64).unsqueeze(-1) * onehot).sum(dim=0).cpu().numpy()
    finally:
        if was_training:
            model.train()
    return matrix


def _center_profile(matrix: np.ndarray, width_bp: int = _CENTER_PROFILE_BP) -> np.ndarray:
    """Per-position total importance, trimmed to the central width_bp."""
    length = matrix.shape[0]
    half = min(width_bp // 2, length // 2)
    return matrix[length // 2 - half : length // 2 + half].sum(axis=1)


def _write_logo(
    matrix: np.ndarray,
    png_path: Path,
    half_width: int = _LOGO_HALF_WIDTH,
    title: str = "gradient×input saliency logo (validation positives)",
) -> None:
    """Render the central ±half_width bp as a stacked PWM-style sequence logo.

    Each column's stack height is the column's total saliency mass, normalized
    so the tallest stack spans 1.0; letters within a column are stacked
    bottom-up, largest first, with heights set by the within-column base
    probabilities. Glyph sizes are derived from the data-coordinate stack
    heights via the fixed axes geometry, so the logo fills the figure
    regardless of dpi. Uses the Agg canvas directly; no display is needed.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    length = matrix.shape[0]
    center = length // 2
    lo = max(0, center - half_width)
    hi = min(length, center + half_width)
    region = matrix[lo:hi].astype(np.float64)
    totals = region.sum(axis=1)
    probs = np.divide(
        region,
        totals[:, None],
        out=np.zeros_like(region),
        where=totals[:, None] != 0,
    )
    max_mass = float(totals.max())
    stacks = totals / max_mass if max_mass > 0.0 else totals

    n_cols = region.shape[0]
    fig_width = max(8.0, n_cols * 0.16)
    fig_height = 4.2
    fig = Figure(figsize=(fig_width, fig_height))
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0.075, 0.14, 0.905, 0.75])
    ylim = 1.05
    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(0.0, ylim)
    # DejaVu Sans cap height is ~0.72 em; map stack heights to font points.
    points_per_unit = fig_height * 0.75 * 72.0 / ylim

    for col in range(n_cols):
        order = np.argsort(-probs[col], kind="stable")
        bottom = 0.0
        for base_index in order:
            share = probs[col, base_index]
            if share <= 0.0:
                continue
            height = share * stacks[col]
            fontsize = height * points_per_unit / 0.72
            if fontsize < 1.0:
                continue
            base = _BASES[base_index]
            ax.text(
                col,
                bottom + height / 2.0,
                base,
                ha="center",
                va="center",
                fontsize=fontsize,
                fontweight="bold",
                family="sans-serif",
                color=_BASE_COLORS[base],
            )
            bottom += height
    step = max(1, n_cols // 10)
    ticks = np.arange(0, n_cols, step)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(t + lo - center)) for t in ticks])
    ax.set_xlabel("position relative to window center (bp)", family="sans-serif")
    ax.set_ylabel("importance share", family="sans-serif")
    ax.set_title(title, family="sans-serif")
    fig.savefig(png_path, dpi=150)


def _positive_sequences(
    dataset: WindowDataset, split: str, n_windows: int
) -> np.ndarray:
    """First n_windows positive windows of the split, in dataset order."""
    if n_windows < 1:
        raise ValueError(f"n_windows must be >= 1, got {n_windows}")
    rows = []
    for row in dataset.iter_split(split):
        if row["label"] == 1:
            rows.append(row["sequence"])
            if len(rows) >= n_windows:
                break
    if not rows:
        raise ValueError(f"split '{split}' has no positive windows to score")
    return np.stack(rows).astype(np.int64)


def run_interpret(
    data_config_path: Union[str, Path],
    checkpoint_path: Union[str, Path],
    split: str = "validation",
    n_windows: int = 2000,
    out_prefix: Union[str, Path] = "motif",
    device: str = "auto",
    method: str = "saliency",
) -> dict[str, Any]:
    """Score positive windows and write the motif npy/json/png artifacts.

    method selects the importance source: "saliency" (gradient×input,
    default) or "rollout" (attention rollout). Artifacts are written as
    <out-prefix>_motif.* for saliency and <out-prefix>_rollout.* for rollout.

    Returns the JSON summary payload that is also written to
    <out-prefix>_<method-tag>.json.
    """
    if method not in ("saliency", "rollout"):
        raise ValueError(f"unknown method {method!r}; expected saliency|rollout")
    device = resolve_device(device)
    data_config = load_config(data_config_path)
    dataset_path = Path(data_config["paths"]["dataset"])
    model, _ = load_model(checkpoint_path, device)
    with WindowDataset(dataset_path) as dataset:
        sequences = _positive_sequences(dataset, split, n_windows)

    tag = "motif" if method == "saliency" else "rollout"
    title = (
        "gradient×input saliency logo (validation positives)"
        if method == "saliency"
        else "attention rollout logo (validation positives)"
    )
    if method == "saliency":
        matrix = saliency_motif(model, sequences, device)
    else:
        matrix = rollout_motif(model, sequences, device)
    profile = _center_profile(matrix)
    summary: dict[str, Any] = {
        "n_windows": int(sequences.shape[0]),
        "split": split,
        "center_profile": [float(value) for value in profile],
        "peak_offset_of_max_importance": int(np.argmax(profile) - profile.shape[0] // 2),
    }

    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    npy_path = out_prefix.parent / f"{out_prefix.name}_{tag}.npy"
    json_path = out_prefix.parent / f"{out_prefix.name}_{tag}.json"
    png_path = out_prefix.parent / f"{out_prefix.name}_{tag}.png"
    np.save(npy_path, matrix)
    with json_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")
    _write_logo(matrix, png_path, title=title)

    print(
        f"Scored {summary['n_windows']} positive {split} windows on {device} "
        f"({method}); "
        f"peak importance at offset {summary['peak_offset_of_max_importance']} bp "
        f"from window center",
        flush=True,
    )
    print(f"Wrote {npy_path}", flush=True)
    print(f"Wrote {json_path}", flush=True)
    print(f"Wrote {png_path}", flush=True)
    return summary


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Extract an importance motif from a trained window classifier"
    )
    parser.add_argument("--data-config", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="train.py <slug>_model.pt checkpoint")
    parser.add_argument("--split", default="validation", choices=["validation", "test"])
    parser.add_argument(
        "--n-windows",
        type=int,
        default=2000,
        help="positive windows to score, in dataset order, capped at available",
    )
    parser.add_argument(
        "--method",
        default="saliency",
        choices=["saliency", "rollout"],
        help="importance source: gradient×input saliency (default) or attention rollout",
    )
    parser.add_argument("--out-prefix", required=True, type=Path)
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    args = parser.parse_args(argv)
    run_interpret(
        args.data_config,
        args.model,
        split=args.split,
        n_windows=args.n_windows,
        out_prefix=args.out_prefix,
        device=args.device,
        method=args.method,
    )


if __name__ == "__main__":
    main()
