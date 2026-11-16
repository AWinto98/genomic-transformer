"""Tests for interpret.py. All fixtures are synthetic: a tiny random
SmallTransformer saved in train.py's model payload format, plus a tiny
random windows.npz that satisfies the WindowDataset schema. No real data
and no real checkpoints are touched here.
"""

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from genomic_transformer import interpret
from genomic_transformer.models.transformer import SmallTransformer, TransformerConfig

WINDOW_LENGTH = 96
MOTIF_START = 44
MOTIF_STOP = 52
MOTIF = np.asarray([0, 1, 2, 3, 3, 2, 1, 0])  # embedded at the window center

TINY_CONFIG = TransformerConfig(
    d_model=16,
    n_layers=1,
    n_heads=2,
    d_ff=32,
    dropout=0.0,
    max_len=128,
)

# Deeper sibling so parity/rollout tests exercise cross-layer composition.
DEEP_CONFIG = TransformerConfig(
    d_model=16,
    n_layers=3,
    n_heads=4,
    d_ff=32,
    dropout=0.0,
    max_len=128,
)


def write_synthetic_dataset(path: Path) -> Path:
    """Random-token windows.npz in the prepare_ctcf schema (24 windows)."""
    n = 24
    rng = np.random.default_rng(7)
    arrays = {
        "sequences": rng.integers(0, 5, size=(n, WINDOW_LENGTH), dtype=np.uint8),
        "labels": np.asarray(
            [1] * 4 + [0] * 4 + [1] * 3 + [0] * 5 + [1] * 2 + [0] * 6,
            dtype=np.int8,
        ),
        "chrom": np.asarray(["chrFake1"] * 8 + ["chrFake2"] * 8 + ["chrFake3"] * 8),
        "start": (np.arange(n) * 1000).astype(np.int64),
        "end": (np.arange(n) * 1000 + WINDOW_LENGTH).astype(np.int64),
        "gc_fraction": (np.arange(n) / n).astype(np.float64),
        "repeat_fraction": (1.0 - np.arange(n) / n).astype(np.float64),
        "gc_bin": (np.arange(n) % 3).astype(np.int16),
        "repeat_bin": ((np.arange(n) + 1) % 3).astype(np.int16),
        "peak_index": np.asarray(
            [0, 1, 2, 3, -1, -1, -1, -1, 4, 5, 6, -1, -1, -1, -1, -1, 7, 8]
            + [-1] * 6,
            dtype=np.int32,
        ),
        "split": np.asarray([0] * 8 + [1] * 8 + [2] * 8, dtype=np.int8),
    }
    np.savez_compressed(path, **arrays)
    return path


def make_random_windows(n: int, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 4, size=(n, WINDOW_LENGTH))


def plant_motif(background: np.ndarray) -> np.ndarray:
    windows = background.copy()
    windows[:, MOTIF_START:MOTIF_STOP] = MOTIF[None, :]
    return windows


def save_checkpoint(model: SmallTransformer, path: Path) -> Path:
    """Mirror train.py's <slug>_model.pt payload exactly."""
    torch.save(
        {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "model_config": dataclasses.asdict(model.config),
        },
        path,
    )
    return path


@pytest.fixture()
def checkpoint_path(tmp_path: Path) -> Path:
    torch.manual_seed(0)
    model = SmallTransformer(TINY_CONFIG)
    return save_checkpoint(model, tmp_path / "tiny_model.pt")


@pytest.fixture()
def dataset_path(tmp_path: Path) -> Path:
    return write_synthetic_dataset(tmp_path / "windows.npz")


def train_motif_model(positives: np.ndarray, negatives: np.ndarray) -> SmallTransformer:
    """Briefly train a tiny model on a trivial planted-motif task so its
    saliency keys on the motif positions (deterministic on CPU)."""
    torch.manual_seed(0)
    model = SmallTransformer(TINY_CONFIG)
    sequences = torch.from_numpy(np.concatenate([positives, negatives])).long()
    labels = torch.from_numpy(
        np.asarray([1] * len(positives) + [0] * len(negatives))
    ).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.0)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    model.train()
    for _ in range(300):
        optimizer.zero_grad()
        loss = loss_fn(model(sequences), labels)
        loss.backward()
        optimizer.step()
    model.eval()
    return model


def test_load_model_builds_configured_model_in_eval_mode(
    checkpoint_path: Path,
) -> None:
    model, model_config = interpret.load_model(checkpoint_path, torch.device("cpu"))
    assert isinstance(model, SmallTransformer)
    assert isinstance(model_config, TransformerConfig)
    assert model_config.d_model == 16
    assert model_config.n_layers == 1
    assert not model.training
    logits = model(torch.zeros(2, 8, dtype=torch.long))
    assert logits.shape == (2,)


def test_load_model_rejects_foreign_payload(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign.pt"
    torch.save({"weights": {}}, foreign)
    with pytest.raises(ValueError, match="state_dict"):
        interpret.load_model(foreign, torch.device("cpu"))


def test_saliency_motif_shape_finite_non_negative(checkpoint_path: Path) -> None:
    model, _ = interpret.load_model(checkpoint_path, "cpu")
    sequences = torch.from_numpy(make_random_windows(10))
    matrix = interpret.saliency_motif(model, sequences, "cpu", batch_size=4)
    assert matrix.shape == (WINDOW_LENGTH, 4)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0).all()
    assert matrix.sum() > 0


def test_saliency_motif_is_batch_size_independent(checkpoint_path: Path) -> None:
    model, _ = interpret.load_model(checkpoint_path, "cpu")
    sequences = make_random_windows(9)
    matrix_4 = interpret.saliency_motif(model, sequences, "cpu", batch_size=4)
    matrix_64 = interpret.saliency_motif(model, sequences, "cpu", batch_size=64)
    np.testing.assert_allclose(matrix_4, matrix_64, rtol=1e-6, atol=1e-10)


def test_saliency_motif_skips_n_positions(checkpoint_path: Path) -> None:
    model, _ = interpret.load_model(checkpoint_path, "cpu")
    sequences = make_random_windows(8)
    sequences[:, 5] = 4  # N tokens
    matrix = interpret.saliency_motif(model, torch.from_numpy(sequences), "cpu")
    assert matrix[5].sum() == 0.0
    assert matrix[6].sum() > 0.0


def test_saliency_motif_accumulates_planted_motif_at_center() -> None:
    positives = plant_motif(make_random_windows(64, seed=11))
    negatives = make_random_windows(64, seed=12)
    model = train_motif_model(positives, negatives)

    # Sanity: the briefly trained model separates the synthetic classes.
    with torch.no_grad():
        sequences = torch.from_numpy(np.concatenate([positives, negatives])).long()
        probabilities = torch.sigmoid(model(sequences)).numpy()
    assert probabilities[:64].mean() > 0.9
    assert probabilities[64:].mean() < 0.1

    matrix = interpret.saliency_motif(
        model, torch.from_numpy(positives), torch.device("cpu")
    )
    assert matrix.shape == (WINDOW_LENGTH, 4)
    totals = matrix.sum(axis=1)
    # Per-position mean over the 16 bp around the motif beats the flanks,
    # and the single most important position sits inside the planted motif.
    central_mean = totals[MOTIF_START - 4 : MOTIF_STOP + 4].mean()
    flank_mean = np.concatenate([totals[: MOTIF_START - 4], totals[MOTIF_STOP + 4 :]]).mean()
    assert central_mean > 2 * flank_mean
    assert int(np.argmax(totals)) in range(MOTIF_START, MOTIF_STOP)


def test_cli_end_to_end_writes_three_artifacts(
    tmp_path: Path, dataset_path: Path, checkpoint_path: Path
) -> None:
    data_config = tmp_path / "data.yaml"
    data_config.write_text(f'paths:\n  dataset: "{dataset_path}"\n', encoding="utf-8")
    out_prefix = tmp_path / "out" / "tiny"

    interpret.main(
        [
            "--data-config",
            str(data_config),
            "--model",
            str(checkpoint_path),
            "--split",
            "validation",
            "--n-windows",
            "2",  # validation has 3 positives; exercises the cap logic
            "--out-prefix",
            str(out_prefix),
            "--device",
            "cpu",
        ]
    )

    npy_path = tmp_path / "out" / "tiny_motif.npy"
    json_path = tmp_path / "out" / "tiny_motif.json"
    png_path = tmp_path / "out" / "tiny_motif.png"
    assert npy_path.exists()
    assert json_path.exists()
    assert png_path.exists()

    matrix = np.load(npy_path)
    assert matrix.shape == (WINDOW_LENGTH, 4)

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["n_windows"] == 2
    assert summary["split"] == "validation"
    assert len(summary["center_profile"]) == WINDOW_LENGTH  # shorter than 200 bp
    assert isinstance(summary["peak_offset_of_max_importance"], int)
    assert abs(summary["peak_offset_of_max_importance"]) <= WINDOW_LENGTH // 2

    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_interpret_rejects_split_without_positives(
    tmp_path: Path, dataset_path: Path, checkpoint_path: Path
) -> None:
    # Rewrite with an all-negative test split.
    with np.load(dataset_path) as data:
        arrays = {name: data[name] for name in data.files}
    arrays["labels"] = np.where(np.asarray(arrays["split"]) == 2, 0, np.asarray(arrays["labels"]))
    np.savez_compressed(dataset_path, **arrays)
    data_config = tmp_path / "data.yaml"
    data_config.write_text(f'paths:\n  dataset: "{dataset_path}"\n', encoding="utf-8")

    with pytest.raises(ValueError, match="no positive windows"):
        interpret.run_interpret(
            data_config,
            checkpoint_path,
            split="test",
            n_windows=5,
            out_prefix=tmp_path / "unused",
            device="cpu",
        )


def test_capture_forward_matches_plain_forward() -> None:
    """Manual-with-capture replay must reproduce model(input_ids) on CPU."""
    torch.manual_seed(0)
    model = SmallTransformer(DEEP_CONFIG)
    model.eval()
    sequences = torch.from_numpy(make_random_windows(6))
    with torch.no_grad():
        logits_plain = model(sequences)
        logits_capture, attn_weights = interpret._forward_with_attention(model, sequences)
    assert logits_plain.shape == (6,)
    assert logits_capture.shape == (6,)
    torch.testing.assert_close(logits_capture, logits_plain, rtol=1e-4, atol=1e-4)
    assert len(attn_weights) == DEEP_CONFIG.n_layers
    batch, length = sequences.shape
    for weights in attn_weights:
        assert weights.shape == (batch, DEEP_CONFIG.n_heads, length, length)
        assert torch.isfinite(weights).all()
        assert (weights >= 0).all()
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(batch, DEEP_CONFIG.n_heads, length),
            rtol=1e-5,
            atol=1e-6,
        )


def test_rollout_joint_and_importance_math() -> None:
    """Handcrafted layers: identity attention then uniform attention."""
    batch, length, heads = 2, 6, 3
    identity = torch.eye(length)
    uniform = torch.full((length, length), 1.0 / length)
    weights = [
        identity.expand(batch, heads, length, length).clone(),
        uniform.expand(batch, heads, length, length).clone(),
    ]

    rollout = interpret._joint_rollout(weights)
    assert rollout.shape == (batch, length, length)
    assert torch.isfinite(rollout).all()
    assert (rollout >= 0).all()
    # Â_1 = 0.5*(I + I) = I, so R = Â_2 = 0.5*(uniform + I): rows and
    # columns both sum to 1.
    torch.testing.assert_close(
        rollout.sum(dim=-1), torch.ones(batch, length), rtol=1e-6, atol=1e-8
    )
    expected = 0.5 * (uniform + identity)
    torch.testing.assert_close(
        rollout, expected.expand(batch, length, length), rtol=1e-6, atol=1e-8
    )

    importance = interpret._rollout_importance(weights)
    assert importance.shape == (batch, length)
    assert torch.isfinite(importance).all()
    assert (importance >= 0).all()
    # Every key receives the same total attention (column sums of R are 1),
    # so the L2-normalized received-attention profile is uniform 1/sqrt(L).
    torch.testing.assert_close(
        importance,
        torch.full((batch, length), 1.0 / math.sqrt(length)),
        rtol=1e-6,
        atol=1e-8,
    )
    torch.testing.assert_close(
        importance.norm(dim=-1), torch.ones(batch), rtol=1e-6, atol=1e-8
    )

    with pytest.raises(ValueError, match="no attention layers"):
        interpret._joint_rollout([])


def test_rollout_from_captured_attention_on_tiny_model() -> None:
    """R and importance profile from a real tiny model are well formed."""
    torch.manual_seed(0)
    model = SmallTransformer(DEEP_CONFIG)
    model.eval()
    sequences = torch.from_numpy(make_random_windows(5))
    with torch.no_grad():
        _, attn_weights = interpret._forward_with_attention(model, sequences)
    batch, length = sequences.shape

    rollout = interpret._joint_rollout(attn_weights)
    assert rollout.shape == (batch, length, length)
    assert (rollout >= 0).all()
    # Product of row-stochastic Â_l matrices stays row-stochastic.
    torch.testing.assert_close(
        rollout.sum(dim=-1), torch.ones(batch, length), rtol=1e-5, atol=1e-6
    )
    # Composition genuinely mixes layers: R differs from the first layer alone.
    first_layer = 0.5 * (attn_weights[0].mean(dim=1) + torch.eye(length))
    assert not torch.allclose(rollout, first_layer, rtol=1e-4, atol=1e-6)

    importance = interpret._rollout_importance(attn_weights)
    assert importance.shape == (batch, length)
    assert torch.isfinite(importance).all()
    assert (importance >= 0).all()
    torch.testing.assert_close(
        importance.norm(dim=-1), torch.ones(batch), rtol=1e-6, atol=1e-8
    )


def test_rollout_motif_shape_finite_non_negative(checkpoint_path: Path) -> None:
    model, _ = interpret.load_model(checkpoint_path, "cpu")
    sequences = make_random_windows(10)
    matrix = interpret.rollout_motif(model, sequences, "cpu", batch_size=4)
    assert matrix.shape == (WINDOW_LENGTH, 4)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0).all()
    assert matrix.sum() > 0


def test_rollout_motif_is_deterministic(checkpoint_path: Path) -> None:
    model, _ = interpret.load_model(checkpoint_path, "cpu")
    sequences = make_random_windows(9)
    matrix_a = interpret.rollout_motif(model, sequences, "cpu", batch_size=4)
    matrix_b = interpret.rollout_motif(model, sequences, "cpu", batch_size=4)
    np.testing.assert_array_equal(matrix_a, matrix_b)


def test_rollout_motif_skips_n_positions(checkpoint_path: Path) -> None:
    model, _ = interpret.load_model(checkpoint_path, "cpu")
    sequences = make_random_windows(8)
    sequences[:, 5] = 4  # N tokens
    matrix = interpret.rollout_motif(model, torch.from_numpy(sequences), "cpu")
    assert matrix.shape == (WINDOW_LENGTH, 4)
    assert matrix[5].sum() == 0.0
    assert matrix[6].sum() > 0.0


def test_cli_rollout_writes_rollout_artifacts(
    tmp_path: Path, dataset_path: Path, checkpoint_path: Path
) -> None:
    data_config = tmp_path / "data.yaml"
    data_config.write_text(f'paths:\n  dataset: "{dataset_path}"\n', encoding="utf-8")
    out_prefix = tmp_path / "out" / "tiny"

    interpret.main(
        [
            "--data-config",
            str(data_config),
            "--model",
            str(checkpoint_path),
            "--split",
            "validation",
            "--n-windows",
            "2",  # validation has 3 positives; exercises the cap logic
            "--method",
            "rollout",
            "--out-prefix",
            str(out_prefix),
            "--device",
            "cpu",
        ]
    )

    npy_path = tmp_path / "out" / "tiny_rollout.npy"
    json_path = tmp_path / "out" / "tiny_rollout.json"
    png_path = tmp_path / "out" / "tiny_rollout.png"
    assert npy_path.exists()
    assert json_path.exists()
    assert png_path.exists()
    # Rollout must not also write the saliency artifacts.
    assert not (tmp_path / "out" / "tiny_motif.npy").exists()
    assert not (tmp_path / "out" / "tiny_motif.json").exists()
    assert not (tmp_path / "out" / "tiny_motif.png").exists()

    matrix = np.load(npy_path)
    assert matrix.shape == (WINDOW_LENGTH, 4)
    assert np.isfinite(matrix).all()
    assert (matrix >= 0).all()

    summary = json.loads(json_path.read_text(encoding="utf-8"))
    assert summary["n_windows"] == 2
    assert summary["split"] == "validation"
    assert len(summary["center_profile"]) == WINDOW_LENGTH  # shorter than 200 bp
    assert isinstance(summary["peak_offset_of_max_importance"], int)
    assert abs(summary["peak_offset_of_max_importance"]) <= WINDOW_LENGTH // 2

    assert png_path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_run_interpret_rejects_unknown_method(
    tmp_path: Path, dataset_path: Path, checkpoint_path: Path
) -> None:
    data_config = tmp_path / "data.yaml"
    data_config.write_text(f'paths:\n  dataset: "{dataset_path}"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="unknown method"):
        interpret.run_interpret(
            data_config,
            checkpoint_path,
            split="validation",
            n_windows=2,
            out_prefix=tmp_path / "unused",
            device="cpu",
            method="attention",
        )
