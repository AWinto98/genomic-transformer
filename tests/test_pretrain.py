"""Synthetic-only tests for MLM pretraining and train.py --init-from.

Everything runs on CPU: MPS is not bitwise deterministic, and these tests
only need shapes, masking statistics, optimization behavior, artifact
schemas, and determinism. No real genomic data is used.
"""

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from genomic_transformer.models.pretrain import (
    MASK_RATE,
    MASK_TOKEN,
    MLMModel,
    build_corpus,
    main as pretrain_main,
    mask_batch,
    run_pretraining,
)
from genomic_transformer.models.train import _load_initial_weights, run_training
from genomic_transformer.models.transformer import (
    SmallTransformer,
    TransformerConfig,
)


def _tiny_config(max_len: int = 64) -> TransformerConfig:
    return TransformerConfig(
        d_model=32, n_layers=1, n_heads=2, d_ff=64, dropout=0.0, max_len=max_len
    )


def _write_fasta(path: Path, chroms: dict[str, str]) -> Path:
    path.write_text(
        "".join(f">{name}\n{sequence}\n" for name, sequence in chroms.items()),
        encoding="utf-8",
    )
    return path


def _write_model_config(path: Path, max_len: int = 64) -> Path:
    config = {
        "model": dataclasses.asdict(_tiny_config(max_len)),
        "training": {
            "precision": "fp32",
            "warmup_steps": 5,
            "grad_clip": 1.0,
            "weight_decay": 0.0,
            "heartbeat_steps": 4,
            "seed": 42,
        },
    }
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _planted_repeat_fasta(path: Path, window: int = 64, contigs: int = 8, repeats: int = 30) -> Path:
    """Contigs that are an identical random base pattern tiled; trivially learnable."""
    rng = np.random.default_rng(0)
    pattern = rng.integers(0, 4, size=window)
    sequence = "".join("ACGT"[token] for token in pattern) * repeats
    return _write_fasta(path, {f"chr{i}": sequence for i in range(1, contigs + 1)})


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# --- masking -----------------------------------------------------------------


def test_masking_rate_within_tolerance_on_10k_positions() -> None:
    tokens = np.zeros((10, 1000), dtype=np.uint8)
    _, targets = mask_batch(tokens, np.random.default_rng(0))

    masked = targets != -100
    rate = masked.sum() / tokens.size
    assert abs(rate - MASK_RATE) <= 0.02


def test_masking_80_10_10_split_sanity() -> None:
    tokens = np.zeros((40, 1000), dtype=np.uint8)
    inputs, targets = mask_batch(tokens, np.random.default_rng(1))
    masked = targets != -100
    n_masked = masked.sum()
    assert n_masked > 0

    to_mask_token = (inputs[masked] == MASK_TOKEN).sum() / n_masked
    unchanged = (inputs[masked] == tokens[masked]).sum() / n_masked
    randomized = 1.0 - to_mask_token - unchanged
    # Loose bounds around 80/10/10 to stay far from flaky; random-base draws
    # can coincidentally reproduce the original token, inflating "unchanged".
    assert 0.70 <= to_mask_token <= 0.88
    assert 0.05 <= unchanged <= 0.22
    assert 0.03 <= randomized <= 0.18


def test_masking_targets_and_untouched_positions() -> None:
    rng = np.random.default_rng(2)
    tokens = rng.integers(0, 5, size=(16, 256), dtype=np.uint8)
    inputs, targets = mask_batch(tokens, np.random.default_rng(3))
    masked = targets != -100

    assert targets.dtype == np.int64
    np.testing.assert_array_equal(targets[masked], tokens[masked].astype(np.int64))
    assert (targets[~masked] == -100).all()
    # Only masked positions may be modified.
    np.testing.assert_array_equal(inputs[~masked], tokens[~masked])
    assert set(np.unique(inputs)) <= set(range(5))


def test_masking_deterministic_with_seeded_rng() -> None:
    tokens = np.random.default_rng(4).integers(0, 5, size=(8, 128), dtype=np.uint8)
    first_inputs, first_targets = mask_batch(tokens, np.random.default_rng(42))
    second_inputs, second_targets = mask_batch(tokens, np.random.default_rng(42))

    np.testing.assert_array_equal(first_inputs, second_inputs)
    np.testing.assert_array_equal(first_targets, second_targets)


def test_masking_rejects_non_2d_tokens() -> None:
    with pytest.raises(ValueError, match="tokens"):
        mask_batch(np.zeros(10, dtype=np.uint8), np.random.default_rng(0))


# --- corpus ------------------------------------------------------------------


def test_corpus_window_count_per_contig_and_stride(tmp_path: Path) -> None:
    fasta = _write_fasta(
        tmp_path / "corpus.fa", {"chr1": "ACGT" * 625, "chr2": "ACGT" * 250}
    )
    # len(chr1) = 2500 -> starts 0, 500, 1000, 1500 with stride 500 -> 4 windows.
    corpus = build_corpus(fasta, ["chr1"], window=1000, stride=500)
    assert corpus.shape == (4, 1000)
    # len(chr2) = 1000 -> exactly one window.
    corpus = build_corpus(fasta, ["chr2"], window=1000, stride=1000)
    assert corpus.shape == (1, 1000)
    corpus = build_corpus(fasta, ["chr1", "chr2"], window=1000, stride=1000)
    assert corpus.shape == (3, 1000)  # chr1: 2 (0, 1000), chr2: 1; tail dropped
    assert corpus.dtype == np.uint8
    assert corpus.max() <= 4


def test_corpus_tokens_match_reference(tmp_path: Path) -> None:
    fasta = _write_fasta(tmp_path / "corpus.fa", {"chr1": "AACCGGTT" * 4})
    corpus = build_corpus(fasta, ["chr1"], window=8, stride=8)
    assert corpus.shape == (4, 8)
    expected = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.uint8)
    for window_tokens in corpus:
        np.testing.assert_array_equal(window_tokens, expected)


def test_corpus_n_filtering(tmp_path: Path) -> None:
    fasta = _write_fasta(
        tmp_path / "corpus.fa",
        {"chr1": "A" * 30 + "N" * 10 + "A" * 20 + "N" * 2 + "A" * 8},
    )
    # 7 candidate 10-bp windows; N-fractions 0,0,1.0,0,0.2,0,0. At the 0.1
    # limit the all-N window and the 0.2 window are dropped.
    corpus = build_corpus(fasta, ["chr1"], window=10, stride=10, max_n_fraction=0.1)
    assert corpus.shape == (5, 10)
    assert (corpus == MASK_TOKEN).sum() == 0


def test_corpus_limit_and_seed_determinism(tmp_path: Path) -> None:
    fasta = _write_fasta(tmp_path / "corpus.fa", {"chr1": "ACGT" * 250})
    full = build_corpus(fasta, ["chr1"], window=10, stride=10)
    assert full.shape == (100, 10)

    first = build_corpus(fasta, ["chr1"], window=10, stride=10, limit=25, seed=7)
    second = build_corpus(fasta, ["chr1"], window=10, stride=10, limit=25, seed=7)
    assert first.shape == (25, 10)
    np.testing.assert_array_equal(first, second)

    # limit >= available keeps stable genome order, no shuffle.
    unchanged = build_corpus(fasta, ["chr1"], window=10, stride=10, limit=100, seed=7)
    np.testing.assert_array_equal(unchanged, full)


def test_corpus_missing_chromosome_errors(tmp_path: Path) -> None:
    fasta = _write_fasta(tmp_path / "reference.fa", {"chr1": "ACGT" * 10})
    with pytest.raises(ValueError, match="not present"):
        build_corpus(fasta, ["chrMissing"], window=10, stride=10)


def test_corpus_chroms_default_is_all_records(tmp_path: Path) -> None:
    fasta = _write_fasta(tmp_path / "reference.fa", {"chr1": "A" * 40, "chrZ": "C" * 40})
    corpus = build_corpus(fasta, None, window=10, stride=10)
    assert corpus.shape == (8, 10)


# --- MLMModel ----------------------------------------------------------------


def test_mlm_model_forward_shape() -> None:
    config = _tiny_config()
    model = MLMModel(config)
    logits = model(torch.randint(0, 5, (2, 64)))

    assert logits.shape == (2, 64, 5)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_mlm_state_dict_overlaps_classifier_and_head_is_unmatched() -> None:
    config = _tiny_config()
    mlm = MLMModel(config)
    classifier = SmallTransformer(config)
    classifier_keys = set(classifier.state_dict().keys())

    source = mlm.state_dict()
    remapped = {
        key[len("encoder.") :]: value
        for key, value in source.items()
        if key.startswith("encoder.")
    }
    matched = {key: value for key, value in remapped.items() if key in classifier_keys}

    assert matched, "expected a nonempty overlap with the classifier state dict"
    # Encoder weights cover every classifier key except the classification head.
    assert set(matched) == classifier_keys - {"head.weight", "head.bias"}
    unmatched = set(source) - {"encoder." + key for key in matched}
    assert unmatched == {"mlm_head.weight", "mlm_head.bias"}

    missing, unexpected = classifier.load_state_dict(remapped, strict=False)
    assert set(missing) == {"head.weight", "head.bias"}
    assert not unexpected


# --- end-to-end pretraining ---------------------------------------------------


def test_pretraining_end_to_end_loss_decreases_and_artifacts(tmp_path: Path) -> None:
    fasta = _planted_repeat_fasta(tmp_path / "corpus.fa")
    model_config_path = _write_model_config(tmp_path / "model_config.yaml")
    output_dir = tmp_path / "out"

    metrics = run_pretraining(
        fasta,
        model_config_path,
        out_name="mlm",
        output_dir=output_dir,
        windows=240,
        epochs=4,
        batch_size=32,
        lr=1.0e-3,
        seed=1,
        device="cpu",
        window=64,
        stride=64,
    )

    log = _read_jsonl(output_dir / "mlm_pretrain_training_log.jsonl")
    assert [record["epoch"] for record in log] == [1, 2, 3, 4]
    losses = [record["train_loss"] for record in log]
    assert losses[-1] < losses[0]
    assert metrics["metrics"]["train_loss"] == losses[-1]
    assert metrics["metrics"]["initial_train_loss"] == losses[0]
    assert metrics["metrics"]["optimizer_steps"] == log[-1]["optimizer_steps"]
    for record in log:
        assert {"epoch", "train_loss", "lr", "epoch_seconds", "optimizer_steps"} <= set(record)

    status = json.loads((output_dir / "mlm_pretrain_status.json").read_text(encoding="utf-8"))
    assert status["epoch"] == 4
    assert status["total_epochs"] == 4
    assert status["finished"] is True
    assert status["final_metrics"]["train_loss"] == losses[-1]

    saved = torch.load(output_dir / "mlm_pretrain_model.pt", weights_only=True)
    assert set(saved) == {"state_dict", "model_config", "mlm_config"}
    assert saved["model_config"]["max_len"] == 64
    assert saved["mlm_config"]["mask_rate"] == pytest.approx(MASK_RATE)
    restored = MLMModel(saved["model_config"])
    restored.load_state_dict(saved["state_dict"])
    assert (output_dir / "mlm_pretrain_checkpoint.pt").is_file()

    assert metrics["versions"]["device"] == "cpu"
    assert metrics["corpus"]["windows"] == 240
    on_disk = json.loads((output_dir / "mlm_pretrain_metrics.json").read_text(encoding="utf-8"))
    assert on_disk == metrics


def test_pretraining_resume_continues_epochs_and_log(tmp_path: Path) -> None:
    fasta = _planted_repeat_fasta(tmp_path / "corpus.fa")
    model_config_path = _write_model_config(tmp_path / "model_config.yaml")
    output_dir = tmp_path / "out"

    run_pretraining(
        fasta, model_config_path, out_name="mlm", output_dir=output_dir,
        windows=240, epochs=2, batch_size=32, lr=1.0e-3, seed=1, device="cpu",
        window=64, stride=64,
    )
    assert [r["epoch"] for r in _read_jsonl(output_dir / "mlm_pretrain_training_log.jsonl")] == [1, 2]

    metrics = run_pretraining(
        fasta, model_config_path, out_name="mlm", output_dir=output_dir,
        windows=240, epochs=4, batch_size=32, lr=1.0e-3, seed=1, device="cpu",
        window=64, stride=64, resume=True,
    )

    log = _read_jsonl(output_dir / "mlm_pretrain_training_log.jsonl")
    assert [record["epoch"] for record in log] == [1, 2, 3, 4]
    assert metrics["metrics"]["optimizer_steps"] == log[-1]["optimizer_steps"]
    status = json.loads((output_dir / "mlm_pretrain_status.json").read_text(encoding="utf-8"))
    assert status["finished"] is True


def test_pretraining_cli_main_smoke(tmp_path: Path) -> None:
    fasta = _planted_repeat_fasta(tmp_path / "corpus.fa")
    model_config_path = _write_model_config(tmp_path / "model_config.yaml")
    output_dir = tmp_path / "out"

    pretrain_main(
        [
            "--fasta", str(fasta),
            "--config", str(model_config_path),
            "--out-name", "climlm",
            "--output-dir", str(output_dir),
            "--windows", "120",
            "--epochs", "1",
            "--batch", "32",
            "--lr", "1e-3",
            "--seed", "1",
            "--device", "cpu",
            "--window", "64",
            "--chroms", "chr1,chr2",
            "--max-n-fraction", "0.5",
        ]
    )

    metrics_path = output_dir / "climlm_pretrain_metrics.json"
    assert metrics_path.is_file()
    doc = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert doc["corpus"]["windows"] == 60  # 2 contigs x 30 windows each at window 64
    assert doc["corpus"]["chroms"] == ["chr1", "chr2"]


# --- train.py --init-from ------------------------------------------------------


def _write_synthetic_dataset(path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_windows, width = 24, 32
    labels = np.asarray(
        [1] * 6 + [0] * 6 + [1, 1, 1, 0, 0, 0] + [1, 1, 1, 0, 0, 0], dtype=np.int8
    )
    split = np.asarray([0] * 12 + [1] * 6 + [2] * 6, dtype=np.int8)
    sequences = rng.integers(0, 5, size=(n_windows, width), dtype=np.uint8)
    arrays = {
        "sequences": sequences,
        "labels": labels,
        "chrom": np.asarray(["chrFake"] * n_windows),
        "start": (np.arange(n_windows) * 100).astype(np.int64),
        "end": (np.arange(n_windows) * 100 + width).astype(np.int64),
        "gc_fraction": np.zeros(n_windows, dtype=np.float64),
        "repeat_fraction": np.zeros(n_windows, dtype=np.float64),
        "gc_bin": np.zeros(n_windows, dtype=np.int16),
        "repeat_bin": np.zeros(n_windows, dtype=np.int16),
        "peak_index": np.asarray(
            [i if i < 12 else -1 for i in range(n_windows)], dtype=np.int32
        ),
        "split": split,
    }
    np.savez_compressed(path, **arrays)
    return path


def _write_classifier_configs(tmp_path: Path) -> tuple[Path, Path]:
    dataset_path = _write_synthetic_dataset(tmp_path / "windows.npz")
    data_config = {
        "paths": {
            "dataset": str(dataset_path),
            "output_dir": str(tmp_path / "out"),
        },
        "splits": {"validation": ["chrFake17"], "test": ["chrFake8"]},
    }
    data_config_path = tmp_path / "data_config.yaml"
    data_config_path.write_text(
        yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8"
    )
    model_config = {
        "model": {
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "d_ff": 32,
            "dropout": 0.0,
            "pooling": "mean",
            "max_len": 32,
        },
        "training": {
            "epochs": 2,
            "batch_size": 4,
            "lr": 1.0e-3,
            "weight_decay": 0.0,
            "warmup_steps": 2,
            "patience": 2,
            "seed": 42,
            "grad_clip": 1.0,
        },
    }
    model_config_path = tmp_path / "model_config.yaml"
    model_config_path.write_text(yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8")
    return data_config_path, model_config_path


def _write_pretrain_checkpoint(path: Path, config: TransformerConfig) -> Path:
    torch.manual_seed(123)
    mlm = MLMModel(config)
    torch.save(
        {
            "state_dict": mlm.state_dict(),
            "model_config": dataclasses.asdict(config),
            "mlm_config": {"optimizer_steps": 10},
        },
        path,
    )
    return path


def test_load_initial_weights_copies_encoder_and_reports(tmp_path: Path) -> None:
    config = TransformerConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0, max_len=32)
    checkpoint_path = _write_pretrain_checkpoint(tmp_path / "mlm_model.pt", config)
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    classifier = SmallTransformer(config)
    reference_head = classifier.head.weight.detach().clone()

    _load_initial_weights(classifier, checkpoint_path, start=time.monotonic())

    torch.testing.assert_close(
        classifier.blocks[0].attn.in_proj_weight,
        saved["state_dict"]["encoder.blocks.0.attn.in_proj_weight"],
    )
    torch.testing.assert_close(
        classifier.norm.weight, saved["state_dict"]["encoder.norm.weight"]
    )
    # The classification head is not part of the pretrain checkpoint.
    torch.testing.assert_close(classifier.head.weight, reference_head)


def test_init_from_pretrain_checkpoint_records_and_runs(tmp_path: Path, capsys) -> None:
    data_config_path, model_config_path = _write_classifier_configs(tmp_path)
    config = TransformerConfig(d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0, max_len=32)
    init_path = _write_pretrain_checkpoint(tmp_path / "mlm_model.pt", config)

    metrics = run_training(
        data_config_path,
        model_config_path,
        device="cpu",
        out_name="ft",
        init_from=init_path,
    )

    assert metrics["config"]["init_from"] == str(init_path)
    captured = capsys.readouterr().out
    assert "init-from" in captured
    assert "matched=" in captured
    assert "mlm_head.weight" in captured  # reported as ignored, not loaded


def test_init_from_classifier_checkpoint_roundtrip(tmp_path: Path, capsys) -> None:
    data_config_path, model_config_path = _write_classifier_configs(tmp_path)

    run_training(data_config_path, model_config_path, device="cpu", out_name="t1")
    init_path = tmp_path / "out" / "t1_model.pt"
    assert init_path.is_file()

    second = run_training(
        data_config_path, model_config_path, device="cpu", out_name="t2", init_from=init_path
    )

    captured = capsys.readouterr().out
    assert "matched=" in captured
    # A classifier checkpoint matches every key, including the head.
    assert "mlm_head" not in captured
    assert second["config"]["init_from"] == str(init_path)


def test_init_from_garbage_path_errors(tmp_path: Path) -> None:
    data_config_path, model_config_path = _write_classifier_configs(tmp_path)
    with pytest.raises((OSError, ValueError)):
        run_training(
            data_config_path,
            model_config_path,
            device="cpu",
            out_name="ft",
            init_from=tmp_path / "does_not_exist.pt",
        )


def test_init_from_zero_matching_keys_errors(tmp_path: Path) -> None:
    data_config_path, model_config_path = _write_classifier_configs(tmp_path)
    bogus = tmp_path / "bogus.pt"
    torch.save({"state_dict": {"unrelated.weight": torch.zeros(1)}}, bogus)

    with pytest.raises(ValueError, match="no keys"):
        run_training(
            data_config_path,
            model_config_path,
            device="cpu",
            out_name="ft",
            init_from=bogus,
        )
