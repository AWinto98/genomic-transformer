"""Synthetic-only tests for the transformer model and training CLI.

Everything runs on CPU: MPS is not bitwise deterministic, and these tests
only need shapes, optimization behavior, artifact schema, and determinism.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from genomic_transformer.config import load_config
from genomic_transformer.models.train import (
    _apply_mps_watermark,
    _freeze_encoder,
    resolve_device,
    resolve_precision,
    run_training,
)
from genomic_transformer.models.transformer import (
    SmallTransformer,
    TransformerConfig,
    count_parameters,
)

_MODEL_CONFIG_PATH = Path("config/model/small_transformer.yaml")
_CLOUD_CONFIG_PATH = Path("config/model/small_transformer_cloud.yaml")


def test_forward_pass_shape_dtype_and_finite() -> None:
    config = TransformerConfig.from_dict(load_config(_MODEL_CONFIG_PATH)["model"])
    model = SmallTransformer(config)
    input_ids = torch.randint(0, 5, (3, config.max_len))

    logits = model(input_ids)

    assert logits.shape == (3,)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


def test_default_config_parameter_count_under_5m() -> None:
    model = SmallTransformer(TransformerConfig.from_dict(load_config(_MODEL_CONFIG_PATH)["model"]))

    assert count_parameters(model) < 5_000_000


def test_builds_from_plain_dict_config() -> None:
    config = {
        "d_model": 16,
        "n_layers": 1,
        "n_heads": 2,
        "d_ff": 32,
        "dropout": 0.0,
        "pooling": "mean",
        "max_len": 64,
    }
    model = SmallTransformer(config)
    logits = model(torch.randint(0, 5, (2, 64)))
    assert logits.shape == (2,)


def test_tiny_overfit_smoke_loss_decreases() -> None:
    torch.manual_seed(0)
    config = TransformerConfig(
        d_model=64, n_layers=2, n_heads=4, d_ff=128, dropout=0.0, max_len=64
    )
    model = SmallTransformer(config)
    sequences = torch.randint(0, 5, (128, 64))
    labels = torch.randint(0, 2, (128,)).float()
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-4)

    initial_loss = loss_fn(model(sequences), labels).item()
    for _ in range(60):
        optimizer.zero_grad()
        loss = loss_fn(model(sequences), labels)
        loss.backward()
        optimizer.step()
    final_loss = loss_fn(model(sequences), labels).item()

    assert final_loss < initial_loss


def test_resolve_device_auto_returns_valid_device() -> None:
    device = resolve_device("auto")
    assert isinstance(device, torch.device)
    assert device.type in ("cpu", "mps")
    assert resolve_device("cpu").type == "cpu"


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


def _write_model_config(path: Path) -> Path:
    config = {
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
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def test_training_cli_end_to_end_on_synthetic_npz(tmp_path: Path) -> None:
    dataset_path = _write_synthetic_dataset(tmp_path / "windows.npz")
    output_dir = tmp_path / "out"
    data_config = {
        "paths": {
            "dataset": str(dataset_path),
            "output_dir": str(output_dir),
        },
        "splits": {"validation": ["chrFake17"], "test": ["chrFake8"]},
    }
    data_config_path = tmp_path / "data_config.yaml"
    data_config_path.write_text(
        yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8"
    )
    model_config_path = _write_model_config(tmp_path / "model_config.yaml")

    metrics = run_training(
        data_config_path, model_config_path, device="cpu", out_name="transformer"
    )

    metrics_path = output_dir / "transformer_metrics.json"
    model_path = output_dir / "transformer_model.pt"
    assert metrics_path.is_file()
    assert model_path.is_file()

    assert set(metrics) == {"config", "versions", "counts", "class_balance", "metrics"}
    assert metrics["counts"] == {"train": 12, "validation": 6, "test": 6}
    assert metrics["class_balance"]["validation"] == {
        "negatives": 3,
        "positives": 3,
    }
    assert metrics["versions"]["device"] == "cpu"
    assert metrics["versions"]["mps_available"] == torch.backends.mps.is_available()
    assert "torch" in metrics["versions"]
    assert metrics["config"]["model"]["max_len"] == 32
    for split in ("validation", "test"):
        assert set(metrics["metrics"][split]) == {"auroc", "auprc", "accuracy_at_0.5"}
        for value in metrics["metrics"][split].values():
            assert 0.0 <= value <= 1.0

    saved = torch.load(model_path, weights_only=True)
    assert set(saved) == {"state_dict", "model_config"}
    assert saved["model_config"]["max_len"] == 32
    restored = SmallTransformer(saved["model_config"])
    restored.load_state_dict(saved["state_dict"])

    on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert on_disk == metrics

    # Determinism: a second CPU run reproduces identical metrics.
    second = run_training(
        data_config_path, model_config_path, device="cpu", out_name="transformer"
    )
    assert second == metrics


def _training_setup(
    tmp_path: Path, out_name: str, training_overrides: "dict | None" = None
):
    """Synthetic dataset + configs; returns (data_config, model_config, output_dir)."""
    dataset_path = _write_synthetic_dataset(tmp_path / "windows.npz")
    output_dir = tmp_path / "out"
    data_config = {
        "paths": {"dataset": str(dataset_path), "output_dir": str(output_dir)},
        "splits": {"validation": ["chrFake17"], "test": ["chrFake8"]},
    }
    data_config_path = tmp_path / "data_config.yaml"
    data_config_path.write_text(
        yaml.safe_dump(data_config, sort_keys=False), encoding="utf-8"
    )
    model_config_path = _write_model_config(tmp_path / "model_config.yaml")
    if training_overrides:
        model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
        model_config["training"].update(training_overrides)
        model_config_path.write_text(
            yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8"
        )
    return data_config_path, model_config_path, output_dir


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_grad_accumulation_optimizer_step_count(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    # 12 train windows, batch 4, accum 2 -> 3 micro-batches -> 2 optimizer steps/epoch.
    step_calls: list[int] = []
    original_step = torch.optim.AdamW.step

    def _counting_step(self, closure=None):
        step_calls.append(1)
        return original_step(self, closure)

    monkeypatch.setattr(torch.optim.AdamW, "step", _counting_step)
    data_config_path, model_config_path, output_dir = _training_setup(
        tmp_path,
        "accum",
        {"grad_accum_steps": 2, "heartbeat_steps": 1},
    )

    metrics = run_training(
        data_config_path, model_config_path, device="cpu", out_name="accum"
    )

    assert len(step_calls) == 4  # 2 optimizer steps/epoch x 2 epochs
    captured = capsys.readouterr().out
    assert "epoch 1 step 1/2" in captured
    assert "epoch 1 step 2/2" in captured
    assert "epoch 2 step 1/2" in captured
    log = _read_jsonl(output_dir / "accum_training_log.jsonl")
    assert len(log) == 2

    second = run_training(
        data_config_path, model_config_path, device="cpu", out_name="accum"
    )
    assert second == metrics


def test_observability_files_and_flushed_output(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    data_config_path, model_config_path, output_dir = _training_setup(
        tmp_path,
        "obs",
        {"grad_accum_steps": 2, "heartbeat_steps": 1, "mps_high_watermark_ratio": 0.5},
    )

    run_training(data_config_path, model_config_path, device="cpu", out_name="obs")

    captured = capsys.readouterr().out
    for epoch in (1, 2):
        assert re.search(
            rf"\[\d{{2}}:\d{{2}}:\d{{2}}\] epoch {epoch}/2 "
            r"train_loss=\S+ val_auroc=\S+ best=\S+ epoch_seconds=\S+",
            captured,
        )
    assert re.search(r"\[\d{2}:\d{2}:\d{2}\] epoch 1 step 1/2 loss=\S+", captured)

    log_path = output_dir / "obs_training_log.jsonl"
    log = _read_jsonl(log_path)
    assert len(log) == 2
    required = {
        "epoch",
        "train_loss",
        "val_auroc",
        "val_auprc",
        "val_accuracy",
        "lr",
        "epoch_seconds",
        "best_val_auroc",
        "epochs_since_improvement",
    }
    for record in log:
        assert required <= set(record)

    status = json.loads((output_dir / "obs_status.json").read_text(encoding="utf-8"))
    assert status["epoch"] == 2
    assert status["total_epochs"] == 2
    assert status["finished"] is True
    assert set(status["final_metrics"]) == {"validation", "test"}
    # device is cpu -> the MPS watermark must not be applied, and must not crash
    assert "PYTORCH_MPS_HIGH_WATERMARK_RATIO" not in os.environ
    assert (output_dir / "obs_checkpoint.pt").is_file()


def test_checkpoint_resume_appends_log_and_carries_best(
    tmp_path: Path, capsys
) -> None:
    data_config_path, model_config_path, output_dir = _training_setup(
        tmp_path,
        "resume",
        {"grad_accum_steps": 2, "heartbeat_steps": 1, "patience": 10},
    )

    first = run_training(
        data_config_path, model_config_path, device="cpu", out_name="resume"
    )
    first_best = first["metrics"]["validation"]["auroc"]
    assert (output_dir / "resume_checkpoint.pt").is_file()
    assert len(_read_jsonl(output_dir / "resume_training_log.jsonl")) == 2

    second = run_training(
        data_config_path,
        model_config_path,
        device="cpu",
        epochs=4,
        out_name="resume",
        resume=True,
    )
    assert "resumed from" in capsys.readouterr().out

    log = _read_jsonl(output_dir / "resume_training_log.jsonl")
    assert [record["epoch"] for record in log] == [1, 2, 3, 4]
    # Best-so-far carries over as a floor; it must not reset to 0/None.
    assert second["metrics"]["validation"]["auroc"] >= first_best > 0.0
    assert second["config"]["best_epoch"] >= 1
    for record in log[2:]:
        assert record["best_val_auroc"] >= first_best

    status = json.loads((output_dir / "resume_status.json").read_text(encoding="utf-8"))
    assert status["epoch"] == 4
    assert status["finished"] is True
    assert status["best_val_auroc"] >= first_best


def test_mps_watermark_helper(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    _apply_mps_watermark(0.5)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.5"

    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.9")
    _apply_mps_watermark(0.5)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.9"

    _apply_mps_watermark(None)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.9"


def test_mps_watermark_helper_sets_low_watermark(monkeypatch) -> None:
    monkeypatch.delenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", raising=False)
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    _apply_mps_watermark(0.75)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.75"
    # LOW defaults to 1.4 in torch and must be <= HIGH, so the helper
    # derives it from the effective high value.
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.7"

    # A preset LOW env var wins, same as HIGH.
    monkeypatch.setenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.42")
    _apply_mps_watermark(0.75)
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.42"

    # LOW derives from a preset HIGH and is floored at 0.1.
    monkeypatch.delenv("PYTORCH_MPS_LOW_WATERMARK_RATIO", raising=False)
    monkeypatch.setenv("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.12")
    _apply_mps_watermark(0.9)
    assert os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] == "0.12"
    assert os.environ["PYTORCH_MPS_LOW_WATERMARK_RATIO"] == "0.1"


def test_resolve_precision() -> None:
    assert resolve_precision("bf16", torch.device("mps")) == "bf16"
    assert resolve_precision("bf16", torch.device("cuda")) == "bf16"
    assert resolve_precision("bf16", torch.device("cpu")) == "fp32"
    assert resolve_precision("fp32", torch.device("mps")) == "fp32"
    assert resolve_precision(None, torch.device("cpu")) == "fp32"
    with pytest.raises(ValueError, match="precision"):
        resolve_precision("bf17", torch.device("cpu"))


def test_precision_bf16_on_cpu_matches_fp32(tmp_path: Path) -> None:
    bf16_dir = tmp_path / "bf16"
    bf16_dir.mkdir()
    fp32_dir = tmp_path / "fp32"
    fp32_dir.mkdir()
    data_config_path, model_config_path, _ = _training_setup(
        bf16_dir, "t", {"precision": "bf16"}
    )
    fp32_data_config, fp32_model_config, _ = _training_setup(
        fp32_dir, "t", {"precision": "fp32"}
    )

    with_bf16 = run_training(data_config_path, model_config_path, device="cpu", out_name="t")
    with_fp32 = run_training(fp32_data_config, fp32_model_config, device="cpu", out_name="t")

    # CPU forces fp32, so a bf16-configured run is the identical deterministic run.
    assert with_bf16["metrics"] == with_fp32["metrics"]
    assert with_bf16["config"]["training"]["precision"] == "bf16"
    checkpoint = torch.load(
        bf16_dir / "out" / "t_checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["training_config"]["precision"] == "bf16"


def test_resume_rejects_precision_mismatch(tmp_path: Path) -> None:
    data_config_path, model_config_path, output_dir = _training_setup(
        tmp_path, "prec", {"precision": "bf16"}
    )
    run_training(data_config_path, model_config_path, device="cpu", out_name="prec")

    model_config = yaml.safe_load(model_config_path.read_text(encoding="utf-8"))
    model_config["training"]["precision"] = "fp32"
    model_config_path.write_text(
        yaml.safe_dump(model_config, sort_keys=False), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="precision"):
        run_training(
            data_config_path,
            model_config_path,
            device="cpu",
            epochs=4,
            out_name="prec",
            resume=True,
        )


def test_freeze_encoder_freezes_all_but_head() -> None:
    config = TransformerConfig(
        d_model=16, n_layers=1, n_heads=2, d_ff=32, dropout=0.0, max_len=32
    )
    model = SmallTransformer(config)
    total = sum(p.numel() for p in model.parameters())

    trainable, frozen = _freeze_encoder(model)

    assert trainable == total - frozen
    head_parameters = dict(model.named_parameters())
    assert head_parameters["head.weight"].requires_grad
    assert head_parameters["head.bias"].requires_grad
    for name, parameter in head_parameters.items():
        if not name.startswith("head."):
            assert not parameter.requires_grad

    loss = model(torch.randint(0, 5, (2, 32))).sum()
    loss.backward()
    assert model.head.weight.grad is not None
    assert model.blocks[0].attn.in_proj_weight.grad is None


def test_freeze_encoder_tiny_cpu_run(tmp_path: Path, capsys) -> None:
    data_config_path, model_config_path, output_dir = _training_setup(
        tmp_path, "freeze"
    )

    metrics = run_training(
        data_config_path,
        model_config_path,
        device="cpu",
        out_name="freeze",
        freeze_encoder=True,
    )

    assert metrics["config"]["training"]["freeze_encoder"] is True
    assert "freeze-encoder:" in capsys.readouterr().out

    on_disk = json.loads(
        (output_dir / "freeze_metrics.json").read_text(encoding="utf-8")
    )
    assert on_disk["config"]["training"]["freeze_encoder"] is True

    checkpoint = torch.load(
        output_dir / "freeze_checkpoint.pt", map_location="cpu", weights_only=True
    )
    assert checkpoint["training_config"]["freeze_encoder"] is True
    # The optimizer is built over the head only: head.weight + head.bias.
    param_groups = checkpoint["optimizer_state_dict"]["param_groups"]
    assert sum(len(group["params"]) for group in param_groups) == 2


def test_freeze_encoder_resume_and_mismatch(tmp_path: Path, capsys) -> None:
    data_config_path, model_config_path, output_dir = _training_setup(
        tmp_path, "fz", {"patience": 10}
    )

    run_training(
        data_config_path,
        model_config_path,
        device="cpu",
        out_name="fz",
        freeze_encoder=True,
    )

    second = run_training(
        data_config_path,
        model_config_path,
        device="cpu",
        epochs=4,
        out_name="fz",
        resume=True,
        freeze_encoder=True,
    )
    assert "resumed from" in capsys.readouterr().out
    assert second["config"]["training"]["freeze_encoder"] is True
    log = _read_jsonl(output_dir / "fz_training_log.jsonl")
    assert [record["epoch"] for record in log] == [1, 2, 3, 4]

    with pytest.raises(ValueError, match="freeze_encoder"):
        run_training(
            data_config_path,
            model_config_path,
            device="cpu",
            epochs=4,
            out_name="fz",
            resume=True,
            freeze_encoder=False,
        )


def test_resolve_device_auto_prefers_cuda_when_mps_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto").type == "cuda"


def test_resolve_device_auto_prefers_mps_over_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto").type == "mps"


def test_resolve_device_auto_falls_back_to_cpu(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert resolve_device("auto").type == "cpu"


def test_resolve_device_explicit_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("cuda").type == "cuda"

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(ValueError, match="CUDA is not available"):
        resolve_device("cuda")


def test_resolve_device_unknown_choice_still_raises() -> None:
    with pytest.raises(ValueError, match=re.escape("expected auto|mps|cuda|cpu")):
        resolve_device("tpu")


def test_cloud_config_loads_and_parses() -> None:
    config = load_config(_CLOUD_CONFIG_PATH)

    assert config["model"] == load_config(_MODEL_CONFIG_PATH)["model"]
    training = config["training"]
    assert training["batch_size"] == 64
    assert training["grad_accum_steps"] == 1
    assert training["precision"] == "bf16"
    assert training["epochs"] == 20
    assert training["patience"] == 4
    assert training["lr"] == 3.0e-4
    assert training["warmup_steps"] == 500
    assert training["seed"] == 42
    assert training["grad_clip"] == 1.0
    # The MPS allocator cap is Mac-only; the cloud config must not set it.
    assert "mps_high_watermark_ratio" not in training
    # The model block must still build a valid model.
    TransformerConfig.from_dict(config["model"])


def test_cloud_precision_bf16_resolves_on_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    device = resolve_device("auto")

    assert device.type == "cuda"
    assert resolve_precision("bf16", device) == "bf16"
