import json
from pathlib import Path

import joblib
import numpy as np
import yaml

from genomic_transformer.models.baseline import run_baseline
from genomic_transformer.models.kmers import (
    apply_normalizer,
    canonical_kmer_map,
    fit_normalizer,
    fit_transform_kmers,
    transform_kmers,
)

_TOKENS = {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4}


def as_sequences(*sequences: str) -> np.ndarray:
    return np.asarray(
        [[_TOKENS[base] for base in sequence] for sequence in sequences],
        dtype=np.uint8,
    )


def test_k2_counts_known_values_and_reverse_complement_aggregation() -> None:
    features, code_to_feature, names = fit_transform_kmers(as_sequences("ACGT"), k=2)

    assert features.shape == (1, 10)
    assert set(names) == {
        "AA", "AC", "AG", "AT", "CA", "CC", "CG", "GA", "GC", "TA",
    }
    index = {name: i for i, name in enumerate(names)}
    row = features[0]
    # AC and its revcomp GT fold onto the same feature; CG is self-revcomp.
    assert row[index["AC"]] == 2
    assert row[index["CG"]] == 1
    assert row[index["AA"]] == 0
    assert row.sum() == 3

    # A lone revcomp maps back onto the same canonical feature as its partner.
    revcomp_only = transform_kmers(as_sequences("GT"), code_to_feature)
    assert revcomp_only[0, index["AC"]] == 1
    assert revcomp_only[0].sum() == 1


def test_k3_counts_skip_n_kmers_and_fold_revcomp() -> None:
    # Sequences share one width (N padding contributes no counted k-mers).
    features, code_to_feature, names = fit_transform_kmers(
        as_sequences("AAACNN", "AANNNN", "AACNAA", "CATNNN", "ATGNNN"), k=3
    )

    assert features.shape == (5, 32)
    index = {name: i for i, name in enumerate(names)}
    aaac, aan, mixed, cat, atg = features

    assert aaac[index["AAA"]] == 1
    assert aaac[index["AAC"]] == 1
    assert aaac.sum() == 2

    # Every k-mer of "AAN" contains N, so nothing is counted.
    assert aan.sum() == 0

    # Only "AAC" survives the N skip in AACNAA (ACN, CNA, NAA all contain N).
    assert mixed[index["AAC"]] == 1
    assert mixed.sum() == 1

    # CAT and ATG are revcomps; the canonical name is the smaller code ("ATG").
    assert names[index["ATG"]] == "ATG"
    assert cat[index["ATG"]] == 1
    assert cat.sum() == 1
    assert atg[index["ATG"]] == 1
    assert atg.sum() == 1


def test_k6_canonical_map_has_2080_features() -> None:
    code_to_feature, names = canonical_kmer_map(6)

    assert code_to_feature.shape == (4096,)
    assert code_to_feature.dtype == np.int32
    assert len(names) == 2080
    assert len(set(names)) == 2080


def test_normalizer_uses_train_statistics_without_leakage() -> None:
    train = np.asarray(
        [[7.0, 0.0], [7.0, 4.0], [7.0, 0.0], [7.0, 4.0]], dtype=np.float32
    )
    mean, scale = fit_normalizer(train)
    assert mean[0] == 7.0
    assert scale[0] == 1.0  # zero-variance train feature clamps, never NaN

    validation = np.asarray([[7.0, 4.0], [7.0, 4.0]], dtype=np.float32)
    z_scores = apply_normalizer(validation, mean, scale)
    assert np.isfinite(z_scores).all()
    assert (z_scores[:, 0] == 0.0).all()
    # Column 1: train mean/std are 2/2, so (4 - 2) / 2 = 1; validation's own
    # constant column would give 0 if its statistics leaked in.
    assert (z_scores[:, 1] == 1.0).all()


def write_synthetic_dataset(path: Path) -> Path:
    rng = np.random.default_rng(0)
    n_windows, width = 20, 16
    labels = np.asarray(
        [1] * 5 + [0] * 5 + [1, 1, 1, 0, 0] + [1, 1, 0, 0, 0], dtype=np.int8
    )
    split = np.asarray([0] * 10 + [1] * 5 + [2] * 5, dtype=np.int8)
    sequences = np.empty((n_windows, width), dtype=np.uint8)
    for i in range(n_windows):
        probabilities = (
            [0.1, 0.4, 0.4, 0.1] if labels[i] == 1 else [0.4, 0.1, 0.1, 0.4]
        )
        sequences[i] = rng.choice(4, size=width, p=probabilities).astype(np.uint8)
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
            [i if i < 10 else -1 for i in range(n_windows)], dtype=np.int32
        ),
        "split": split,
    }
    np.savez_compressed(path, **arrays)
    return path


def test_baseline_pipeline_end_to_end_on_synthetic_npz(tmp_path: Path) -> None:
    dataset_path = write_synthetic_dataset(tmp_path / "windows.npz")
    output_dir = tmp_path / "out"
    config = {
        "paths": {
            "dataset": str(dataset_path),
            "output_dir": str(output_dir),
        },
        "splits": {"validation": ["chrFake17"], "test": ["chrFake8"]},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    metrics = run_baseline(config_path, k=2)

    metrics_path = output_dir / "baseline_metrics.json"
    model_path = output_dir / "baseline_model.joblib"
    assert metrics_path.is_file()
    assert model_path.is_file()

    assert metrics["config"]["k"] == 2
    assert metrics["config"]["validation_chroms"] == ["chrFake17"]
    assert metrics["features"]["n_features"] == 10
    assert metrics["counts"] == {"train": 10, "validation": 5, "test": 5}
    assert metrics["class_balance"]["validation"] == {"negatives": 2, "positives": 3}
    assert set(metrics["versions"]) == {"python", "numpy", "scikit_learn"}
    for split in ("validation", "test"):
        assert set(metrics["metrics"][split]) == {"auroc", "auprc", "accuracy_at_0.5"}
        for value in metrics["metrics"][split].values():
            assert 0.0 <= value <= 1.0

    saved = joblib.load(model_path)
    assert saved["k"] == 2
    assert len(saved["feature_names"]) == 10
    assert saved["mean"].shape == (10,)
    assert saved["scale"].shape == (10,)
    assert hasattr(saved["model"], "predict_proba")

    on_disk = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert on_disk == metrics

    # Determinism: a second run reproduces identical metrics.
    second = run_baseline(config_path, k=2)
    assert second == metrics
