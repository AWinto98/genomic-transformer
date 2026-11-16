"""K-mer logistic-regression baseline for window classification.

Reads the prepared windows.npz through WindowDataset, builds reverse-complement
aggregated k-mer count features, z-scores them with training statistics only,
fits an lbfgs LogisticRegression on the train split, and reports AUROC, AUPRC,
and accuracy at 0.5 on the chromosome-held-out validation and test splits.

Artifacts are written into paths.output_dir: baseline_metrics.json (metrics
plus a config/versions snapshot) and baseline_model.joblib (fitted model,
normalization vectors, and feature names for reuse).
"""

import argparse
import json
import platform
from pathlib import Path
from typing import Any, Optional, Union

import joblib
import numpy as np
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from genomic_transformer.config import load_config
from genomic_transformer.data.dataset import WindowDataset
from genomic_transformer.models.kmers import (
    apply_normalizer,
    fit_normalizer,
    fit_transform_kmers,
    transform_kmers,
)

_SPLITS = ("train", "validation", "test")
_METRICS_FILENAME = "baseline_metrics.json"
_MODEL_FILENAME = "baseline_model.joblib"


def _split_sequences(dataset: WindowDataset, split: str) -> np.ndarray:
    return np.stack([row["sequence"] for row in dataset.iter_split(split)])


def _split_labels(dataset: WindowDataset, split: str) -> np.ndarray:
    return dataset.labels[dataset.split_indices(split)]


def _evaluate(model: LogisticRegression, features: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "auroc": float(roc_auc_score(labels, probabilities)),
        "auprc": float(average_precision_score(labels, probabilities)),
        "accuracy_at_0.5": float(np.mean((probabilities >= 0.5) == labels)),
    }


def run_baseline(
    config_path: Union[str, Path],
    k: int = 6,
    chunk_size: int = 8192,
) -> dict[str, Any]:
    """Run the full baseline pipeline and persist metrics + model artifacts."""
    config_path = Path(config_path)
    config = load_config(config_path)
    dataset_path = Path(config["paths"]["dataset"])
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    with WindowDataset(dataset_path) as dataset:
        sequences = {split: _split_sequences(dataset, split) for split in _SPLITS}
        labels = {split: _split_labels(dataset, split) for split in _SPLITS}
        counts = dataset.counts()
        class_balance = {split: dataset.counts(split) for split in _SPLITS}

    x_train, code_to_feature, feature_names = fit_transform_kmers(
        sequences["train"], k=k, chunk_size=chunk_size
    )
    x_validation = transform_kmers(
        sequences["validation"], code_to_feature, k=k, chunk_size=chunk_size
    )
    x_test = transform_kmers(
        sequences["test"], code_to_feature, k=k, chunk_size=chunk_size
    )

    mean, scale = fit_normalizer(x_train)
    x_train = apply_normalizer(x_train, mean, scale)
    x_validation = apply_normalizer(x_validation, mean, scale)
    x_test = apply_normalizer(x_test, mean, scale)

    model = LogisticRegression(max_iter=1000, solver="lbfgs")
    model.fit(x_train, labels["train"])

    split_metrics = {
        "validation": _evaluate(model, x_validation, labels["validation"]),
        "test": _evaluate(model, x_test, labels["test"]),
    }

    splits_config = config.get("splits") or {}
    metrics_doc: dict[str, Any] = {
        "config": {
            "path": str(config_path),
            "dataset": str(dataset_path),
            "k": k,
            "validation_chroms": [str(c) for c in splits_config.get("validation", [])],
            "test_chroms": [str(c) for c in splits_config.get("test", [])],
        },
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "features": {
            "k": k,
            "n_features": len(feature_names),
            "n_train_windows": int(x_train.shape[0]),
        },
        "counts": counts,
        "class_balance": {
            split: {"negatives": balance[0], "positives": balance[1]}
            for split, balance in class_balance.items()
        },
        "metrics": split_metrics,
    }

    metrics_path = output_dir / _METRICS_FILENAME
    with metrics_path.open("w", encoding="utf-8") as stream:
        json.dump(metrics_doc, stream, indent=2, sort_keys=True)
        stream.write("\n")
    model_path = output_dir / _MODEL_FILENAME
    joblib.dump(
        {
            "model": model,
            "k": k,
            "feature_names": feature_names,
            "mean": mean,
            "scale": scale,
        },
        model_path,
    )

    _print_summary(metrics_doc, metrics_path, model_path)
    return metrics_doc


def _print_summary(
    metrics: dict[str, Any], metrics_path: Path, model_path: Path
) -> None:
    config = metrics["config"]
    print(
        f"k-mer logistic-regression baseline: k={config['k']}, "
        f"{metrics['features']['n_features']} features, "
        f"dataset={config['dataset']}"
    )
    print(f"{'split':<12}{'windows':>10}{'neg/pos':>12}{'AUROC':>10}{'AUPRC':>10}{'ACC@0.5':>10}")
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
        print(line)
    print(f"Wrote {metrics_path}")
    print(f"Wrote {model_path}")


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the k-mer logistic-regression baseline"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--k", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=8192)
    args = parser.parse_args(argv)
    run_baseline(args.config, k=args.k, chunk_size=args.chunk_size)


if __name__ == "__main__":
    main()
