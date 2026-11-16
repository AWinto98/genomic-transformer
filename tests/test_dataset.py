from pathlib import Path
from typing import Optional

import numpy as np
import pytest

from genomic_transformer.data.dataset import WindowDataset

EXPECTED_KEYS = {
    "sequence",
    "label",
    "chrom",
    "start",
    "end",
    "gc_fraction",
    "repeat_fraction",
    "gc_bin",
    "repeat_bin",
    "peak_index",
}


def write_dataset(
    path: Path,
    drop: tuple[str, ...] = (),
    split: Optional[list[int]] = None,
) -> Path:
    n = 12
    split = [0] * 4 + [1] * 4 + [2] * 4 if split is None else split
    arrays = {
        "sequences": np.arange(n * 8, dtype=np.uint8).reshape(n, 8),
        "labels": np.asarray([1] * 6 + [0] * 6, dtype=np.int8),
        "chrom": np.asarray(
            ["chrFake1"] * 4 + ["chrFake2"] * 4 + ["chrFake3"] * 4
        ),
        "start": (np.arange(n) * 100).astype(np.int64),
        "end": (np.arange(n) * 100 + 8).astype(np.int64),
        "gc_fraction": (np.arange(n) / n).astype(np.float64),
        "repeat_fraction": (1.0 - np.arange(n) / n).astype(np.float64),
        "gc_bin": (np.arange(n) % 3).astype(np.int16),
        "repeat_bin": ((np.arange(n) + 1) % 3).astype(np.int16),
        "peak_index": np.asarray(
            [0, 1, 2, 3, 4, 5, -1, -1, -1, -1, -1, -1], dtype=np.int32
        ),
        "split": np.asarray(split, dtype=np.int8),
    }
    for name in drop:
        del arrays[name]
    np.savez_compressed(path, **arrays)
    return path


@pytest.fixture()
def dataset_path(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "windows.npz")


def test_load_exposes_shapes_labels_and_split_names(dataset_path: Path) -> None:
    ds = WindowDataset(dataset_path)
    assert ds.n_windows == 12
    assert ds.n_features == 8
    assert ds.labels.tolist() == [1] * 6 + [0] * 6
    assert ds.split_names == ("train", "validation", "test")
    ds.close()


def test_split_indices_are_correct_int64(dataset_path: Path) -> None:
    ds = WindowDataset(dataset_path)
    train = ds.split_indices("train")
    assert train.dtype == np.int64
    assert train.tolist() == [0, 1, 2, 3]
    assert ds.split_indices("validation").tolist() == [4, 5, 6, 7]
    assert ds.split_indices("test").tolist() == [8, 9, 10, 11]
    ds.close()


def test_iter_split_yields_dicts_with_exact_keys_and_known_values(
    dataset_path: Path,
) -> None:
    ds = WindowDataset(dataset_path)
    rows = list(ds.iter_split("validation"))
    assert [row["start"] for row in rows] == [400, 500, 600, 700]
    for row in rows:
        assert isinstance(row, dict)
        assert set(row) == EXPECTED_KEYS
    positive = rows[0]
    assert positive["sequence"].tolist() == list(range(32, 40))
    assert positive["label"] == 1
    assert positive["chrom"] == "chrFake2"
    assert positive["start"] == 400
    assert positive["end"] == 408
    assert positive["gc_fraction"] == pytest.approx(4 / 12)
    assert positive["repeat_fraction"] == pytest.approx(1 - 4 / 12)
    assert positive["gc_bin"] == 1
    assert positive["repeat_bin"] == 2
    assert positive["peak_index"] == 4
    negative = rows[2]
    assert negative["label"] == 0
    assert negative["peak_index"] == -1
    ds.close()


def test_iter_split_shuffle_is_deterministic_permutation(dataset_path: Path) -> None:
    ds = WindowDataset(dataset_path)
    ordered = [(row["start"], row["end"]) for row in ds.iter_split("train")]
    shuffled_a = [(row["start"], row["end"]) for row in ds.iter_split("train", shuffle=True, seed=7)]
    shuffled_b = [(row["start"], row["end"]) for row in ds.iter_split("train", shuffle=True, seed=7)]
    shuffled_c = [(row["start"], row["end"]) for row in ds.iter_split("train", shuffle=True, seed=8)]
    assert shuffled_a == shuffled_b
    assert sorted(shuffled_a) == sorted(ordered)
    assert shuffled_a != ordered
    assert sorted(shuffled_c) == sorted(ordered)
    ds.close()


def test_iter_split_shuffle_does_not_touch_global_rng(dataset_path: Path) -> None:
    ds = WindowDataset(dataset_path)
    np.random.seed(0)
    expected_next = float(np.random.random())
    np.random.seed(0)
    list(ds.iter_split("train", shuffle=True, seed=1))
    assert float(np.random.random()) == expected_next
    ds.close()


def test_unknown_split_name_raises(dataset_path: Path) -> None:
    ds = WindowDataset(dataset_path)
    with pytest.raises(ValueError, match="unknown split"):
        ds.split_indices("heldout")
    with pytest.raises(ValueError, match="unknown split"):
        ds.iter_split("heldout")
    with pytest.raises(ValueError, match="unknown split"):
        ds.counts("heldout")
    ds.close()


def test_empty_split_raises(tmp_path: Path) -> None:
    path = write_dataset(
        tmp_path / "empty_split.npz", split=[0] * 8 + [1] * 4 + [2] * 0
    )
    ds = WindowDataset(path)
    assert ds.split_indices("validation").tolist() == [8, 9, 10, 11]
    with pytest.raises(ValueError, match="no windows"):
        ds.split_indices("test")
    ds.close()


def test_missing_required_array_raises(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "missing_array.npz", drop=("gc_fraction",))
    with pytest.raises(ValueError, match="gc_fraction"):
        WindowDataset(path)


def test_legacy_file_without_split_raises_rerun_prepare(tmp_path: Path) -> None:
    path = write_dataset(tmp_path / "legacy.npz", drop=("split",))
    with pytest.raises(ValueError, match="re-run the prepare step"):
        WindowDataset(path)


def test_counts_per_split_and_per_label(dataset_path: Path) -> None:
    ds = WindowDataset(dataset_path)
    assert ds.counts() == {"train": 4, "validation": 4, "test": 4}
    assert ds.counts("train") == {0: 0, 1: 4}
    assert ds.counts("validation") == {0: 2, 1: 2}
    assert ds.counts("test") == {0: 4, 1: 0}
    ds.close()


def test_context_manager_closes_the_file(dataset_path: Path) -> None:
    with WindowDataset(dataset_path) as ds:
        assert ds.n_windows == 12
        assert len(list(ds.iter_split("test"))) == 4
    with pytest.raises(ValueError, match="closed"):
        ds.n_windows
    with pytest.raises(ValueError, match="closed"):
        list(ds.iter_split("train"))
    ds.close()
