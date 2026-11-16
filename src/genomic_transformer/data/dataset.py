"""Lazy, framework-agnostic access to a prepared windows.npz dataset."""

from pathlib import Path
from typing import Any, Iterator, Optional, Union

import numpy as np


class WindowDataset:
    """Row-wise window loader backed by a memory-mapped npz file.

    Expects the artifact written by prepare_ctcf: positives first, then
    negatives, with per-window sequences, labels, coordinates, and a split
    array (0=train, 1=validation, 2=test).
    """

    _SPLIT_VALUES = {"train": 0, "validation": 1, "test": 2}
    _REQUIRED_ARRAYS = (
        "sequences",
        "labels",
        "chrom",
        "start",
        "end",
        "gc_fraction",
        "repeat_fraction",
        "gc_bin",
        "repeat_bin",
        "peak_index",
    )

    def __init__(self, npz_path: Union[str, Path]) -> None:
        self._path = Path(npz_path)
        self._data = np.load(self._path, mmap_mode="r", allow_pickle=False)
        self._closed = False
        try:
            self._validate()
        except Exception:
            self.close()
            raise

    def _validate(self) -> None:
        names = set(self._data.files)
        if "split" not in names:
            raise ValueError(
                f"{self._path} has no 'split' array; the dataset schema changed, "
                "so re-run the prepare step to regenerate the dataset"
            )
        missing = [name for name in self._REQUIRED_ARRAYS if name not in names]
        if missing:
            raise ValueError(
                f"{self._path} is missing required arrays: {missing}; "
                "re-run the prepare step to regenerate the dataset"
            )

    def _require_open(self) -> None:
        if self._closed:
            raise ValueError("WindowDataset is closed")

    def _split_value(self, split: str) -> int:
        if split not in self._SPLIT_VALUES:
            raise ValueError(
                f"unknown split '{split}'; expected one of {self.split_names}"
            )
        return self._SPLIT_VALUES[split]

    @property
    def n_windows(self) -> int:
        """Total number of windows in the dataset."""
        self._require_open()
        return int(self._data["sequences"].shape[0])

    @property
    def n_features(self) -> int:
        """Window length in tokens (sequences.shape[1])."""
        self._require_open()
        return int(self._data["sequences"].shape[1])

    @property
    def labels(self) -> np.ndarray:
        """Window labels as a 1-D array (1=positive, 0=negative)."""
        self._require_open()
        return np.asarray(self._data["labels"])

    @property
    def split_names(self) -> tuple[str, str, str]:
        """Split names in split-value order: train, validation, test."""
        return ("train", "validation", "test")

    def split_indices(self, split: str) -> np.ndarray:
        """Return int64 row indices of the windows in the named split.

        Raises ValueError for an unknown split name or an empty split.
        """
        self._require_open()
        value = self._split_value(split)
        indices = np.flatnonzero(np.asarray(self._data["split"]) == value)
        if indices.size == 0:
            raise ValueError(
                f"split '{split}' has no windows; check the prepare config"
            )
        return indices.astype(np.int64, copy=False)

    def iter_split(
        self,
        split: str,
        shuffle: bool = False,
        seed: Optional[int] = None,
    ) -> Iterator[dict[str, Any]]:
        """Iterate windows of a split as dicts, in order or shuffled.

        Shuffling uses a local np.random.default_rng(seed) and never touches
        global numpy RNG state. Each yielded dict has the keys: sequence,
        label, chrom, start, end, gc_fraction, repeat_fraction, gc_bin,
        repeat_bin, peak_index.
        """
        indices = self.split_indices(split)
        if shuffle:
            indices = np.random.default_rng(seed).permutation(indices)
        return self._iter_windows(indices)

    def _iter_windows(self, indices: np.ndarray) -> Iterator[dict[str, Any]]:
        self._require_open()
        data = self._data
        sequences = data["sequences"]
        labels = data["labels"]
        chroms = data["chrom"]
        starts = data["start"]
        ends = data["end"]
        gc_fractions = data["gc_fraction"]
        repeat_fractions = data["repeat_fraction"]
        gc_bins = data["gc_bin"]
        repeat_bins = data["repeat_bin"]
        peak_indices = data["peak_index"]
        for index in indices:
            i = int(index)
            yield {
                "sequence": sequences[i],
                "label": int(labels[i]),
                "chrom": str(chroms[i]),
                "start": int(starts[i]),
                "end": int(ends[i]),
                "gc_fraction": float(gc_fractions[i]),
                "repeat_fraction": float(repeat_fractions[i]),
                "gc_bin": int(gc_bins[i]),
                "repeat_bin": int(repeat_bins[i]),
                "peak_index": int(peak_indices[i]),
            }

    def counts(self, split: Optional[str] = None) -> dict[Any, int]:
        """Window counts.

        With split=None, return {split_name: window_count} across the whole
        dataset. With a split name, return {0: n_negative, 1: n_positive}
        within that split.
        """
        self._require_open()
        split_array = np.asarray(self._data["split"])
        if split is None:
            return {
                name: int(np.count_nonzero(split_array == value))
                for name, value in self._SPLIT_VALUES.items()
            }
        value = self._split_value(split)
        labels = np.asarray(self._data["labels"])[split_array == value]
        return {
            0: int(np.count_nonzero(labels == 0)),
            1: int(np.count_nonzero(labels == 1)),
        }

    def close(self) -> None:
        """Close the underlying npz handle."""
        if not self._closed:
            self._closed = True
            self._data.close()

    def __enter__(self) -> "WindowDataset":
        self._require_open()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.close()
