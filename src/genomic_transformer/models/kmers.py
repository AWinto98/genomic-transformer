"""Vectorized k-mer count features for uint8-encoded DNA windows.

Tokens follow the prepare_ctcf encoding: A=0, C=1, G=2, T=3, N=4. Any k-mer
containing N is skipped, and each k-mer is folded together with its reverse
complement so both land on a single canonical feature.
"""

from typing import Optional

import numpy as np

_TOKENS = "ACGT"
DEFAULT_CHUNK_SIZE = 8192
_MAX_K = 9


def canonical_kmer_map(k: int) -> tuple[np.ndarray, list[str]]:
    """Map every k-mer code to a compact canonical feature index.

    The canonical code of a k-mer is min(code, revcomp(code)), so a k-mer and
    its reverse complement share one feature. Returns (code_to_feature, names)
    where code_to_feature has shape (4**k,) int32 and names holds the canonical
    k-mer string for each feature index, sorted lexicographically. For even k
    the feature count is (4**k + 4**(k//2)) / 2, i.e. 2080 for k=6.
    """
    if k < 1 or k > _MAX_K:
        raise ValueError(f"k must be in [1, {_MAX_K}], got {k}")
    n_codes = 4 ** k
    codes = np.arange(n_codes, dtype=np.int64)
    powers_desc = 4 ** np.arange(k - 1, -1, -1)
    digits = (codes[:, None] // powers_desc[None, :]) % 4
    revcomp = ((3 - digits) * powers_desc[::-1][None, :]).sum(axis=1)
    canonical = np.minimum(codes, revcomp)
    _, code_to_feature = np.unique(canonical, return_inverse=True)
    names = [_decode_code(int(code), k) for code in np.unique(canonical)]
    return code_to_feature.astype(np.int32), names


def _decode_code(code: int, k: int) -> str:
    chars = []
    for _ in range(k):
        chars.append(_TOKENS[code & 3])
        code >>= 2
    return "".join(reversed(chars))


def fit_transform_kmers(
    sequences: np.ndarray,
    k: int = 6,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Count canonical k-mer features for every window.

    Returns (features, code_to_feature, feature_names): features is a float32
    [n_windows, n_features] count matrix; code_to_feature maps raw k-mer codes
    to feature columns and can be reused with transform_kmers on other splits.
    """
    code_to_feature, feature_names = canonical_kmer_map(k)
    features = transform_kmers(sequences, code_to_feature, k=k, chunk_size=chunk_size)
    return features, code_to_feature, feature_names


def transform_kmers(
    sequences: np.ndarray,
    code_to_feature: np.ndarray,
    k: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> np.ndarray:
    """Count canonical k-mers using a precomputed code_to_feature map."""
    sequences = np.asarray(sequences)
    if sequences.ndim != 2:
        raise ValueError(f"sequences must be a 2-D array, got shape {sequences.shape}")
    if k is None:
        k = _infer_k(code_to_feature)
    if sequences.shape[1] < k:
        raise ValueError(
            f"window length {sequences.shape[1]} is shorter than k={k}"
        )
    n_windows = sequences.shape[0]
    n_features = int(code_to_feature.max()) + 1
    features = np.zeros((n_windows, n_features), dtype=np.float32)
    for start in range(0, n_windows, chunk_size):
        block = sequences[start : start + chunk_size]
        features[start : start + block.shape[0]] = _count_kmer_block(
            block, code_to_feature, k
        )
    return features


def _infer_k(code_to_feature: np.ndarray) -> int:
    k = int(round(np.log2(code_to_feature.size) / 2.0))
    if 4 ** k != code_to_feature.size:
        raise ValueError(
            f"code_to_feature has size {code_to_feature.size}, not a power of 4"
        )
    return k


def _count_kmer_block(
    sequences: np.ndarray, code_to_feature: np.ndarray, k: int
) -> np.ndarray:
    """Count canonical k-mers for one block of windows (vectorized)."""
    n_windows, width = sequences.shape
    n_features = int(code_to_feature.max()) + 1
    n_positions = width - k + 1
    powers_desc = 4 ** np.arange(k - 1, -1, -1)
    codes = np.zeros((n_windows, n_positions), dtype=np.int32)
    valid = np.ones((n_windows, n_positions), dtype=bool)
    for offset in range(k):
        column = sequences[:, offset : offset + n_positions]
        codes += column.astype(np.int32) * int(powers_desc[offset])
        valid &= column < 4
    # N tokens can push codes past 4**k - 1; remap invalid positions to a
    # dummy in-bounds code, then drop them from the flat counts via `valid`.
    safe_codes = np.where(valid, codes, 0)
    features = code_to_feature[safe_codes]
    flat = (
        np.arange(n_windows, dtype=np.int64)[:, None] * n_features + features
    ).ravel()
    flat = flat[valid.ravel()]
    counts = np.bincount(flat, minlength=n_windows * n_features)
    return counts.reshape(n_windows, n_features).astype(np.float32)


def fit_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std from TRAINING features only.

    Zero-variance features get scale 1 so downstream z-scores stay finite.
    """
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0.0] = 1.0
    return mean, scale


def apply_normalizer(
    features: np.ndarray, mean: np.ndarray, scale: np.ndarray
) -> np.ndarray:
    """Z-score features with precomputed train statistics (no leakage)."""
    return ((features - mean) / scale).astype(np.float32)
