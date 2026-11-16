import argparse
import bisect
import gzip
import hashlib
import json
import os
import re
import shutil
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pyfaidx

from genomic_transformer.config import load_config


@dataclass(frozen=True)
class Peak:
    chrom: str
    start: int
    end: int
    name: str
    signal_value: float
    p_value: float
    q_value: float
    summit_offset: int


@dataclass(frozen=True)
class Window:
    chrom: str
    start: int
    end: int
    gc_fraction: float
    repeat_fraction: float
    gc_bin: int
    repeat_bin: int
    peak: Optional[Peak] = None


class IntervalIndex:
    def __init__(self, intervals: dict[str, list[tuple[int, int]]]) -> None:
        self.intervals = {
            chrom: _merge_intervals(values) for chrom, values in intervals.items()
        }
        self.starts = {
            chrom: [start for start, _ in values]
            for chrom, values in self.intervals.items()
        }

    def overlaps(self, chrom: str, start: int, end: int) -> bool:
        return _overlaps_sorted(self.intervals.get(chrom, []), self.starts.get(chrom, []), start, end)


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return merged


def _overlaps_sorted(
    intervals: list[tuple[int, int]], starts: list[int], start: int, end: int
) -> bool:
    index = bisect.bisect_left(starts, end) - 1
    return index >= 0 and intervals[index][1] > start


def file_md5(path: Union[str, Path], chunk_size: int = 1048576) -> str:
    digest = hashlib.md5()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Union[str, Path], expected_bytes: int, expected_md5: str) -> None:
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Required source file is missing: {file_path}; run download first")
    actual_bytes = file_path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"Source size mismatch for {file_path}: expected {expected_bytes}, got {actual_bytes}"
        )
    actual_md5 = file_md5(file_path)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Source MD5 mismatch for {file_path}: expected {expected_md5}, got {actual_md5}"
        )


def _finalize_partial(partial: Path, path: Path, expected_md5: str) -> None:
    actual_md5 = file_md5(partial)
    if actual_md5 != expected_md5:
        raise ValueError(
            f"Downloaded MD5 mismatch for {path}: expected {expected_md5}, got {actual_md5}"
        )
    os.replace(partial, path)


def _validate_content_range(
    header: Optional[str],
    offset: int,
    expected_bytes: int,
    expected_end: Optional[int] = None,
) -> None:
    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", header or "", re.IGNORECASE)
    if match is None:
        raise ValueError(f"Invalid or missing Content-Range for resumed download: {header}")
    start, end, total = (int(value) for value in match.groups())
    if (
        start != offset
        or total != expected_bytes
        or end < start
        or end >= total
        or (expected_end is not None and end != expected_end)
    ):
        raise ValueError(
            f"Unexpected Content-Range for resumed download: {header}; "
            f"expected start {offset}, end {expected_end}, and total {expected_bytes}"
        )


def _download_attempt(
    source: dict[str, Any], options: dict[str, Any], path: Path, partial: Path
) -> bool:
    expected_bytes = int(source["bytes"])
    expected_md5 = source["md5"]
    offset = partial.stat().st_size if partial.exists() else 0
    if offset == expected_bytes:
        try:
            _finalize_partial(partial, path, expected_md5)
            return True
        except ValueError:
            partial.unlink()
            offset = 0
    elif offset > expected_bytes:
        partial.unlink()
        offset = 0

    headers = {"Range": f"bytes={offset}-"} if offset else {}
    request = urllib.request.Request(source["url"], headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=int(options["timeout_seconds"]))
    except urllib.error.HTTPError as error:
        if error.code != 416:
            raise
        error.close()
        if not partial.exists() or partial.stat().st_size != expected_bytes:
            raise ValueError(
                f"Server returned 416 but partial size is not {expected_bytes}: {partial}"
            ) from error
        _finalize_partial(partial, path, expected_md5)
        return True

    with response:
        status = response.getcode()
        if offset and status == 206:
            _validate_content_range(
                response.headers.get("Content-Range"), offset, expected_bytes
            )
            mode = "ab"
        elif status == 200:
            mode = "wb"
            offset = 0
        else:
            raise ValueError(
                f"Unexpected HTTP status {status} for download at offset {offset}: {source['url']}"
            )
        bytes_written = offset
        with partial.open(mode) as stream:
            while True:
                chunk = response.read(int(options["chunk_size"]))
                if not chunk:
                    break
                bytes_written += len(chunk)
                if bytes_written > expected_bytes:
                    raise ValueError(f"Download exceeds expected size for {source['url']}")
                stream.write(chunk)
            stream.flush()
            os.fsync(stream.fileno())
    if bytes_written != expected_bytes:
        raise ValueError(
            f"Downloaded size mismatch for {source['url']}: expected {expected_bytes}, "
            f"got {bytes_written}"
        )
    _finalize_partial(partial, path, expected_md5)
    return True


def _parallel_state_path(state_dir: Path) -> Path:
    return state_dir / "state.json"


def _read_parallel_state(state_dir: Path) -> Optional[dict[str, Any]]:
    try:
        with _parallel_state_path(state_dir).open("r", encoding="utf-8") as stream:
            state = json.load(stream)
        return state if isinstance(state, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_parallel_state(state_dir: Path, state: dict[str, Any]) -> None:
    state_dir.mkdir(parents=True, exist_ok=True)
    path = _parallel_state_path(state_dir)
    temporary = path.with_name(path.name + ".partial")
    temporary.unlink(missing_ok=True)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(state, stream, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _prepare_parallel_state(
    source: dict[str, Any], partial: Path, state_dir: Path, segment_size: int
) -> int:
    source_identity = {
        "url": source["url"],
        "bytes": int(source["bytes"]),
        "md5": source["md5"],
    }
    previous = _read_parallel_state(state_dir) if state_dir.exists() else None
    if state_dir.exists() and (
        previous is None or previous.get("source") != source_identity
    ):
        shutil.rmtree(state_dir)
        partial.unlink(missing_ok=True)
    prefix_size = partial.stat().st_size if partial.exists() else 0
    state = {
        "source": source_identity,
        "segment_size": segment_size,
        "prefix_size": prefix_size,
        "prefix_md5": file_md5(partial) if prefix_size else None,
    }
    if state_dir.exists() and _read_parallel_state(state_dir) != state:
        shutil.rmtree(state_dir)
    _write_parallel_state(state_dir, state)
    return prefix_size


def _segment_path(state_dir: Path, start: int, end: int) -> Path:
    return state_dir / f"{start:020d}-{end:020d}.segment"


def _download_segment(
    source: dict[str, Any],
    options: dict[str, Any],
    state_dir: Path,
    start: int,
    end: int,
) -> None:
    expected_bytes = int(source["bytes"])
    segment_length = end - start + 1
    segment = _segment_path(state_dir, start, end)
    temporary = segment.with_name(segment.name + ".partial")
    if segment.exists():
        if segment.stat().st_size == segment_length:
            return
        segment.unlink()
    if temporary.exists() and temporary.stat().st_size > segment_length:
        temporary.unlink()

    max_retries = int(options["max_retries"])
    backoff_seconds = float(options["backoff_seconds"])
    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            existing = temporary.stat().st_size if temporary.exists() else 0
            if existing == segment_length:
                os.replace(temporary, segment)
                return
            request_start = start + existing
            request = urllib.request.Request(
                source["url"], headers={"Range": f"bytes={request_start}-{end}"}
            )
            with urllib.request.urlopen(
                request, timeout=int(options["timeout_seconds"])
            ) as response:
                if response.getcode() != 206:
                    raise ValueError(
                        f"Parallel range request requires HTTP 206, got {response.getcode()} "
                        f"for bytes={request_start}-{end}"
                    )
                _validate_content_range(
                    response.headers.get("Content-Range"),
                    request_start,
                    expected_bytes,
                    expected_end=end,
                )
                bytes_written = existing
                with temporary.open("ab") as stream:
                    while True:
                        chunk = response.read(int(options["chunk_size"]))
                        if not chunk:
                            break
                        bytes_written += len(chunk)
                        if bytes_written > segment_length:
                            raise ValueError(
                                f"Segment exceeds requested range bytes={start}-{end}"
                            )
                        stream.write(chunk)
                    stream.flush()
                    os.fsync(stream.fileno())
            if bytes_written != segment_length:
                raise ValueError(
                    f"Segment size mismatch for bytes={start}-{end}: "
                    f"expected {segment_length}, got {bytes_written}"
                )
            os.replace(temporary, segment)
            return
        except Exception as error:
            last_error = error
            if temporary.exists() and temporary.stat().st_size > segment_length:
                temporary.unlink()
            if attempt < max_retries:
                time.sleep(backoff_seconds)
    raise ValueError(
        f"Segment bytes={start}-{end} failed after {max_retries + 1} attempts: {last_error}"
    ) from last_error


def _copy_complete_file(
    source: Path, output: Any, expected_bytes: int, chunk_size: int
) -> None:
    copied = 0
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            copied += len(chunk)
            output.write(chunk)
    if copied != expected_bytes:
        raise ValueError(
            f"Assembly input size mismatch for {source}: expected {expected_bytes}, got {copied}"
        )


def _download_parallel(
    source: dict[str, Any], options: dict[str, Any], path: Path, partial: Path
) -> bool:
    expected_bytes = int(source["bytes"])
    parallel = options["parallel"]
    connections = int(parallel["connections"])
    segment_size = int(parallel["segment_size"])
    state_dir = path.with_name(path.name + ".segments")
    prefix_size = _prepare_parallel_state(source, partial, state_dir, segment_size)
    ranges = [
        (start, min(start + segment_size, expected_bytes) - 1)
        for start in range(prefix_size, expected_bytes, segment_size)
    ]
    with ThreadPoolExecutor(max_workers=connections) as executor:
        futures = [
            executor.submit(_download_segment, source, options, state_dir, start, end)
            for start, end in ranges
        ]
        for future in futures:
            future.result()

    assembly = path.with_name(path.name + ".assemble")
    assembly.unlink(missing_ok=True)
    try:
        with assembly.open("xb") as output:
            if prefix_size:
                _copy_complete_file(
                    partial, output, prefix_size, int(options["chunk_size"])
                )
            for start, end in ranges:
                _copy_complete_file(
                    _segment_path(state_dir, start, end),
                    output,
                    end - start + 1,
                    int(options["chunk_size"]),
                )
            output.flush()
            os.fsync(output.fileno())
        if assembly.stat().st_size != expected_bytes:
            raise ValueError(
                f"Assembled size mismatch for {path}: expected {expected_bytes}, "
                f"got {assembly.stat().st_size}"
            )
        actual_md5 = file_md5(assembly)
        if actual_md5 != source["md5"]:
            shutil.rmtree(state_dir)
            partial.unlink(missing_ok=True)
            raise ValueError(
                f"Downloaded MD5 mismatch for {path}: expected {source['md5']}, "
                f"got {actual_md5}"
            )
    except Exception:
        assembly.unlink(missing_ok=True)
        raise
    os.replace(assembly, path)
    shutil.rmtree(state_dir)
    partial.unlink(missing_ok=True)
    return True


def download_source(source: dict[str, Any], options: dict[str, Any]) -> bool:
    path = Path(source["path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    expected_bytes = int(source["bytes"])
    expected_md5 = source["md5"]
    if path.exists():
        verify_file(path, expected_bytes, expected_md5)
        return False

    max_retries = int(options["max_retries"])
    backoff_seconds = float(options["backoff_seconds"])
    if max_retries < 0 or backoff_seconds < 0:
        raise ValueError("Download max_retries and backoff_seconds must be non-negative")
    partial = path.with_name(path.name + ".partial")
    state_dir = path.with_name(path.name + ".segments")
    if partial.exists() and partial.stat().st_size >= expected_bytes:
        if partial.stat().st_size == expected_bytes:
            try:
                _finalize_partial(partial, path, expected_md5)
                if state_dir.exists():
                    shutil.rmtree(state_dir)
                return True
            except ValueError:
                partial.unlink()
        else:
            partial.unlink()

    parallel = options["parallel"]
    connections = int(parallel["connections"])
    segment_size = int(parallel["segment_size"])
    minimum_size = int(parallel["min_file_size"])
    if connections <= 0 or segment_size <= 0 or minimum_size < 0:
        raise ValueError("Parallel download connections and segment_size must be positive")
    if connections > 1 and expected_bytes >= minimum_size:
        return _download_parallel(source, options, path, partial)

    last_error: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            result = _download_attempt(source, options, path, partial)
            if state_dir.exists():
                shutil.rmtree(state_dir)
            return result
        except Exception as error:
            last_error = error
            if partial.exists() and partial.stat().st_size >= expected_bytes:
                partial.unlink()
            if attempt < max_retries:
                time.sleep(backoff_seconds)
    raise ValueError(
        f"Download failed after {max_retries + 1} attempts for {source['url']}: {last_error}"
    ) from last_error


def download_all(config: dict[str, Any]) -> list[Path]:
    downloaded = []
    for source in config["sources"].values():
        if download_source(source, config["download"]):
            downloaded.append(Path(source["path"]))
    return downloaded


def _open_text(path: Path, compression: str) -> Any:
    if compression == "gzip":
        return gzip.open(path, "rt", encoding="utf-8")
    if compression == "none":
        return path.open("r", encoding="utf-8")
    raise ValueError(f"Unsupported compression: {compression}")


def _fields(line: str) -> Optional[list[str]]:
    stripped = line.strip()
    if not stripped or stripped.startswith(("#", "track", "browser")):
        return None
    return stripped.split()


def _interval(fields: list[str], path: Path, line_number: int) -> tuple[str, int, int]:
    if len(fields) < 3:
        raise ValueError(f"BED row has fewer than 3 columns: {path}:{line_number}")
    chrom = fields[0]
    try:
        start, end = int(fields[1]), int(fields[2])
    except ValueError as error:
        raise ValueError(f"Invalid BED coordinates: {path}:{line_number}") from error
    if start < 0 or end <= start:
        raise ValueError(f"BED requires 0-based half-open start < end: {path}:{line_number}")
    return chrom, start, end


def parse_peaks(
    path: Path,
    source: dict[str, Any],
    canonical: set[str],
    chrom_lengths: dict[str, int],
    processing: dict[str, Any],
) -> list[Peak]:
    peaks = []
    summit_column = int(processing["summit_offset_column"]) - 1
    with _open_text(path, source["compression"]) as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = _fields(line)
            if fields is None:
                continue
            chrom, start, end = _interval(fields, path, line_number)
            if chrom not in canonical:
                continue
            if len(fields) <= summit_column:
                raise ValueError(
                    f"narrowPeak row lacks summit column {summit_column + 1}: "
                    f"{path}:{line_number}"
                )
            try:
                summit_offset = int(fields[summit_column])
            except ValueError as error:
                raise ValueError(f"Invalid summit offset: {path}:{line_number}") from error
            if summit_offset < int(processing["missing_summit"]):
                raise ValueError(f"Invalid summit offset: {path}:{line_number}")
            if end > chrom_lengths[chrom]:
                raise ValueError(f"Peak exceeds reference bounds: {path}:{line_number}")
            peaks.append(
                Peak(
                    chrom=chrom,
                    start=start,
                    end=end,
                    name=fields[3] if len(fields) > 3 else f"{chrom}:{start}-{end}",
                    signal_value=float(fields[6]) if len(fields) > 6 else float("nan"),
                    p_value=float(fields[7]) if len(fields) > 7 else float("nan"),
                    q_value=float(fields[8]) if len(fields) > 8 else float("nan"),
                    summit_offset=summit_offset,
                )
            )
    return peaks


def parse_blacklist(
    path: Path,
    source: dict[str, Any],
    canonical: set[str],
    chrom_lengths: dict[str, int],
) -> dict[str, list[tuple[int, int]]]:
    intervals = {chrom: [] for chrom in canonical}
    with _open_text(path, source["compression"]) as stream:
        for line_number, line in enumerate(stream, start=1):
            fields = _fields(line)
            if fields is None:
                continue
            chrom, start, end = _interval(fields, path, line_number)
            if chrom not in canonical:
                continue
            if end > chrom_lengths[chrom]:
                raise ValueError(f"Blacklist interval exceeds reference bounds: {path}:{line_number}")
            intervals[chrom].append((start, end))
    return intervals


def ensure_reference(source: dict[str, Any], chunk_size: int) -> pyfaidx.Fasta:
    compressed = Path(source["path"])
    fasta_path = Path(source["decompressed_path"])
    index_path = Path(source["index_path"])
    fasta_path.parent.mkdir(parents=True, exist_ok=True)
    if not fasta_path.exists():
        partial = fasta_path.with_name(fasta_path.name + ".partial")
        partial.unlink(missing_ok=True)
        try:
            with gzip.open(compressed, "rb") as input_stream, partial.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=chunk_size)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            os.replace(partial, fasta_path)
        except Exception:
            partial.unlink(missing_ok=True)
            raise
    if not index_path.exists():
        partial_index = index_path.with_name(index_path.name + ".partial")
        partial_index.unlink(missing_ok=True)
        try:
            indexed = pyfaidx.Fasta(str(fasta_path), indexname=str(partial_index))
            indexed.close()
            os.replace(partial_index, index_path)
        except Exception:
            partial_index.unlink(missing_ok=True)
            raise
    return pyfaidx.Fasta(str(fasta_path), indexname=str(index_path))


def sequence_metrics(sequence: str, gc_denominator: str) -> tuple[float, float, float]:
    upper = sequence.upper()
    length = len(sequence)
    gc_count = upper.count("G") + upper.count("C")
    denominator = sum(upper.count(base) for base in gc_denominator)
    gc_fraction = gc_count / denominator if denominator else 0.0
    repeat_fraction = sum(base.islower() for base in sequence) / length
    n_fraction = upper.count("N") / length
    return gc_fraction, repeat_fraction, n_fraction


def bin_index(value: float, edges: list[float]) -> int:
    if value < edges[0] or value > edges[-1]:
        raise ValueError(f"Value {value} is outside configured bin edges {edges}")
    return min(bisect.bisect_right(edges, value) - 1, len(edges) - 2)


def encode_sequence(sequence: str, encoding: dict[str, Any]) -> np.ndarray:
    if encoding["dtype"] != "uint8" or encoding["input_case"] != "upper":
        raise ValueError("CTCF pilot encoding requires uppercase input and uint8 tokens")
    tokens = encoding["tokens"]
    unknown = encoding["unknown_character"]
    lookup = np.full(256, int(tokens[unknown]), dtype=np.uint8)
    for character in encoding["alphabet"]:
        lookup[ord(character.upper())] = int(tokens[character])
    return lookup[np.frombuffer(sequence.upper().encode("ascii"), dtype=np.uint8)]


def _sequence(fasta: pyfaidx.Fasta, chrom: str, start: int, end: int) -> str:
    sequence = str(fasta[chrom][start:end])
    if len(sequence) != end - start:
        raise ValueError(f"Could not read complete window {chrom}:{start}-{end}")
    return sequence


def _window(
    fasta: pyfaidx.Fasta,
    chrom: str,
    start: int,
    processing: dict[str, Any],
    peak: Optional[Peak] = None,
) -> tuple[Window, float]:
    window_length = int(processing["window_length"])
    sequence = _sequence(fasta, chrom, start, start + window_length)
    gc, repeat, n_fraction = sequence_metrics(
        sequence, processing["encoding"]["gc_denominator"]
    )
    matching = processing["matching"]
    window = Window(
        chrom=chrom,
        start=start,
        end=start + window_length,
        gc_fraction=gc,
        repeat_fraction=repeat,
        gc_bin=bin_index(gc, matching["gc_bin_edges"]),
        repeat_bin=bin_index(repeat, matching["repeat_bin_edges"]),
        peak=peak,
    )
    return window, n_fraction


def select_positive_windows(
    fasta: pyfaidx.Fasta,
    peaks: list[Peak],
    blacklist: IntervalIndex,
    processing: dict[str, Any],
) -> tuple[list[Window], dict[str, int]]:
    selected = []
    rejected = {
        "missing_summit": 0,
        "out_of_bounds": 0,
        "blacklist_overlap": 0,
        "excess_n": 0,
    }
    window_length = int(processing["window_length"])
    for peak in peaks:
        if peak.summit_offset == int(processing["missing_summit"]):
            rejected["missing_summit"] += 1
            continue
        summit = peak.start + peak.summit_offset
        start = summit - window_length // 2
        end = start + window_length
        if start < 0 or end > len(fasta[peak.chrom]):
            rejected["out_of_bounds"] += 1
            continue
        if blacklist.overlaps(peak.chrom, start, end):
            rejected["blacklist_overlap"] += 1
            continue
        window, n_fraction = _window(fasta, peak.chrom, start, processing, peak)
        if n_fraction > float(processing["max_n_fraction"]):
            rejected["excess_n"] += 1
            continue
        selected.append(window)
    return selected, rejected


def sample_negative_windows(
    fasta: pyfaidx.Fasta,
    positives: list[Window],
    exclusion: IntervalIndex,
    processing: dict[str, Any],
) -> tuple[list[Window], int, list[dict[str, Any]]]:
    matching = processing["matching"]
    ratio = int(matching["negatives_per_positive"])
    window_length = int(processing["window_length"])
    batch_size = int(matching["batch_size"])
    max_attempts = int(matching["max_attempts_per_negative"])
    allow_relaxed = bool(matching.get("allow_relaxed_fallback", False))
    n_gc_bins = len(matching["gc_bin_edges"]) - 1
    n_repeat_bins = len(matching["repeat_bin_edges"]) - 1
    max_ring = max(n_gc_bins, n_repeat_bins) - 1
    rng = np.random.default_rng(int(matching["seed"]))
    groups: dict[tuple[str, int, int], list[Window]] = {}
    for positive in positives:
        key = (positive.chrom, positive.gc_bin, positive.repeat_bin)
        groups.setdefault(key, []).append(positive)

    negatives: list[Window] = []
    selected_intervals = {chrom: [] for chrom in fasta.keys()}
    selected_starts = {chrom: [] for chrom in fasta.keys()}
    total_attempts = 0
    relaxed_strata: list[dict[str, Any]] = []
    for (chrom, gc_bin, repeat_bin), group in groups.items():
        required = len(group) * ratio
        accepted = 0
        attempts = 0
        attempt_limit = required * max_attempts
        max_start = len(fasta[chrom]) - window_length
        while accepted < required and attempts < attempt_limit:
            current_batch = min(batch_size, attempt_limit - attempts)
            starts = rng.integers(0, max_start + 1, size=current_batch)
            attempts += current_batch
            total_attempts += current_batch
            for start_value in starts:
                start = int(start_value)
                end = start + window_length
                if exclusion.overlaps(chrom, start, end) or _overlaps_sorted(
                    selected_intervals[chrom], selected_starts[chrom], start, end
                ):
                    continue
                window, n_fraction = _window(fasta, chrom, start, processing)
                if n_fraction > float(processing["max_n_fraction"]):
                    continue
                if window.gc_bin != gc_bin or window.repeat_bin != repeat_bin:
                    continue
                negatives.append(window)
                bisect.insort(selected_intervals[chrom], (start, end))
                selected_starts[chrom].insert(
                    bisect.bisect_right(selected_starts[chrom], start), start
                )
                accepted += 1
                if accepted == required:
                    break
        if accepted < required and allow_relaxed:
            accepted_before = accepted
            ring = 0
            while accepted < required and ring < max_ring:
                ring += 1
                allowed_bins = {
                    (other_gc, other_repeat)
                    for other_gc in range(
                        max(0, gc_bin - ring), min(n_gc_bins, gc_bin + ring + 1)
                    )
                    for other_repeat in range(
                        max(0, repeat_bin - ring),
                        min(n_repeat_bins, repeat_bin + ring + 1),
                    )
                    if (other_gc, other_repeat) != (gc_bin, repeat_bin)
                }
                ring_attempts = 0
                ring_attempt_limit = max_attempts * (required - accepted)
                while accepted < required and ring_attempts < ring_attempt_limit:
                    current_batch = min(batch_size, ring_attempt_limit - ring_attempts)
                    starts = rng.integers(0, max_start + 1, size=current_batch)
                    ring_attempts += current_batch
                    total_attempts += current_batch
                    for start_value in starts:
                        start = int(start_value)
                        end = start + window_length
                        if exclusion.overlaps(chrom, start, end) or _overlaps_sorted(
                            selected_intervals[chrom], selected_starts[chrom], start, end
                        ):
                            continue
                        window, n_fraction = _window(fasta, chrom, start, processing)
                        if n_fraction > float(processing["max_n_fraction"]):
                            continue
                        if (window.gc_bin, window.repeat_bin) not in allowed_bins:
                            continue
                        negatives.append(window)
                        bisect.insort(selected_intervals[chrom], (start, end))
                        selected_starts[chrom].insert(
                            bisect.bisect_right(selected_starts[chrom], start), start
                        )
                        accepted += 1
                        if accepted == required:
                            break
            if accepted > accepted_before:
                relaxed_accepted = accepted - accepted_before
                print(
                    "Relaxed-bin fallback filled stratum: "
                    f"chrom={chrom}, gc_bin={gc_bin}, repeat_bin={repeat_bin}, "
                    f"ring={ring}, count={relaxed_accepted}"
                )
                relaxed_strata.append(
                    {
                        "chrom": chrom,
                        "gc_bin": gc_bin,
                        "repeat_bin": repeat_bin,
                        "ring": ring,
                        "count": relaxed_accepted,
                    }
                )
        if accepted != required:
            raise RuntimeError(
                "Negative matching exhausted configured attempts: "
                f"chrom={chrom}, gc_bin={gc_bin}, repeat_bin={repeat_bin}, "
                f"required={required}, selected={accepted}, attempts={attempts}, "
                f"max_attempts={attempt_limit}. No relaxed-bin fallback is permitted."
            )
    return negatives, total_attempts, relaxed_strata


def assign_splits(
    config: dict[str, Any], chroms: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    splits = config.get("splits")
    if splits is None:
        raise ValueError(
            "Config is missing the required 'splits' section; define 'validation' and "
            "'test' chromosome lists (train is all canonical chromosomes not listed)"
        )
    canonical = set(config["processing"]["canonical_chromosomes"])
    validation = [str(chrom) for chrom in splits.get("validation", [])]
    test = [str(chrom) for chrom in splits.get("test", [])]
    overlap = sorted(set(validation) & set(test))
    if overlap:
        raise ValueError(f"splits.validation and splits.test must not overlap: {overlap}")
    unknown = sorted((set(validation) | set(test)) - canonical)
    if unknown:
        raise ValueError(
            "splits.validation and splits.test must name chromosomes from "
            f"processing.canonical_chromosomes, got: {unknown}"
        )
    held_out = {chrom: 1 for chrom in validation}
    held_out.update({chrom: 2 for chrom in test})
    split = np.asarray([held_out.get(chrom, 0) for chrom in chroms], dtype=np.int8)
    scheme = {
        "validation": validation,
        "test": test,
        "counts": {
            "train": int(np.count_nonzero(split == 0)),
            "validation": int(np.count_nonzero(split == 1)),
            "test": int(np.count_nonzero(split == 2)),
        },
    }
    return split, scheme


def write_outputs(
    config: dict[str, Any],
    config_path: Path,
    positives: list[Window],
    negatives: list[Window],
    fasta: pyfaidx.Fasta,
    input_peak_count: int,
    rejected: dict[str, int],
    negative_attempts: int,
    relaxed_strata: list[dict[str, Any]],
) -> None:
    windows = positives + negatives
    processing = config["processing"]
    encoding = processing["encoding"]
    sequences = np.stack(
        [
            encode_sequence(_sequence(fasta, w.chrom, w.start, w.end), encoding)
            for w in windows
        ]
    )
    peaks = [w.peak for w in windows]
    chroms = np.asarray([w.chrom for w in windows], dtype=np.str_)
    split, split_scheme = assign_splits(config, chroms)
    arrays = {
        "sequences": sequences,
        "labels": np.asarray([1] * len(positives) + [0] * len(negatives), dtype=np.int8),
        "chrom": chroms,
        "start": np.asarray([w.start for w in windows], dtype=np.int64),
        "end": np.asarray([w.end for w in windows], dtype=np.int64),
        "gc_fraction": np.asarray([w.gc_fraction for w in windows], dtype=np.float64),
        "repeat_fraction": np.asarray([w.repeat_fraction for w in windows], dtype=np.float64),
        "gc_bin": np.asarray([w.gc_bin for w in windows], dtype=np.int16),
        "repeat_bin": np.asarray([w.repeat_bin for w in windows], dtype=np.int16),
        "split": split,
        "peak_index": np.asarray(
            [index if peak is not None else -1 for index, peak in enumerate(peaks)],
            dtype=np.int32,
        ),
        "peak_start": np.asarray([p.start if p else -1 for p in peaks], dtype=np.int64),
        "peak_end": np.asarray([p.end if p else -1 for p in peaks], dtype=np.int64),
        "peak_summit": np.asarray(
            [p.start + p.summit_offset if p else -1 for p in peaks], dtype=np.int64
        ),
        "peak_signal_value": np.asarray(
            [p.signal_value if p else np.nan for p in peaks], dtype=np.float64
        ),
        "peak_p_value": np.asarray([p.p_value if p else np.nan for p in peaks], dtype=np.float64),
        "peak_q_value": np.asarray([p.q_value if p else np.nan for p in peaks], dtype=np.float64),
    }
    dataset_path = Path(config["paths"]["dataset"])
    manifest_path = Path(config["paths"]["manifest"])
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_partial = dataset_path.with_name(dataset_path.name + ".partial")
    manifest_partial = manifest_path.with_name(manifest_path.name + ".partial")
    dataset_partial.unlink(missing_ok=True)
    manifest_partial.unlink(missing_ok=True)
    try:
        with dataset_partial.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        manifest = {
            "dataset": config["dataset"],
            "config": {
                "path": str(config_path),
                "sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
            },
            "sources": config["sources"],
            "parameters": processing,
            "splits": split_scheme,
            "counts": {
                "input_canonical_peaks": input_peak_count,
                "positive": len(positives),
                "negative": len(negatives),
                "total": len(windows),
                "rejected_positive": rejected,
                "negative_candidates_sampled": negative_attempts,
                "relaxed_negative_windows": sum(
                    stratum["count"] for stratum in relaxed_strata
                ),
                "relaxed_strata": relaxed_strata,
            },
            "output": {
                "dataset": str(dataset_path),
                "sequence_dtype": str(sequences.dtype),
                "sequence_shape": list(sequences.shape),
                "label_dtype": str(arrays["labels"].dtype),
                "arrays": sorted(arrays),
            },
        }
        with manifest_partial.open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(dataset_partial, dataset_path)
        os.replace(manifest_partial, manifest_path)
    except Exception:
        dataset_partial.unlink(missing_ok=True)
        manifest_partial.unlink(missing_ok=True)
        raise


def prepare_dataset(config_path: Union[str, Path]) -> tuple[Path, Path]:
    path = Path(config_path)
    config = load_config(path)
    for source in config["sources"].values():
        verify_file(source["path"], int(source["bytes"]), source["md5"])
    fasta = ensure_reference(
        config["sources"]["reference"], int(config["download"]["chunk_size"])
    )
    try:
        processing = config["processing"]
        canonical = set(processing["canonical_chromosomes"])
        chrom_lengths = {chrom: len(fasta[chrom]) for chrom in canonical}
        peaks = parse_peaks(
            Path(config["sources"]["peaks"]["path"]),
            config["sources"]["peaks"],
            canonical,
            chrom_lengths,
            processing,
        )
        blacklist_intervals = parse_blacklist(
            Path(config["sources"]["blacklist"]["path"]),
            config["sources"]["blacklist"],
            canonical,
            chrom_lengths,
        )
        blacklist = IntervalIndex(blacklist_intervals)
        positives, rejected = select_positive_windows(fasta, peaks, blacklist, processing)
        negative_exclusions = {
            chrom: list(blacklist_intervals[chrom]) for chrom in canonical
        }
        for peak in peaks:
            negative_exclusions[peak.chrom].append((peak.start, peak.end))
        for positive in positives:
            negative_exclusions[positive.chrom].append((positive.start, positive.end))
        negatives, attempts, relaxed_strata = sample_negative_windows(
            fasta, positives, IntervalIndex(negative_exclusions), processing
        )
        write_outputs(
            config,
            path,
            positives,
            negatives,
            fasta,
            len(peaks),
            rejected,
            attempts,
            relaxed_strata,
        )
    finally:
        fasta.close()
    return Path(config["paths"]["dataset"]), Path(config["paths"]["manifest"])


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Prepare the CTCF/GM12878 pilot dataset")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("command", choices=("download", "prepare"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.command == "download":
        downloaded = download_all(config)
        if downloaded:
            for path in downloaded:
                print(f"Downloaded and verified {path}")
        else:
            print("All source files already present and verified")
    else:
        dataset, manifest = prepare_dataset(args.config)
        print(f"Wrote {dataset}")
        print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
