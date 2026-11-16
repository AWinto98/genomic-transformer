import gzip
import hashlib
import json
import re
import threading
import time
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import pytest
import yaml

from genomic_transformer.data.prepare_ctcf import (
    download_source,
    encode_sequence,
    main,
    prepare_dataset,
    verify_file,
)


PEAK_ROWS = [
    "chr1\t25\t35\tpeak_one\t100\t.\t5.0\t6.0\t7.0\t5",
    "chr1\t60\t70\tmissing_summit\t100\t.\t5.0\t6.0\t7.0\t-1",
    "chr1\t2\t12\tout_of_bounds\t100\t.\t5.0\t6.0\t7.0\t0",
    "chr1\t100\t110\tblacklisted\t100\t.\t5.0\t6.0\t7.0\t5",
    "chr1\t170\t180\ttoo_many_ns\t100\t.\t5.0\t6.0\t7.0\t5",
    "chr1\t225\t235\tpeak_two\t100\t.\t8.0\t9.0\t10.0\t5",
    "chr2\t20\t30\tnoncanonical\t100\t.\t5.0\t6.0\t7.0\t5",
]


@contextmanager
def local_payload_server(
    payload: bytes,
    interrupt_first_at: Optional[int] = None,
    interrupt_range: Optional[str] = None,
    interrupt_range_at: int = 1,
    request_delay: float = 0,
) -> Iterator[tuple[str, list[Optional[str]], dict[str, int]]]:
    requests: list[Optional[str]] = []
    stats = {"max_active": 0}
    interrupted = False
    active = 0
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            nonlocal interrupted, active
            range_header = self.headers.get("Range")
            with lock:
                requests.append(range_header)
                active += 1
                stats["max_active"] = max(stats["max_active"], active)
            try:
                if request_delay:
                    time.sleep(request_delay)
                if interrupt_first_at is not None and not interrupted:
                    interrupted = True
                    self.send_response(200)
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(payload[:interrupt_first_at])
                    self.wfile.flush()
                    self.close_connection = True
                    return
                if range_header is not None:
                    match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header)
                    if match is None:
                        self.send_response(400)
                        self.end_headers()
                        return
                    start = int(match.group(1))
                    end = int(match.group(2)) if match.group(2) else len(payload) - 1
                    if start >= len(payload) or end >= len(payload) or end < start:
                        self.send_response(416)
                        self.send_header("Content-Range", f"bytes */{len(payload)}")
                        self.end_headers()
                        return
                    body = payload[start : end + 1]
                    self.send_response(206)
                    self.send_header("Content-Range", f"bytes {start}-{end}/{len(payload)}")
                    self.send_header("Content-Length", str(len(body)))
                    if range_header == interrupt_range and not interrupted:
                        interrupted = True
                        self.end_headers()
                        self.wfile.write(body[:interrupt_range_at])
                        self.wfile.flush()
                        self.close_connection = True
                        return
                else:
                    body = payload
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            finally:
                with lock:
                    active -= 1

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/source", requests, stats
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def write_gzip(path: Path, content: bytes) -> tuple[int, str]:
    compressed = gzip.compress(content, mtime=0)
    path.write_bytes(compressed)
    return len(compressed), hashlib.md5(compressed).hexdigest()


def make_fixture(
    tmp_path: Path,
    peak_rows: Optional[list[str]] = None,
    chromosome: Optional[str] = None,
    chromosomes: Optional[dict[str, str]] = None,
    canonical: Optional[list[str]] = None,
) -> tuple[Path, dict]:
    rows = PEAK_ROWS if peak_rows is None else peak_rows
    if chromosomes is None:
        chr1 = chromosome if chromosome is not None else "a" * 170 + "N" * 5 + "a" * 125
        chromosomes = {"chr1": chr1, "chr2": "a" * 100}
    peaks_path = tmp_path / "peaks.bed.gz"
    blacklist_path = tmp_path / "blacklist.bed.gz"
    reference_path = tmp_path / "reference.fa.gz"
    peak_size, peak_md5 = write_gzip(peaks_path, ("\n".join(rows) + "\n").encode())
    blacklist_size, blacklist_md5 = write_gzip(blacklist_path, b"chr1\t100\t101\n")
    fasta = "".join(
        f">{name}\n{sequence}\n" for name, sequence in chromosomes.items()
    ).encode()
    reference_size, reference_md5 = write_gzip(reference_path, fasta)
    config = {
        "dataset": {"name": "synthetic_ctcf", "assembly": "synthetic"},
        "download": {
            "chunk_size": 32,
            "timeout_seconds": 1,
            "max_retries": 1,
            "backoff_seconds": 0,
            "parallel": {
                "connections": 1,
                "segment_size": 16384,
                "min_file_size": 33554432,
            },
        },
        "paths": {
            "raw_dir": str(tmp_path),
            "output_dir": str(tmp_path / "output"),
            "dataset": str(tmp_path / "output" / "windows.npz"),
            "manifest": str(tmp_path / "output" / "manifest.json"),
        },
        "sources": {
            "peaks": {
                "file_accession": "synthetic_peaks",
                "format": "narrowPeak",
                "compression": "gzip",
                "url": "https://invalid.example/peaks",
                "path": str(peaks_path),
                "bytes": peak_size,
                "md5": peak_md5,
            },
            "blacklist": {
                "file_accession": "synthetic_blacklist",
                "format": "BED",
                "compression": "gzip",
                "url": "https://invalid.example/blacklist",
                "path": str(blacklist_path),
                "bytes": blacklist_size,
                "md5": blacklist_md5,
            },
            "reference": {
                "file_accession": "synthetic_reference",
                "format": "FASTA",
                "compression": "gzip",
                "url": "https://invalid.example/reference",
                "path": str(reference_path),
                "bytes": reference_size,
                "md5": reference_md5,
                "decompressed_path": str(tmp_path / "reference.fa"),
                "index_path": str(tmp_path / "reference.fa.fai"),
            },
        },
        "processing": {
            "canonical_chromosomes": canonical if canonical is not None else ["chr1"],
            "coordinate_system": "0-based half-open",
            "window_length": 20,
            "summit_offset_column": 10,
            "missing_summit": -1,
            "max_n_fraction": 0.1,
            "matching": {
                "negatives_per_positive": 1,
                "seed": 123,
                "gc_bin_edges": [0.0, 0.5, 1.0],
                "repeat_bin_edges": [0.0, 0.5, 1.0],
                "batch_size": 16,
                "max_attempts_per_negative": 100,
            },
            "encoding": {
                "alphabet": "ACGTN",
                "tokens": {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4},
                "input_case": "upper",
                "unknown_character": "N",
                "dtype": "uint8",
                "gc_denominator": "ACGT",
            },
        },
        "splits": {"validation": [], "test": []},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path, config


def overlaps(start: int, end: int, interval: tuple[int, int]) -> bool:
    return start < interval[1] and end > interval[0]


def test_prepare_centers_filters_matches_and_writes_deterministic_outputs(tmp_path: Path) -> None:
    config_path, config = make_fixture(tmp_path)

    main(["--config", str(config_path), "prepare"])

    dataset_path = Path(config["paths"]["dataset"])
    manifest_path = Path(config["paths"]["manifest"])
    first_bytes = dataset_path.read_bytes()
    first_manifest = manifest_path.read_bytes()
    with np.load(dataset_path, allow_pickle=False) as data:
        assert data["sequences"].shape == (4, 20)
        assert data["sequences"].dtype == np.uint8
        assert set(np.unique(data["sequences"])) == {0}
        assert data["labels"].tolist() == [1, 1, 0, 0]
        assert data["start"].tolist()[:2] == [20, 220]
        assert data["end"].tolist()[:2] == [40, 240]
        assert data["peak_summit"].tolist() == [30, 230, -1, -1]
        assert data["peak_index"].tolist() == [0, 1, -1, -1]
        assert data["split"].tolist() == [0, 0, 0, 0]
        assert data["peak_signal_value"].tolist()[:2] == [5.0, 8.0]
        assert data["gc_fraction"].tolist() == [0.0, 0.0, 0.0, 0.0]
        assert data["repeat_fraction"].tolist() == [1.0, 1.0, 1.0, 1.0]
        assert data["gc_bin"].tolist() == [0, 0, 0, 0]
        assert data["repeat_bin"].tolist() == [1, 1, 1, 1]
        negative_starts = data["start"].tolist()[2:]
        negative_ends = data["end"].tolist()[2:]
        assert len(set(negative_starts)) == 2

    manifest = json.loads(first_manifest)
    assert manifest["counts"]["input_canonical_peaks"] == 6
    assert manifest["counts"]["positive"] == 2
    assert manifest["counts"]["negative"] == 2
    assert manifest["counts"]["rejected_positive"] == {
        "blacklist_overlap": 1,
        "excess_n": 1,
        "missing_summit": 1,
        "out_of_bounds": 1,
    }
    assert manifest["sources"]["peaks"]["file_accession"] == "synthetic_peaks"
    assert manifest["parameters"]["window_length"] == 20

    excluded = [(25, 35), (60, 70), (2, 12), (100, 110), (170, 180), (225, 235)]
    excluded += [(20, 40), (220, 240), (100, 101)]
    for start, end in zip(negative_starts, negative_ends):
        assert not any(overlaps(start, end, interval) for interval in excluded)
    assert not overlaps(
        negative_starts[0], negative_ends[0], (negative_starts[1], negative_ends[1])
    )

    prepare_dataset(config_path)
    assert dataset_path.read_bytes() == first_bytes
    assert manifest_path.read_bytes() == first_manifest


def test_split_assignment_marks_train_validation_test_by_chromosome(tmp_path: Path) -> None:
    chromosomes = {
        "chr1": "a" * 300,
        "chr2": "a" * 300,
        "chr8": "a" * 300,
        "chr17": "a" * 300,
        "chrY": "a" * 300,
    }
    rows = [
        "chr1\t25\t35\tsplit_p1\t100\t.\t5.0\t6.0\t7.0\t5",
        "chr1\t225\t235\tsplit_p2\t100\t.\t5.0\t6.0\t7.0\t5",
        "chr2\t25\t35\tsplit_p3\t100\t.\t5.0\t6.0\t7.0\t5",
        "chr17\t25\t35\tsplit_p4\t100\t.\t5.0\t6.0\t7.0\t5",
        "chr8\t25\t35\tsplit_p5\t100\t.\t5.0\t6.0\t7.0\t5",
    ]
    config_path, config = make_fixture(
        tmp_path,
        rows,
        chromosomes=chromosomes,
        canonical=["chr1", "chr2", "chr8", "chr17", "chrY"],
    )
    config["splits"] = {"validation": ["chr17"], "test": ["chr8"]}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    main(["--config", str(config_path), "prepare"])

    with np.load(config["paths"]["dataset"], allow_pickle=False) as data:
        chroms = data["chrom"]
        split = data["split"]
        assert split.dtype == np.int8
        assert len(chroms) == 10
        expected_chroms = {0: {"chr1", "chr2"}, 1: {"chr17"}, 2: {"chr8"}}
        for value, names in expected_chroms.items():
            assert set(chroms[split == value].tolist()) == names
        assert int((split == 0).sum()) == 6
        assert int((split == 1).sum()) == 2
        assert int((split == 2).sum()) == 2
        assert data["peak_index"].tolist() == [0, 1, 2, 3, 4, -1, -1, -1, -1, -1]
        assert data["peak_index"].dtype == np.int32

    manifest = json.loads(Path(config["paths"]["manifest"]).read_text())
    assert manifest["splits"] == {
        "validation": ["chr17"],
        "test": ["chr8"],
        "counts": {"train": 6, "validation": 2, "test": 2},
    }
    output_arrays = manifest["output"]["arrays"]
    assert "split" in output_arrays
    assert "peak_index" in output_arrays
    assert "peak_name" not in output_arrays


def test_peak_index_numbers_positives_and_marks_negatives(tmp_path: Path) -> None:
    rows = [
        "chr1\t25\t35\tidx_p1\t100\t.\t5.0\t6.0\t7.0\t5",
        "chr1\t225\t235\tidx_p2\t100\t.\t5.0\t6.0\t7.0\t5",
        "chr1\t150\t160\tidx_p3\t100\t.\t5.0\t6.0\t7.0\t5",
    ]
    config_path, config = make_fixture(tmp_path, rows)

    main(["--config", str(config_path), "prepare"])

    with np.load(config["paths"]["dataset"], allow_pickle=False) as data:
        assert data["peak_index"].dtype == np.int32
        assert data["peak_index"].tolist() == [0, 1, 2, -1, -1, -1]


def test_split_assignment_rejects_overlapping_validation_and_test(tmp_path: Path) -> None:
    config_path, config = make_fixture(tmp_path)
    config["splits"] = {"validation": ["chr1"], "test": ["chr1"]}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="must not overlap"):
        prepare_dataset(config_path)

    assert not Path(config["paths"]["dataset"]).exists()


def test_split_assignment_rejects_noncanonical_chromosomes(tmp_path: Path) -> None:
    config_path, config = make_fixture(tmp_path)
    config["splits"] = {"validation": ["chrM"], "test": []}
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="canonical_chromosomes"):
        prepare_dataset(config_path)

    assert not Path(config["paths"]["dataset"]).exists()


def test_split_assignment_requires_splits_section(tmp_path: Path) -> None:
    config_path, config = make_fixture(tmp_path)
    del config["splits"]
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="missing the required 'splits' section"):
        prepare_dataset(config_path)

    assert not Path(config["paths"]["dataset"]).exists()


def test_matching_exhaustion_fails_with_stratum_counts(tmp_path: Path) -> None:
    chromosome = "a" * 20 + "G" * 20 + "a" * 260
    rows = ["chr1\t25\t35\tunique_stratum\t100\t.\t5.0\t6.0\t7.0\t5"]
    config_path, config = make_fixture(tmp_path, rows, chromosome)
    config["processing"]["matching"]["max_attempts_per_negative"] = 50
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError) as error:
        prepare_dataset(config_path)

    message = str(error.value)
    assert "required=1, selected=0, attempts=50" in message
    assert "No relaxed-bin fallback" in message
    assert not Path(config["paths"]["dataset"]).exists()


def test_relaxed_fallback_fills_from_adjacent_bin(tmp_path: Path) -> None:
    chromosome = "a" * 20 + "G" * 20 + "a" * 260
    rows = ["chr1\t25\t35\tunique_stratum\t100\t.\t5.0\t6.0\t7.0\t5"]
    config_path, config = make_fixture(tmp_path, rows, chromosome)
    matching = config["processing"]["matching"]
    matching["max_attempts_per_negative"] = 50
    matching["allow_relaxed_fallback"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    main(["--config", str(config_path), "prepare"])

    manifest = json.loads(Path(config["paths"]["manifest"]).read_text())
    assert manifest["counts"]["negative"] == 1
    assert manifest["counts"]["relaxed_negative_windows"] == 1
    assert manifest["counts"]["negative_candidates_sampled"] > 50
    assert manifest["counts"]["relaxed_strata"] == [
        {"chrom": "chr1", "gc_bin": 1, "repeat_bin": 0, "ring": 1, "count": 1}
    ]
    with np.load(config["paths"]["dataset"], allow_pickle=False) as data:
        assert data["labels"].tolist() == [1, 0]
        assert (int(data["gc_bin"][1]), int(data["repeat_bin"][1])) in {
            (0, 0),
            (0, 1),
            (1, 1),
        }


def test_relaxed_fallback_expands_to_larger_ring(tmp_path: Path) -> None:
    chromosome = "a" * 20 + "G" * 20 + "a" * 160 + "ggggggggggggAAAAAAAA" * 3 + "a" * 140
    rows = ["chr1\t25\t35\tunique_stratum\t100\t.\t5.0\t6.0\t7.0\t5"]
    config_path, config = make_fixture(tmp_path, rows, chromosome)
    matching = config["processing"]["matching"]
    matching["gc_bin_edges"] = [0.0, 0.25, 0.5, 0.75, 1.0]
    matching["repeat_bin_edges"] = [0.0, 0.25, 0.5, 0.75, 1.0]
    matching["allow_relaxed_fallback"] = True
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    main(["--config", str(config_path), "prepare"])

    manifest = json.loads(Path(config["paths"]["manifest"]).read_text())
    assert manifest["counts"]["negative"] == 1
    assert manifest["counts"]["relaxed_negative_windows"] == 1
    assert manifest["counts"]["relaxed_strata"] == [
        {"chrom": "chr1", "gc_bin": 3, "repeat_bin": 0, "ring": 2, "count": 1}
    ]
    with np.load(config["paths"]["dataset"], allow_pickle=False) as data:
        assert (int(data["gc_bin"][1]), int(data["repeat_bin"][1])) in {(1, 2), (2, 2)}


def test_relaxed_fallback_defaults_off_when_knob_absent(tmp_path: Path) -> None:
    chromosome = "a" * 20 + "G" * 20 + "a" * 260
    rows = ["chr1\t25\t35\tunique_stratum\t100\t.\t5.0\t6.0\t7.0\t5"]
    config_path, config = make_fixture(tmp_path, rows, chromosome)
    matching = config["processing"]["matching"]
    matching["max_attempts_per_negative"] = 50
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    with pytest.raises(RuntimeError, match="No relaxed-bin fallback"):
        prepare_dataset(config_path)

    assert not Path(config["paths"]["dataset"]).exists()


def test_encoding_uses_configured_uint8_tokens() -> None:
    encoding = {
        "alphabet": "ACGTN",
        "tokens": {"A": 0, "C": 1, "G": 2, "T": 3, "N": 4},
        "input_case": "upper",
        "unknown_character": "N",
        "dtype": "uint8",
    }

    encoded = encode_sequence("acgtnR", encoding)

    assert encoded.dtype == np.uint8
    assert encoded.tolist() == [0, 1, 2, 3, 4, 4]


def test_existing_download_is_revalidated_and_checksum_corruption_fails(tmp_path: Path) -> None:
    path = tmp_path / "source.bed.gz"
    content = b"synthetic"
    path.write_bytes(content)
    source = {
        "path": str(path),
        "url": "https://invalid.example/source",
        "bytes": len(content),
        "md5": hashlib.md5(content).hexdigest(),
    }

    assert download_source(source, {"chunk_size": 8, "timeout_seconds": 1}) is False
    path.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="MD5 mismatch"):
        download_source(source, {"chunk_size": 8, "timeout_seconds": 1})
    with pytest.raises(ValueError, match="MD5 mismatch"):
        verify_file(path, len(content), source["md5"])
    assert not Path(str(path) + ".partial").exists()


def download_options(
    max_retries: int = 1,
    connections: int = 4,
    segment_size: int = 16384,
    min_file_size: int = 33554432,
) -> dict:
    return {
        "chunk_size": 4096,
        "timeout_seconds": 2,
        "max_retries": max_retries,
        "backoff_seconds": 0,
        "parallel": {
            "connections": connections,
            "segment_size": segment_size,
            "min_file_size": min_file_size,
        },
    }


def test_interrupted_download_preserves_partial_and_resumes_on_rerun(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 256
    interrupted_at = 20000
    path = tmp_path / "reference.fa.gz"
    source = {
        "path": str(path),
        "url": "",
        "bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
    }

    with local_payload_server(payload, interrupt_first_at=interrupted_at) as (
        url,
        requests,
        _,
    ):
        source["url"] = url
        with pytest.raises(ValueError, match="size mismatch"):
            download_source(source, download_options(max_retries=0))
        partial = Path(str(path) + ".partial")
        assert partial.read_bytes() == payload[:interrupted_at]
        assert not path.exists()

        assert download_source(source, download_options(max_retries=2))
        assert path.read_bytes() == payload
        assert not partial.exists()
        assert requests == [None, f"bytes={interrupted_at}-"]


def test_full_size_checksum_mismatch_is_retried_without_keeping_corrupt_partial(
    tmp_path: Path,
) -> None:
    expected = b"A" * 32768
    corrupted = b"B" * 32768
    path = tmp_path / "reference.fa.gz"
    source = {
        "path": str(path),
        "url": "",
        "bytes": len(expected),
        "md5": hashlib.md5(expected).hexdigest(),
    }

    with local_payload_server(corrupted) as (url, requests, _):
        source["url"] = url
        with pytest.raises(ValueError, match="MD5 mismatch"):
            download_source(source, download_options(max_retries=1))

    assert requests == [None, None]
    assert not path.exists()
    assert not Path(str(path) + ".partial").exists()


def test_parallel_download_uses_concurrent_exact_ranges_and_assembles(tmp_path: Path) -> None:
    payload = bytes(range(256)) * 1024
    segment_size = 32768
    path = tmp_path / "reference.fa.gz"
    source = {
        "path": str(path),
        "url": "",
        "bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
    }
    expected_ranges = {
        f"bytes={start}-{min(start + segment_size, len(payload)) - 1}"
        for start in range(0, len(payload), segment_size)
    }

    with local_payload_server(payload, request_delay=0.05) as (url, requests, stats):
        source["url"] = url
        assert download_source(
            source,
            download_options(
                connections=4, segment_size=segment_size, min_file_size=1
            ),
        )

    assert path.read_bytes() == payload
    assert set(requests) == expected_ranges
    assert stats["max_active"] > 1
    assert not Path(str(path) + ".partial").exists()
    assert not Path(str(path) + ".segments").exists()
    assert not Path(str(path) + ".assemble").exists()


def test_parallel_download_resumes_prefix_segments_and_interrupted_segment(
    tmp_path: Path,
) -> None:
    payload = bytes(range(256)) * 768
    segment_size = 32768
    prefix_size = 10000
    interrupted_start = prefix_size + segment_size
    interrupted_end = interrupted_start + segment_size - 1
    interrupted_range = f"bytes={interrupted_start}-{interrupted_end}"
    path = tmp_path / "reference.fa.gz"
    partial = Path(str(path) + ".partial")
    partial.write_bytes(payload[:prefix_size])
    source = {
        "path": str(path),
        "url": "",
        "bytes": len(payload),
        "md5": hashlib.md5(payload).hexdigest(),
    }
    options = download_options(
        max_retries=0,
        connections=3,
        segment_size=segment_size,
        min_file_size=1,
    )

    with local_payload_server(
        payload,
        interrupt_range=interrupted_range,
        interrupt_range_at=5000,
        request_delay=0.01,
    ) as (url, requests, _):
        source["url"] = url
        with pytest.raises(ValueError, match="Segment bytes="):
            download_source(source, options)

        state_dir = Path(str(path) + ".segments")
        assert state_dir.is_dir()
        assert partial.read_bytes() == payload[:prefix_size]
        interrupted_partial = state_dir / (
            f"{interrupted_start:020d}-{interrupted_end:020d}.segment.partial"
        )
        interrupted_size = interrupted_partial.stat().st_size
        assert 0 < interrupted_size < segment_size
        completed = list(state_dir.glob("*.segment"))
        assert completed
        first_request_count = len(requests)

        assert download_source(source, options)
        resume_requests = requests[first_request_count:]
        assert resume_requests == [
            f"bytes={interrupted_start + interrupted_size}-{interrupted_end}"
        ]

    assert path.read_bytes() == payload
    assert not partial.exists()
    assert not Path(str(path) + ".segments").exists()
    assert not Path(str(path) + ".assemble").exists()


def test_parallel_checksum_mismatch_removes_untrusted_state(tmp_path: Path) -> None:
    expected = b"A" * 131072
    corrupted = b"B" * 131072
    path = tmp_path / "reference.fa.gz"
    source = {
        "path": str(path),
        "url": "",
        "bytes": len(expected),
        "md5": hashlib.md5(expected).hexdigest(),
    }

    with local_payload_server(corrupted) as (url, _, _):
        source["url"] = url
        with pytest.raises(ValueError, match="MD5 mismatch"):
            download_source(
                source,
                download_options(
                    max_retries=0,
                    connections=4,
                    segment_size=32768,
                    min_file_size=1,
                ),
            )

    assert not path.exists()
    assert not Path(str(path) + ".partial").exists()
    assert not Path(str(path) + ".segments").exists()
    assert not Path(str(path) + ".assemble").exists()
