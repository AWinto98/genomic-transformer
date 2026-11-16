"""Synthetic-only tests for the predict inference CLI.

Tiny random-ACGT FASTAs, tiny transformers, and checkpoints written in
train.py's exact `<out-name>_model.pt` payload format ({state_dict,
model_config}). Everything runs on CPU so output is bitwise deterministic.
"""

import dataclasses
import gzip
from pathlib import Path

import numpy as np
import pytest
import torch

from genomic_transformer.models.predict import (
    main,
    read_regions,
    resolve_device,
    run_prediction,
)
from genomic_transformer.models.transformer import SmallTransformer, TransformerConfig

_WINDOW = 100


def _write_fasta(path: Path, contigs: dict[str, str]) -> Path:
    with path.open("w", encoding="utf-8") as stream:
        for chrom, sequence in contigs.items():
            stream.write(f">{chrom}\n")
            for offset in range(0, len(sequence), 60):
                stream.write(sequence[offset : offset + 60] + "\n")
    return path


def _write_gz_fasta(path: Path, contigs: dict[str, str]) -> Path:
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        for chrom, sequence in contigs.items():
            stream.write(f">{chrom}\n{sequence}\n")
    return path


def _random_contigs(rng: np.random.Generator, lengths: dict[str, int]) -> dict[str, str]:
    alphabet = np.asarray(list("ACGT"))
    return {
        chrom: "".join(rng.choice(alphabet, size=length))
        for chrom, length in lengths.items()
    }


def _write_checkpoint(path: Path, max_len: int = 128) -> Path:
    config = TransformerConfig(
        d_model=8, n_layers=1, n_heads=2, d_ff=16, dropout=0.0, max_len=max_len
    )
    model = SmallTransformer(config)
    torch.save(
        {"state_dict": model.state_dict(), "model_config": dataclasses.asdict(config)},
        path,
    )
    return path


def _write_regions(path: Path, lines: list[str]) -> Path:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _parse_bed(text: str) -> list[list[str]]:
    rows = [line.split("\t") for line in text.strip().splitlines()]
    assert all(len(row) == 6 for row in rows)
    return rows


def _fixture(tmp_path: Path, max_len: int = 128):
    rng = np.random.default_rng(0)
    fasta = _write_fasta(
        tmp_path / "synthetic.fa",
        _random_contigs(rng, {"chrA": 350, "chrB": 200, "chrC": 60}),
    )
    model = _write_checkpoint(tmp_path / "model.pt", max_len=max_len)
    return model, fasta


def test_sliding_mode_counts_schema_and_summary(tmp_path: Path, capsys) -> None:
    model, fasta = _fixture(tmp_path)

    counts = run_prediction(
        model, fasta, window=_WINDOW, stride=100, batch_size=2, device="cpu"
    )

    captured = capsys.readouterr()
    assert counts == {"scored": 6, "skipped": 1}
    rows = _parse_bed(captured.out)
    assert len(rows) == 6
    by_chrom: dict[str, list[tuple[int, int, float]]] = {}
    for chrom, start, end, name, score, strand in rows:
        start, end = int(start), int(end)
        probability = float(score)
        assert name == "."
        assert strand == "."
        assert 0.0 <= probability <= 1.0
        by_chrom.setdefault(chrom, []).append((start, end, probability))

    # chrA (350 bp) -> 4 tiles, chrB (200 bp) -> 2, chrC (60 bp) -> skipped.
    assert sorted(by_chrom) == ["chrA", "chrB"]
    assert [(s, e) for s, e, _ in by_chrom["chrA"]] == [
        (0, 100),
        (100, 200),
        (200, 300),
        (300, 350),
    ]
    assert [(s, e) for s, e, _ in by_chrom["chrB"]] == [(0, 100), (100, 200)]
    for spans in by_chrom.values():
        assert spans == sorted(spans)  # deterministic input order
        for (start, end, _), (next_start, _, _) in zip(spans, spans[1:]):
            assert start < end  # half-open
            assert next_start >= end  # non-overlapping

    assert "scored 6 windows, skipped 1" in captured.err
    assert "chrC" in captured.err  # short-contig warning names the contig


def test_regions_mode_centered_clamped_and_unknown_chrom(tmp_path: Path, capsys) -> None:
    model, fasta = _fixture(tmp_path)
    regions = _write_regions(
        tmp_path / "regions.bed",
        ["chrA\t100\t200", "chrB\t10\t20", "chrNOPE\t0\t50"],
    )

    counts = run_prediction(
        model, fasta, regions_path=regions, window=_WINDOW, device="cpu"
    )

    captured = capsys.readouterr()
    assert counts == {"scored": 2, "skipped": 1}
    rows = _parse_bed(captured.out)
    assert [(row[0], int(row[1]), int(row[2])) for row in rows] == [
        ("chrA", 100, 200),  # centered on region midpoint 150
        ("chrB", 0, 100),  # midpoint 15 clamped so the window stays in bounds
    ]
    assert "chrNOPE" in captured.err
    assert "scored 2 windows, skipped 1" in captured.err


def test_region_on_contig_shorter_than_window_is_skipped(tmp_path: Path, capsys) -> None:
    model, fasta = _fixture(tmp_path)  # chrC is 60 bp < window 100
    regions = _write_regions(tmp_path / "regions.bed", ["chrC\t0\t60"])

    counts = run_prediction(
        model, fasta, regions_path=regions, window=_WINDOW, device="cpu"
    )

    captured = capsys.readouterr()
    assert counts == {"scored": 0, "skipped": 1}
    assert captured.out == ""
    assert "chrC" in captured.err  # warning, no crash, exit path clean


def test_two_runs_are_byte_identical(tmp_path: Path, capsys) -> None:
    model, fasta = _fixture(tmp_path)

    run_prediction(model, fasta, window=_WINDOW, stride=100, batch_size=2, device="cpu")
    first = capsys.readouterr().out
    run_prediction(model, fasta, window=_WINDOW, stride=100, batch_size=2, device="cpu")
    second = capsys.readouterr().out

    assert first == second


def test_main_argv_writes_bed_file(tmp_path: Path, capsys) -> None:
    model, fasta = _fixture(tmp_path)
    out = tmp_path / "scores.bed"

    main(
        [
            "--model", str(model),
            "--fasta", str(fasta),
            "--out", str(out),
            "--window", "100",
            "--stride", "100",
            "--batch", "2",
            "--device", "cpu",
        ]
    )

    captured = capsys.readouterr()
    assert captured.out == ""
    rows = _parse_bed(out.read_text(encoding="utf-8"))
    assert len(rows) == 6
    assert "scored 6 windows, skipped 1" in captured.err


def test_window_default_comes_from_data_config(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(1)
    fasta = _write_fasta(
        tmp_path / "synthetic.fa", _random_contigs(rng, {"chrA": 2500})
    )
    model = _write_checkpoint(tmp_path / "model.pt", max_len=1000)

    counts = run_prediction(model, fasta, stride=1000, device="cpu")

    assert counts == {"scored": 3, "skipped": 0}
    rows = _parse_bed(capsys.readouterr().out)
    assert [(int(row[1]), int(row[2])) for row in rows] == [
        (0, 1000),
        (1000, 2000),
        (2000, 2500),  # last tile clipped at the contig end
    ]


def test_gz_fasta_is_decompressed_and_indexed(tmp_path: Path, capsys) -> None:
    rng = np.random.default_rng(2)
    gz_path = _write_gz_fasta(
        tmp_path / "synthetic.fa.gz", _random_contigs(rng, {"chrA": 250})
    )
    model = _write_checkpoint(tmp_path / "model.pt")

    counts = run_prediction(model, gz_path, window=_WINDOW, stride=100, device="cpu")

    assert counts == {"scored": 3, "skipped": 0}
    assert (tmp_path / "synthetic.fa").is_file()
    assert (tmp_path / "synthetic.fa.fai").is_file()
    assert len(_parse_bed(capsys.readouterr().out)) == 3


def test_missing_model_and_fasta_raise_file_not_found(tmp_path: Path) -> None:
    _, fasta = _fixture(tmp_path)
    model = _write_checkpoint(tmp_path / "model.pt")
    with pytest.raises(FileNotFoundError, match="model checkpoint"):
        run_prediction(tmp_path / "nope.pt", fasta, window=_WINDOW, device="cpu")
    with pytest.raises(FileNotFoundError, match="FASTA"):
        run_prediction(model, tmp_path / "nope.fa", window=_WINDOW, device="cpu")


def test_malformed_checkpoint_payload_raises_value_error(tmp_path: Path) -> None:
    _, fasta = _fixture(tmp_path)
    bogus = tmp_path / "bogus.pt"
    torch.save({"state_dict": {}}, bogus)

    with pytest.raises(ValueError, match="model artifact"):
        run_prediction(bogus, fasta, window=_WINDOW, device="cpu")


def test_window_exceeding_model_max_len_raises(tmp_path: Path) -> None:
    model, fasta = _fixture(tmp_path, max_len=128)

    with pytest.raises(ValueError, match="max_len"):
        run_prediction(model, fasta, window=200, device="cpu")


def test_malformed_regions_bed_raises_value_error(tmp_path: Path) -> None:
    regions = _write_regions(tmp_path / "bad.bed", ["chrA\tnot-a-number\t10"])

    with pytest.raises(ValueError, match="coordinates"):
        read_regions(regions)


def test_resolve_device_explicit_and_unknown() -> None:
    assert resolve_device("cpu").type == "cpu"
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")


def test_resolve_device_auto_prefers_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert resolve_device("auto").type == "cuda"
