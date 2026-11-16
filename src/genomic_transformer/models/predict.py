"""Inference CLI for the small transformer window classifier.

Scores FASTA sequence windows with a checkpoint produced by train.py
(`<out-name>_model.pt`, payload: `state_dict` + `model_config`) and writes
BED6: chrom, window start, window end, name `.`, sigmoid probability with 5
decimals, strand `.`.

Two windowing modes:
- --regions <bed>: one window per BED3 region, centered on the region and
  clamped so the full window stays inside the contig.
- default: sliding windows across every contig at the given --stride
  (stride == window gives non-overlapping tiles; the last tile of a contig
  is clipped at the contig end).

Regions on unknown contigs and regions/windows on contigs shorter than the
window are skipped with a stderr warning; everything else is fail-loud.
The token map and default window length come from the data config
(processing.encoding.tokens / processing.window_length), matching the
encoding prepare_ctcf.py used at training time.
"""

import argparse
import gzip
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator, Optional, TextIO, Union

import numpy as np
import pyfaidx
import torch

from genomic_transformer.config import load_config
from genomic_transformer.data.prepare_ctcf import encode_sequence
from genomic_transformer.models.transformer import (
    PAD_TOKEN,
    SmallTransformer,
    TransformerConfig,
)

_DEFAULT_DATA_CONFIG = Path("config/data/ctcf_gm12878.yaml")

BedRecord = tuple[str, int, int]  # chrom, start, end, 0-based half-open
WindowEvent = tuple[Any, ...]  # ("window", chrom, start, end) or ("skip", reason)


def resolve_device(choice: str) -> torch.device:
    """Map --device auto|cuda|mps|cpu onto a torch.device.

    auto prefers CUDA, then MPS, then CPU (inference targets the cloud GPUs
    first, unlike training which is Mac-first); explicit choices raise when
    the backend is unavailable.
    """
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise ValueError("--device cuda requested but CUDA is not available")
        return torch.device("cuda")
    if choice == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError("--device mps requested but MPS is not available")
        return torch.device("mps")
    if choice == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unknown device {choice!r}; expected auto|cuda|mps|cpu")


def load_model(
    model_path: Union[str, Path], device: torch.device
) -> tuple[SmallTransformer, TransformerConfig]:
    """Load a train.py `<out-name>_model.pt` artifact onto the device, in eval mode."""
    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"model checkpoint not found: {model_path}")
    payload = torch.load(model_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not {"state_dict", "model_config"} <= set(payload):
        raise ValueError(
            f"checkpoint at {model_path} is not a train.py model artifact "
            f"(expected a dict with 'state_dict' and 'model_config')"
        )
    config = TransformerConfig.from_dict(payload["model_config"])
    model = SmallTransformer(config)
    try:
        model.load_state_dict(payload["state_dict"])
    except RuntimeError as exc:
        raise ValueError(f"checkpoint at {model_path} does not match model shape: {exc}") from exc
    return model.to(device).eval(), config


def ensure_fasta(fasta_path: Union[str, Path]) -> Path:
    """Resolve any .fa[.gz] path to an indexed plain FASTA, building the .fai if missing.

    A .gz input is decompressed to its sibling (x.fa.gz -> x.fa) with the same
    .partial-then-replace pattern prepare_ctcf.ensure_reference uses.
    """
    fasta_path = Path(fasta_path)
    if not fasta_path.is_file():
        raise FileNotFoundError(f"FASTA not found: {fasta_path}")
    if str(fasta_path).endswith(".gz"):
        decompressed = fasta_path.with_suffix("")  # x.fa.gz -> x.fa
        if not decompressed.exists():
            partial = decompressed.with_name(decompressed.name + ".partial")
            partial.unlink(missing_ok=True)
            try:
                with gzip.open(fasta_path, "rb") as input_stream, partial.open("xb") as output_stream:
                    shutil.copyfileobj(input_stream, output_stream)
                os.replace(partial, decompressed)
            except Exception:
                partial.unlink(missing_ok=True)
                raise
        fasta_path = decompressed
    index_path = Path(str(fasta_path) + ".fai")
    if not index_path.exists():
        indexed = pyfaidx.Fasta(str(fasta_path))
        indexed.close()
    return fasta_path


def read_regions(regions_path: Union[str, Path]) -> list[BedRecord]:
    """Parse BED3 (chrom start end); blank lines and '#' comments are ignored."""
    regions_path = Path(regions_path)
    if not regions_path.is_file():
        raise FileNotFoundError(f"regions BED not found: {regions_path}")
    regions = []
    for line_number, raw in enumerate(
        regions_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t") if "\t" in line else line.split()
        if len(fields) < 3:
            raise ValueError(
                f"{regions_path}:{line_number}: expected at least 3 BED columns, "
                f"got {len(fields)}: {line!r}"
            )
        chrom, start_text, end_text = fields[:3]
        try:
            start, end = int(start_text), int(end_text)
        except ValueError:
            raise ValueError(
                f"{regions_path}:{line_number}: non-integer BED coordinates: {line!r}"
            ) from None
        if start < 0 or end < start:
            raise ValueError(
                f"{regions_path}:{line_number}: invalid half-open interval "
                f"{start}-{end} for {chrom}"
            )
        regions.append((chrom, start, end))
    return regions


def _iter_windows(
    fasta: pyfaidx.Fasta,
    regions: Optional[list[BedRecord]],
    window: int,
    stride: int,
) -> Iterator[WindowEvent]:
    """Yield ("window", chrom, start, end) in deterministic order, ("skip", reason) otherwise."""
    if regions is not None:
        for chrom, region_start, region_end in regions:
            if chrom not in fasta:
                yield (
                    "skip",
                    f"region {chrom}:{region_start}-{region_end}: unknown contig {chrom!r}",
                )
                continue
            length = len(fasta[chrom])
            if length < window:
                yield (
                    "skip",
                    f"region {chrom}:{region_start}-{region_end}: contig {chrom} "
                    f"length {length} shorter than window {window}",
                )
                continue
            center = (region_start + region_end) // 2
            start = min(max(center - window // 2, 0), length - window)
            yield ("window", chrom, start, start + window)
    else:
        for chrom in fasta.keys():
            length = len(fasta[chrom])
            if length < window:
                yield (
                    "skip",
                    f"contig {chrom}: length {length} shorter than window {window}",
                )
                continue
            for start in range(0, length, stride):
                yield ("window", chrom, start, min(start + window, length))


def _fetch(fasta: pyfaidx.Fasta, chrom: str, start: int, end: int) -> str:
    sequence = str(fasta[chrom][start:end])
    if len(sequence) != end - start:
        raise ValueError(f"Could not read complete window {chrom}:{start}-{end}")
    return sequence


def run_prediction(
    model_path: Union[str, Path],
    fasta_path: Union[str, Path],
    regions_path: Optional[Union[str, Path]] = None,
    out_path: Optional[Union[str, Path]] = None,
    window: Optional[int] = None,
    stride: int = 1000,
    batch_size: int = 64,
    device: str = "auto",
    data_config_path: Union[str, Path] = _DEFAULT_DATA_CONFIG,
) -> dict[str, int]:
    """Score windows and write BED6; returns {"scored": n, "skipped": m}.

    BED rows go to out_path (or stdout when None); warnings and the final
    summary go to stderr.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    if batch_size < 1:
        raise ValueError(f"batch size must be >= 1, got {batch_size}")

    data_config = load_config(Path(data_config_path))
    encoding = dict(data_config["processing"]["encoding"])
    if window is None:
        window = int(data_config["processing"]["window_length"])
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")

    device = resolve_device(device)
    model, model_config = load_model(model_path, device)
    if window > model_config.max_len:
        raise ValueError(
            f"window {window} exceeds model max_len {model_config.max_len}; "
            f"the checkpoint cannot score it"
        )

    fasta_file = ensure_fasta(fasta_path)
    fasta = pyfaidx.Fasta(str(fasta_file))
    regions = read_regions(regions_path) if regions_path is not None else None

    scored = 0
    skipped = 0
    output_stream: TextIO
    close_output = False
    if out_path is not None:
        output_stream = Path(out_path).open("w", encoding="utf-8")
        close_output = True
    else:
        output_stream = sys.stdout

    pending: list[tuple[str, int, int, str]] = []

    def flush() -> None:
        nonlocal scored
        if not pending:
            return
        # Tail tiles clipped at contig ends are shorter than `window`; pad each
        # batch to its own max length with N, which doubles as the pad token.
        width = max(len(sequence) for _, _, _, sequence in pending)
        tokens = np.full((len(pending), width), PAD_TOKEN, dtype=np.uint8)
        for index, (_, _, _, sequence) in enumerate(pending):
            tokens[index, : len(sequence)] = encode_sequence(sequence, encoding)
        input_ids = torch.from_numpy(tokens).long().to(device)
        with torch.no_grad():
            probabilities = torch.sigmoid(model(input_ids)).float().cpu().numpy()
        for (chrom, start, end, _), probability in zip(pending, probabilities):
            output_stream.write(f"{chrom}\t{start}\t{end}\t.\t{probability:.5f}\t.\n")
        scored += len(pending)
        pending.clear()

    try:
        for event in _iter_windows(fasta, regions, window, stride):
            if event[0] == "skip":
                print(f"warning: skipping {event[1]}", file=sys.stderr)
                skipped += 1
                continue
            _, chrom, start, end = event
            pending.append((chrom, start, end, _fetch(fasta, chrom, start, end)))
            if len(pending) >= batch_size:
                flush()
        flush()
        output_stream.flush()
    finally:
        fasta.close()
        if close_output:
            output_stream.close()

    print(f"scored {scored} windows, skipped {skipped}", file=sys.stderr)
    return {"scored": scored, "skipped": skipped}


def main(argv: Optional[list[str]] = None) -> None:
    parser = argparse.ArgumentParser(
        description="Score FASTA windows with a trained small-transformer checkpoint"
    )
    parser.add_argument("--model", required=True, type=Path, help="train.py <out-name>_model.pt")
    parser.add_argument("--fasta", required=True, type=Path, help="reference FASTA (.fa or .fa.gz)")
    parser.add_argument("--regions", default=None, type=Path, help="BED3 regions to center windows on")
    parser.add_argument("--out", default=None, type=Path, help="output BED (default: stdout)")
    parser.add_argument(
        "--window",
        type=int,
        default=None,
        help="window length (default: data config processing.window_length)",
    )
    parser.add_argument("--stride", type=int, default=1000, help="sliding-window stride")
    parser.add_argument("--batch", type=int, default=64, help="scoring batch size")
    parser.add_argument(
        "--device", default="auto", choices=["auto", "cuda", "mps", "cpu"]
    )
    args = parser.parse_args(argv)
    run_prediction(
        args.model,
        args.fasta,
        regions_path=args.regions,
        out_path=args.out,
        window=args.window,
        stride=args.stride,
        batch_size=args.batch,
        device=args.device,
    )


if __name__ == "__main__":
    main()
