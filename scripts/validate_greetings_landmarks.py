"""Validate extracted Greetings landmarks and analyze sequence lengths."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_METADATA = PROJECT_ROOT / "data/landmarks/include_greetings_landmark_metadata.csv"
EXPECTED_ROWS = 190


def describe(values: list[int]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "min": float(array.min()), "max": float(array.max()),
        "mean": float(array.mean()), "median": float(np.median(array)),
        "std": float(array.std()),
        **{f"p{p}": float(np.percentile(array, p)) for p in (10, 25, 75, 90, 95)},
    }


def format_stats(stats: dict[str, float]) -> str:
    return ", ".join(f"{key}={value:.2f}" for key, value in stats.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()
    metadata = args.metadata.resolve()
    failures: list[str] = []
    with metadata.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_ROWS:
        failures.append(f"metadata rows: expected {EXPECTED_ROWS}, got {len(rows)}")

    sources, outputs = Counter(), Counter()
    class_counts, split_counts = Counter(), Counter()
    lengths: list[int] = []
    by_class: dict[str, list[int]] = defaultdict(list)
    by_split: dict[str, list[int]] = defaultdict(list)
    extraction_failures: list[str] = []
    zero_sequences: list[str] = []
    total_frames = zero_frames = 0
    for number, row in enumerate(rows, 2):
        source = row.get("source_video_path", "")
        sources[source.casefold()] += 1
        class_counts[row.get("label", "")] += 1
        split_counts[row.get("split", "")] += 1
        if row.get("extraction_success", "").strip().lower() != "true":
            extraction_failures.append(f"{source}: {row.get('error', '')}")
            continue
        relative = Path(row.get("landmark_file_path", ""))
        if relative.is_absolute() or ".." in relative.parts:
            failures.append(f"row {number}: unsafe landmark path")
            continue
        outputs[relative.as_posix().casefold()] += 1
        path = PROJECT_ROOT / relative
        if not path.is_file():
            failures.append(f"row {number}: missing {relative.as_posix()}")
            continue
        try:
            sequence = np.load(path, allow_pickle=False)
        except Exception as exc:
            failures.append(f"row {number}: cannot load {relative}: {exc}")
            continue
        if sequence.ndim != 2 or sequence.shape[1:] != (126,):
            failures.append(f"row {number}: invalid shape {sequence.shape}")
            continue
        if sequence.dtype != np.float32:
            failures.append(f"row {number}: invalid dtype {sequence.dtype}")
        if not np.isfinite(sequence).all():
            failures.append(f"row {number}: NaN/Inf values")
        if int(row["frame_count"]) != sequence.shape[0]:
            failures.append(f"row {number}: frame count mismatch")
        count_zero = int((~np.any(sequence, axis=1)).sum())
        if count_zero == sequence.shape[0]:
            zero_sequences.append(source)
        total_frames += sequence.shape[0]
        zero_frames += count_zero
        lengths.append(sequence.shape[0])
        by_class[row["label"]].append(sequence.shape[0])
        by_split[row["split"]].append(sequence.shape[0])

    duplicates = [source for source, count in sources.items() if count != 1]
    if len(sources) != EXPECTED_ROWS or duplicates:
        failures.append(f"unique source videos={len(sources)}, duplicates={len(duplicates)}")
    if any(count != 1 for count in outputs.values()):
        failures.append("duplicate landmark output paths")
    if extraction_failures:
        failures.append(f"extraction failures={len(extraction_failures)}")
    if zero_sequences:
        failures.append(f"completely-zero sequences={len(zero_sequences)}")

    print(f"Validation: {'PASS' if not failures else 'FAIL'}")
    print(f"Metadata rows: {len(rows)}; unique source videos: {len(sources)}")
    print("Class distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(class_counts.items())))
    print("Split distribution: " + ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items())))
    print(f"Successful sequences: {len(lengths)}; extraction failures: {len(extraction_failures)}")
    percentage = 100.0 * zero_frames / total_frames if total_frames else 0.0
    print(f"Total frames: {total_frames}; all-zero frames: {zero_frames} ({percentage:.4f}%)")
    print(f"Completely-zero sequences: {len(zero_sequences)}")
    if lengths:
        stats = describe(lengths)
        print("Overall lengths: " + format_stats(stats))
        for label, values in sorted(by_class.items()):
            print(f"Class {label}: " + format_stats(describe(values)))
        for split, values in sorted(by_split.items()):
            print(f"Split {split}: " + format_stats(describe(values)))
        recommendation = int(np.ceil(stats["p95"]))
        print(f"Recommended fixed GRU length: {recommendation} frames (ceil of p95)")
        print("Baseline handling: right-pad shorter sequences with zero frames; uniformly sample longer sequences to preserve motion across the full clip.")
    print("Problematic videos:")
    for item in extraction_failures + zero_sequences:
        print(f"  {item}")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
