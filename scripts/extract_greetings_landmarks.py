"""Extract normalized, variable-length landmarks for all Greetings videos."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import re
import sys

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.hand_features import HandSlotAssigner
from src.preprocessing.normalize_landmarks import normalize_frame

DEFAULT_CSV = PROJECT_ROOT / "data/splits/include_greetings_full_clean.csv"
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "data/raw/include/extracted"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/landmarks/include_greetings"
DEFAULT_METADATA = PROJECT_ROOT / "data/landmarks/include_greetings_landmark_metadata.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models/mediapipe/hand_landmarker.task"
EXPECTED_VIDEOS = 190
FIELDS = (
    "class_id", "label", "split", "source_video_path", "landmark_file_path",
    "frame_count", "zero_hand_frames", "one_hand_frames", "two_hand_frames",
    "extraction_success", "error",
)


def parse_bool(value: str, name: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"Row {row_number}: invalid {name}: {value!r}")
    return normalized == "true"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != EXPECTED_VIDEOS:
        raise ValueError(f"Expected {EXPECTED_VIDEOS} CSV rows, found {len(rows)}.")
    seen: set[str] = set()
    for number, row in enumerate(rows, 2):
        if row.get("parent_label", "").strip() != "Greetings":
            raise ValueError(f"Row {number}: unexpected parent_label.")
        if parse_bool(row.get("include_50", ""), "include_50", number):
            raise ValueError(f"Row {number}: INCLUDE-50 row is forbidden.")
        source = Path(row.get("video_path", ""))
        if source.is_absolute() or ".." in source.parts:
            raise ValueError(f"Row {number}: unsafe video path {source}.")
        key = source.as_posix().casefold()
        if key in seen:
            raise ValueError(f"Duplicate source video: {source.as_posix()}")
        seen.add(key)
        if not parse_bool(row.get("local_exists", ""), "local_exists", number):
            raise FileNotFoundError(f"Row {number}: source is marked missing: {source}")
    return rows


def extract_video(video: Path, model: Path) -> tuple[np.ndarray, Counter[int]]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the video.")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS: {fps}")
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    assigner = HandSlotAssigner()
    frames: list[np.ndarray] = []
    counts: Counter[int] = Counter({0: 0, 1: 0, 2: 0})
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, bgr = capture.read()
                if not success:
                    break
                index = len(frames)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(round(index * 1000.0 / fps)),
                )
                counts[len(result.hand_landmarks)] += 1
                raw = assigner.assign(result.hand_landmarks, result.handedness).features
                frames.append(normalize_frame(raw))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError("No frames decoded.")
    if reported > 0 and len(frames) != reported:
        raise RuntimeError(f"Decoded {len(frames)} frames; OpenCV reported {reported}.")
    sequence = np.stack(frames).astype(np.float32, copy=False)
    if sequence.ndim != 2 or sequence.shape[1] != 126 or not np.isfinite(sequence).all():
        raise AssertionError(f"Invalid extracted sequence: {sequence.shape} {sequence.dtype}")
    return sequence, counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    args = parser.parse_args()
    split_csv, model = args.split_csv.resolve(), args.model.resolve()
    video_root, output_root, metadata = (
        args.video_root.resolve(), args.output_root.resolve(), args.metadata.resolve()
    )
    for path, description in ((split_csv, "split CSV"), (model, "model")):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {description}: {path}")
    rows = load_rows(split_csv)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        source_relative = Path(row["video_path"])
        source = video_root / source_relative
        label = row["label"].strip()
        match = re.match(r"^(\d+)\.", label)
        if not match:
            raise ValueError(f"Invalid label: {label!r}")
        class_id = match.group(1)
        destination = output_root / class_id / f"{source_relative.stem}.npy"
        landmark_relative = destination.relative_to(PROJECT_ROOT).as_posix()
        record: dict[str, object] = {
            "class_id": class_id, "label": label, "split": row["split"].strip(),
            "source_video_path": source_relative.as_posix(),
            "landmark_file_path": landmark_relative, "frame_count": 0,
            "zero_hand_frames": 0, "one_hand_frames": 0, "two_hand_frames": 0,
            "extraction_success": False, "error": "",
        }
        try:
            if not source.is_file():
                raise FileNotFoundError(f"Source video not found: {source}")
            sequence, counts = extract_video(source, model)
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("wb") as handle:
                np.save(handle, sequence, allow_pickle=False)
            record.update(frame_count=sequence.shape[0], zero_hand_frames=counts[0],
                          one_hand_frames=counts[1], two_hand_frames=counts[2],
                          extraction_success=True)
            print(f"[{index:03d}/{len(rows)}] OK {source_relative.as_posix()} ({len(sequence)} frames)")
        except Exception as exc:  # Continue while recording every failed source.
            record["error"] = f"{type(exc).__name__}: {exc}"
            print(f"[{index:03d}/{len(rows)}] FAILED {source_relative.as_posix()}: {record['error']}")
        records.append(record)
    with metadata.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    succeeded = sum(record["extraction_success"] is True for record in records)
    print(f"Attempted={len(records)} Succeeded={succeeded} Failed={len(records)-succeeded}")
    print(f"Metadata: {metadata}")


if __name__ == "__main__":
    main()
