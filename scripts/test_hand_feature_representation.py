"""Diagnose stable two-hand slot assignment and raw 126-D frame features."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.hand_features import FRAME_FEATURES, HAND_FEATURES, HandSlotAssigner

DEFAULT_MODEL = PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
DEFAULT_VIDEOS = (
    PROJECT_ROOT / "data/raw/include/extracted/Greetings/49. How are you/MVI_0036.MOV",
    PROJECT_ROOT / "data/raw/include/extracted/Greetings/48. Hello/MVI_9914.MOV",
    PROJECT_ROOT / "data/raw/include/extracted/Greetings/51. Good Morning/MVI_9969.MOV",
)


@dataclass
class VideoReport:
    path: Path
    frames: int
    hand_counts: Counter
    strategies: Counter
    all_zero: int
    one_slot_zero: int
    shapes_valid: bool
    finite_valid: bool
    anomalies: list[int]


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def diagnose(path: Path, model: Path) -> VideoReport:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS {fps}: {path}")
    reported = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    assigner = HandSlotAssigner()
    hand_counts: Counter = Counter({0: 0, 1: 0, 2: 0})
    strategies: Counter = Counter()
    all_zero = one_slot_zero = frames = 0
    shapes_valid = finite_valid = True
    anomalies: list[int] = []
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, bgr = capture.read()
                if not success:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(round(frames * 1000.0 / fps)),
                )
                assigned = assigner.assign(result.hand_landmarks, result.handedness)
                vector = assigned.features
                count = len(result.hand_landmarks)
                hand_counts[count] += 1
                strategies[assigned.strategy] += 1
                shapes_valid &= vector.shape == (FRAME_FEATURES,) and vector.dtype == np.float32
                finite_valid &= bool(np.isfinite(vector).all())
                left_zero = not np.any(vector[:HAND_FEATURES])
                right_zero = not np.any(vector[HAND_FEATURES:])
                all_zero += left_zero and right_zero
                one_slot_zero += left_zero != right_zero
                if assigned.slot_switch_anomaly:
                    anomalies.append(frames)
                frames += 1
    finally:
        capture.release()
    if frames == 0:
        raise RuntimeError(f"No frames decoded: {path}")
    if reported > 0 and frames != reported:
        raise RuntimeError(f"Decoded {frames} frames but OpenCV reported {reported}: {path}")
    if all_zero != hand_counts[0] or one_slot_zero != hand_counts[1]:
        raise AssertionError("Zero-filled slot counts do not match detection counts.")
    return VideoReport(path, frames, hand_counts, strategies, all_zero, one_slot_zero,
                       shapes_valid, finite_valid, anomalies)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--videos", type=Path, nargs="+", default=list(DEFAULT_VIDEOS))
    args = parser.parse_args()
    model = require_file(args.model, "Hand Landmarker model")
    reports = [diagnose(require_file(path, "Video"), model) for path in args.videos]

    print("Hand feature representation diagnostic")
    print("=" * 72)
    for report in reports:
        print(f"\n{report.path.relative_to(PROJECT_ROOT).as_posix()}")
        print(f"  frames processed: {report.frames}")
        print(f"  0/1/2-hand frames: {report.hand_counts[0]}/{report.hand_counts[1]}/{report.hand_counts[2]}")
        print(f"  direct distinct handedness: {report.strategies['direct_distinct_labels']}")
        print(f"  temporal fallback: {report.strategies['temporal_fallback']}")
        print(f"  spatial initialization fallback: {report.strategies['spatial_initialization']}")
        print(f"  all-zero 126-D frames: {report.all_zero}")
        print(f"  one-slot-zero frames: {report.one_slot_zero}")
        print(f"  shape validation: {'PASS' if report.shapes_valid else 'FAIL'}")
        print(f"  finite-value validation: {'PASS' if report.finite_valid else 'FAIL'}")
        frames = ", ".join(map(str, report.anomalies)) if report.anomalies else "none"
        print(f"  slot-switch anomalies (zero-based frames): {frames}")

    print("\nAggregate")
    print(f"  videos: {len(reports)}")
    print(f"  frames processed: {sum(r.frames for r in reports)}")
    print("  0/1/2-hand frames: " + "/".join(
        str(sum(r.hand_counts[count] for r in reports)) for count in (0, 1, 2)
    ))
    for key, label in (
        ("direct_distinct_labels", "direct distinct handedness"),
        ("temporal_fallback", "temporal fallback"),
        ("spatial_initialization", "spatial initialization fallback"),
    ):
        print(f"  {label}: {sum(r.strategies[key] for r in reports)}")
    print(f"  all-zero 126-D frames: {sum(r.all_zero for r in reports)}")
    print(f"  one-slot-zero frames: {sum(r.one_slot_zero for r in reports)}")
    print(f"  shape validation: {'PASS' if all(r.shapes_valid for r in reports) else 'FAIL'}")
    print(f"  finite-value validation: {'PASS' if all(r.finite_valid for r in reports) else 'FAIL'}")
    print(f"  slot-switch anomalies: {sum(len(r.anomalies) for r in reports)}")


if __name__ == "__main__":
    main()
