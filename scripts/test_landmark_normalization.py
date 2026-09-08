"""Validate raw-to-normalized hand features on representative videos."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.hand_features import HAND_FEATURES, HandSlotAssigner
from src.preprocessing.normalize_landmarks import normalize_frame

DEFAULT_MODEL = PROJECT_ROOT / "models/mediapipe/hand_landmarker.task"
DEFAULT_VIDEOS = (
    PROJECT_ROOT / "data/raw/include/extracted/Greetings/48. Hello/MVI_0029.MOV",
    PROJECT_ROOT / "data/raw/include/extracted/Greetings/49. How are you/MVI_0036.MOV",
    PROJECT_ROOT / "data/raw/include/extracted/Greetings/55. Thank you/MVI_9986.MOV",
)
TOLERANCE = 1e-6


def validate_video(video: Path, model: Path) -> tuple[int, list[str], float]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open {video}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
    )
    assigner, failures, frames, maximum = HandSlotAssigner(), [], 0, 0.0
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                ok, bgr = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int(round(frames * 1000.0 / fps)),
                )
                raw = assigner.assign(result.hand_landmarks, result.handedness).features
                normalized = normalize_frame(raw)
                if normalized.shape != (126,) or normalized.dtype != np.float32:
                    failures.append(f"frame {frames}: shape/dtype")
                if not np.isfinite(normalized).all():
                    failures.append(f"frame {frames}: non-finite")
                for start in (0, HAND_FEATURES):
                    raw_slot = raw[start:start + HAND_FEATURES]
                    slot = normalized[start:start + HAND_FEATURES]
                    if not np.any(raw_slot) and np.any(slot):
                        failures.append(f"frame {frames}: missing slot changed")
                    if np.any(slot) and not np.allclose(slot[:3], 0.0, atol=TOLERANCE):
                        failures.append(f"frame {frames}: wrist is not zero")
                    maximum = max(maximum, float(np.max(np.abs(slot))))
                    if np.any(slot) and float(np.max(np.linalg.norm(slot.reshape(21, 3), axis=1))) > 1.0 + TOLERANCE:
                        failures.append(f"frame {frames}: normalized radius exceeds 1")
                frames += 1
    finally:
        capture.release()
    if frames == 0:
        failures.append("no decoded frames")
    return frames, failures, maximum


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--videos", type=Path, nargs="+", default=list(DEFAULT_VIDEOS))
    args = parser.parse_args()
    model = args.model.resolve()
    failures: list[str] = []
    total = 0
    # Explicitly cover preservation of missing and degenerate slots.
    if np.any(normalize_frame(np.zeros(126, dtype=np.float32))):
        failures.append("synthetic zero frame was not preserved")
    for video_arg in args.videos:
        video = video_arg.resolve()
        frames, video_failures, maximum = validate_video(video, model)
        total += frames
        failures.extend(f"{video.name}: {failure}" for failure in video_failures)
        print(f"{video.relative_to(PROJECT_ROOT).as_posix()}: frames={frames}, max_abs={maximum:.6f}, failures={len(video_failures)}")
    print(f"Total frames: {total}")
    print(f"Normalization validation: {'PASS' if not failures else 'FAIL'}")
    for failure in failures:
        print(f"  {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
