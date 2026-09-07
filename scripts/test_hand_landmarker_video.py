"""Validate MediaPipe Hand Landmarker on the first INCLUDE pilot video."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "include"
    / "extracted"
    / "Greetings"
    / "48. Hello"
    / "MVI_0029.MOV"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
)
EXPECTED_LANDMARKS_PER_HAND = 21
MAX_HANDS = 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the MediaPipe Tasks Hand Landmarker on one video."
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Input video path (default: the verified Hello pilot video).",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="MediaPipe Hand Landmarker .task model path.",
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def main() -> None:
    args = parse_args()
    video_path = require_file(args.video, "Video")
    model_path = require_file(args.model, "Hand Landmarker model")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Video reports an invalid FPS value: {fps}")

    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_hand_counts: Counter[int] = Counter({0: 0, 1: 0, 2: 0})
    handedness_counts: Counter[str] = Counter()
    handedness_scores: dict[str, list[float]] = {}
    decoded_frames = 0

    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )

    print("=" * 78)
    print("MediaPipe Tasks Hand Landmarker video test")
    print(f"Video       : {video_path}")
    print(f"Model       : {model_path}")
    print(f"Running mode: VIDEO")
    print(f"Max hands   : {MAX_HANDS}")
    print(f"FPS         : {fps:.2f}")
    print(f"Frames      : {reported_frames} reported")
    print("=" * 78)

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, bgr_frame = capture.read()
                if not success:
                    break

                frame_index = decoded_frames
                timestamp_ms = int(round(frame_index * 1000.0 / fps))
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(mp_image, timestamp_ms)

                hand_count = len(result.hand_landmarks)
                if hand_count > MAX_HANDS:
                    raise AssertionError(
                        f"Frame {frame_index}: detected {hand_count} hands; "
                        f"configured maximum is {MAX_HANDS}."
                    )
                if len(result.handedness) != hand_count:
                    raise AssertionError(
                        f"Frame {frame_index}: handedness result count "
                        f"({len(result.handedness)}) does not match hand count "
                        f"({hand_count})."
                    )

                descriptions: list[str] = []
                for hand_index, (landmarks, categories) in enumerate(
                    zip(result.hand_landmarks, result.handedness), start=1
                ):
                    landmark_count = len(landmarks)
                    if landmark_count != EXPECTED_LANDMARKS_PER_HAND:
                        raise AssertionError(
                            f"Frame {frame_index}, hand {hand_index}: expected "
                            f"{EXPECTED_LANDMARKS_PER_HAND} landmarks, got "
                            f"{landmark_count}."
                        )
                    if not categories:
                        raise AssertionError(
                            f"Frame {frame_index}, hand {hand_index}: "
                            "missing handedness classification."
                        )

                    best = categories[0]
                    label = best.category_name or "Unknown"
                    handedness_counts[label] += 1
                    handedness_scores.setdefault(label, []).append(best.score)
                    descriptions.append(
                        f"hand {hand_index}: {label} "
                        f"confidence={best.score:.4f}, landmarks={landmark_count}"
                    )

                frame_hand_counts[hand_count] += 1
                detail = "; ".join(descriptions) if descriptions else "no hands"
                print(
                    f"Frame {frame_index:03d} | {timestamp_ms:4d} ms | "
                    f"hands={hand_count} | {detail}"
                )
                decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames == 0:
        raise RuntimeError("No video frames were decoded.")
    if reported_frames > 0 and decoded_frames != reported_frames:
        raise RuntimeError(
            f"Decoded {decoded_frames} frames, but OpenCV reported "
            f"{reported_frames}."
        )

    print("=" * 78)
    print("Summary")
    print(f"Decoded frames          : {decoded_frames}")
    print(f"Frames with 0 hands     : {frame_hand_counts[0]}")
    print(f"Frames with 1 hand      : {frame_hand_counts[1]}")
    print(f"Frames with 2 hands     : {frame_hand_counts[2]}")
    print("Handedness detections   :")
    if handedness_counts:
        for label in sorted(handedness_counts):
            scores = handedness_scores[label]
            mean_score = sum(scores) / len(scores)
            print(
                f"  {label}: count={handedness_counts[label]}, "
                f"confidence min={min(scores):.4f}, "
                f"mean={mean_score:.4f}, max={max(scores):.4f}"
            )
    else:
        print("  none")
    print(
        "Landmark validation     : PASS "
        f"({EXPECTED_LANDMARKS_PER_HAND} per detected hand)"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()
