"""Real-time desktop webcam demo for the Milestone 1 Greetings GRU."""

from __future__ import annotations

import argparse
from collections import Counter, deque
from pathlib import Path
import sys
import time

import cv2
import mediapipe as mp
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.gru_classifier import GRUClassifier
from src.preprocessing.hand_features import HandSlotAssigner
from src.preprocessing.normalize_landmarks import normalize_frame
from src.training.greetings_dataset import build_class_mapping, load_metadata

DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/gru_greetings_best.pt"
DEFAULT_LANDMARKER = PROJECT_ROOT / "models/mediapipe/hand_landmarker.task"
SEQUENCE_LENGTH = 91
FEATURE_DIMENSION = 126
CONFIDENCE_THRESHOLD = 0.50
SMOOTHING_WINDOW = 5


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def load_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    required = {"model_state_dict", "class_to_index", "model_config", "sequence_length"}
    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")
    if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError(f"Expected sequence length {SEQUENCE_LENGTH}.")
    class_to_index = checkpoint["class_to_index"]
    if class_to_index != build_class_mapping(load_metadata()):
        raise ValueError("Checkpoint class mapping differs from training metadata.")
    if sorted(class_to_index.values()) != list(range(9)):
        raise ValueError("Checkpoint class indices must be contiguous 0..8.")
    model = GRUClassifier(**checkpoint["model_config"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, class_to_index


def create_landmarker(model_path: Path):
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def infer(model, frames: deque[np.ndarray], device: torch.device):
    sequence = np.stack(frames).astype(np.float32, copy=False)
    if sequence.shape != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        raise AssertionError(f"Invalid live model input: {sequence.shape}")
    features = torch.from_numpy(sequence).unsqueeze(0).to(device)
    lengths = torch.tensor([SEQUENCE_LENGTH], dtype=torch.long)
    started = time.perf_counter()
    with torch.no_grad():
        probabilities = torch.softmax(model(features, lengths), dim=1)[0]
    latency_ms = (time.perf_counter() - started) * 1000.0
    confidence, index = torch.max(probabilities, dim=0)
    return int(index.item()), float(confidence.item()), latency_ms


def majority_vote(predictions: deque[int]) -> int:
    counts = Counter(predictions)
    # Prefer the most recent prediction when vote counts tie.
    return max(counts, key=lambda item: (counts[item], list(predictions)[::-1].index(item) * -1))


def draw_status(frame, label: str, confidence: float, buffer_size: int,
                hand_count: int, fps: float, inference_ms: float) -> None:
    lines = [
        f"Prediction: {label}", f"Confidence: {confidence:.1%}",
        f"Buffer: {buffer_size}/{SEQUENCE_LENGTH}", f"Hands: {hand_count}",
        f"FPS: {fps:.1f}  GRU: {inference_ms:.1f} ms", "q: quit   r: reset",
    ]
    for row, text in enumerate(lines):
        cv2.putText(frame, text, (16, 32 + row * 29), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (40, 230, 40), 2, cv2.LINE_AA)


def validate_only(model, landmarker_path: Path, device: torch.device) -> None:
    with create_landmarker(landmarker_path):
        print("MediaPipe Hand Landmarker loaded")
    frames = deque(maxlen=SEQUENCE_LENGTH)
    for _ in range(SEQUENCE_LENGTH):
        frames.append(normalize_frame(np.zeros(FEATURE_DIMENSION, dtype=np.float32)))
    index, confidence, latency = infer(model, frames, device)
    print(f"91-frame buffer validation: PASS ({len(frames)}/{SEQUENCE_LENGTH})")
    print(f"Model inference validation: PASS (class_index={index}, confidence={confidence:.4f}, latency={latency:.2f} ms)")


def run_camera(args, model, index_to_class, landmarker_path: Path,
               device: torch.device) -> None:
    capture = cv2.VideoCapture(args.camera_index)
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open webcam index {args.camera_index}.")
    print(f"Webcam opened: index {args.camera_index}")
    buffer: deque[np.ndarray] = deque(maxlen=SEQUENCE_LENGTH)
    recent_predictions: deque[int] = deque(maxlen=SMOOTHING_WINDOW)
    assigner = HandSlotAssigner()
    displayed_label, confidence, inference_ms = "Collecting frames", 0.0, 0.0
    last_timestamp_ms = -1
    previous_time = time.perf_counter()
    fps = 0.0
    processed = 0
    hand_counts: Counter[int] = Counter({0: 0, 1: 0, 2: 0})
    inference_count = 0
    try:
        with create_landmarker(landmarker_path) as landmarker:
            print("MediaPipe Hand Landmarker loaded")
            while True:
                success, frame = capture.read()
                if not success:
                    raise RuntimeError("Webcam frame capture failed.")
                now = time.perf_counter()
                instant_fps = 1.0 / max(now - previous_time, 1e-9)
                fps = instant_fps if fps == 0.0 else 0.9 * fps + 0.1 * instant_fps
                previous_time = now
                timestamp_ms = max(last_timestamp_ms + 1, time.monotonic_ns() // 1_000_000)
                last_timestamp_ms = timestamp_ms
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = landmarker.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), timestamp_ms
                )
                hand_count = len(result.hand_landmarks)
                if hand_count > 2 or any(len(hand) != 21 for hand in result.hand_landmarks):
                    raise RuntimeError("MediaPipe returned an invalid hand landmark count.")
                hand_counts[hand_count] += 1
                raw = assigner.assign(result.hand_landmarks, result.handedness).features
                buffer.append(normalize_frame(raw))
                if len(buffer) == SEQUENCE_LENGTH:
                    predicted, confidence, inference_ms = infer(model, buffer, device)
                    inference_count += 1
                    recent_predictions.append(predicted)
                    voted = majority_vote(recent_predictions)
                    displayed_label = (index_to_class[voted] if confidence >= CONFIDENCE_THRESHOLD
                                       else "Uncertain")
                if not args.headless:
                    draw_status(frame, displayed_label, confidence, len(buffer), hand_count,
                                fps, inference_ms)
                    cv2.imshow("ISL Greetings Demo", frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
                    if key == ord("r"):
                        buffer.clear(); recent_predictions.clear(); assigner.reset()
                        displayed_label, confidence = "Collecting frames", 0.0
                        print("Temporal buffer reset")
                processed += 1
                if args.max_frames and processed >= args.max_frames:
                    break
    finally:
        capture.release()
        if not args.headless:
            cv2.destroyAllWindows()
    print(
        f"Camera frames processed: {processed}; 0/1/2-hand frames: "
        f"{hand_counts[0]}/{hand_counts[1]}/{hand_counts[2]}"
    )
    print(
        f"Model inferences: {inference_count}; final prediction: {displayed_label}; "
        f"confidence: {confidence:.4f}; final FPS: {fps:.2f}; "
        f"final GRU latency: {inference_ms:.2f} ms"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--landmarker", type=Path, default=DEFAULT_LANDMARKER)
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--headless", action="store_true", help="Do not create an OpenCV window.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames; 0 runs until q.")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_path = require_file(args.checkpoint, "GRU checkpoint")
    landmarker_path = require_file(args.landmarker, "MediaPipe model")
    model, class_to_index = load_model(checkpoint_path, device)
    index_to_class = {index: label for label, index in class_to_index.items()}
    print(f"Device: {device}")
    print(f"Checkpoint loaded: {checkpoint_path}")
    print(f"Class mapping: {class_to_index}")
    if args.validate_only:
        validate_only(model, landmarker_path, device)
        return
    run_camera(args, model, index_to_class, landmarker_path, device)


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
