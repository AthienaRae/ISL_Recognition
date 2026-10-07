"""Extract normalized 126-D landmarks from the static ISL alphabet dataset."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path
import sys

import cv2
import mediapipe as mp
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.hand_features import HandSlotAssigner
from src.preprocessing.normalize_landmarks import normalize_frame


DEFAULT_INPUT_ROOT = PROJECT_ROOT / "data/raw/static_alphabet"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data/landmarks/static_alphabet"
DEFAULT_METADATA = PROJECT_ROOT / "data/landmarks/static_alphabet_metadata.csv"
DEFAULT_MODEL = PROJECT_ROOT / "models/mediapipe/hand_landmarker.task"

EXPECTED_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
FIELDS = (
    "class_id",
    "label",
    "source_image_path",
    "landmark_file_path",
    "hands_detected",
    "extraction_success",
    "error",
)


def extract_image(
    image_path: Path,
    landmarker: mp.tasks.vision.HandLandmarker,
) -> tuple[np.ndarray, int]:
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        raise RuntimeError("OpenCV could not read the image.")

    height, width = bgr.shape[:2]
    if height <= 0 or width <= 0:
        raise RuntimeError(f"Invalid image dimensions: {width}x{height}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb,
    )

    result = landmarker.detect(image)
    hands_detected = len(result.hand_landmarks)

    if hands_detected == 0:
        raise RuntimeError("No hands detected.")

    # Static images are independent samples, so tracking state must not
    # carry from one image to another.
    assigner = HandSlotAssigner()

    raw = assigner.assign(
        result.hand_landmarks,
        result.handedness,
    ).features

    normalized = normalize_frame(raw)

    if normalized.shape != (126,):
        raise AssertionError(
            f"Expected shape (126,), got {normalized.shape}"
        )

    if normalized.dtype != np.float32:
        normalized = normalized.astype(np.float32, copy=False)

    if not np.isfinite(normalized).all():
        raise AssertionError("Normalized landmarks contain NaN or Inf.")

    return normalized, hands_detected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=DEFAULT_METADATA,
    )
    parser.add_argument(
    	"--max-images",
    	type=int,
    	default=None,
     )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
    )
    args = parser.parse_args()

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    metadata = args.metadata.resolve()
    model = args.model.resolve()

    if not input_root.is_dir():
        raise FileNotFoundError(f"Missing input dataset: {input_root}")

    if not model.is_file():
        raise FileNotFoundError(f"Missing MediaPipe model: {model}")

    classes = [
        path.name
        for path in sorted(input_root.iterdir())
        if path.is_dir()
    ]

    if classes != EXPECTED_CLASSES:
        raise ValueError(
            f"Unexpected class directories.\n"
            f"Expected: {EXPECTED_CLASSES}\n"
            f"Found:    {classes}"
        )

    output_root.mkdir(parents=True, exist_ok=True)
    metadata.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    counts: Counter[str] = Counter()

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model)
        ),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.2,
    )

    with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
        all_images = []

        for label in EXPECTED_CLASSES:
            class_dir = input_root / label
            images = sorted(class_dir.glob("*.jpg"))
            all_images.extend((label, image) for image in images)

        if args.max_images is not None:
            all_images = all_images[:args.max_images]

        total = len(all_images)

        for index, (label, image_path) in enumerate(all_images, 1):
            class_id = EXPECTED_CLASSES.index(label)

            destination = (
                output_root
                / label
                / f"{image_path.stem}.npy"
            )

            record: dict[str, object] = {
                "class_id": class_id,
                "label": label,
                "source_image_path": image_path.relative_to(input_root).as_posix(),
                "landmark_file_path": destination.relative_to(PROJECT_ROOT).as_posix(),
                "hands_detected": 0,
                "extraction_success": False,
                "error": "",
            }

            try:
                features, hands_detected = extract_image(
                    image_path,
                    landmarker,
                )

                destination.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                np.save(
                    destination,
                    features,
                    allow_pickle=False,
                )

                record["hands_detected"] = hands_detected
                record["extraction_success"] = True

                counts["success"] += 1
                counts[f"hands_{hands_detected}"] += 1

                print(
                    f"[{index:05d}/{total}] OK "
                    f"{label}/{image_path.name} "
                    f"({hands_detected} hands)"
                )

            except Exception as exc:
                record["error"] = f"{type(exc).__name__}: {exc}"
                counts["failed"] += 1

                print(
                    f"[{index:05d}/{total}] FAILED "
                    f"{label}/{image_path.name}: "
                    f"{record['error']}"
                )

            records.append(record)

    with metadata.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)

    print()
    print("Extraction complete.")
    print(f"Total:      {len(records)}")
    print(f"Succeeded:  {counts['success']}")
    print(f"Failed:     {counts['failed']}")
    print(f"1 hand:     {counts['hands_1']}")
    print(f"2 hands:    {counts['hands_2']}")
    print(f"Metadata:   {metadata}")
    print(f"Output:     {output_root}")


if __name__ == "__main__":
    main()
