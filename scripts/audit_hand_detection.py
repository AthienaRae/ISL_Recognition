"""Audit MediaPipe hand detection and handedness on a deterministic sample.

An instability/change event is counted for a transition between consecutive
decoded frames when both frames contain the same nonzero number of detected
hands, but their sorted handedness-label multisets differ. For example,
``[Left, Right] -> [Left, Left]`` and ``[Left] -> [Right]`` each count as one
event. A change in detection count is excluded, and result ordering alone
cannot create an event.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import cv2
import mediapipe as mp


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_CSV = (
    PROJECT_ROOT / "data" / "splits" / "include_greetings_full_clean.csv"
)
DEFAULT_VIDEO_ROOT = PROJECT_ROOT / "data" / "raw" / "include" / "extracted"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "mediapipe" / "hand_landmarker.task"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "handedness_audit"
SELECTION_SEED = "include-greetings-handedness-audit-v1"
VIDEOS_PER_CLASS = 2
EXPECTED_CLASS_COUNT = 9
EXPECTED_LANDMARKS_PER_HAND = 21
MAX_HANDS = 2
CLASS_ID_PATTERN = re.compile(r"^(\d+)\.\s+.+$")


@dataclass(frozen=True)
class Sample:
    class_id: int
    label: str
    filename: str
    split: str
    relative_video_path: str
    video_path: Path


@dataclass
class AuditResult:
    class_id: int
    label: str
    filename: str
    split: str
    video_path: str
    total_decoded_frames: int
    zero_hand_frames: int
    one_hand_frames: int
    two_hand_frames: int
    frame_detection_rate_percent: float
    left_predictions: int
    right_predictions: int
    other_handedness_predictions: int
    same_handedness_conflict_frames: int
    handedness_instability_events: int
    left_confidence_min: str
    left_confidence_mean: str
    left_confidence_max: str
    right_confidence_min: str
    right_confidence_mean: str
    right_confidence_max: str
    all_hands_have_21_landmarks: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-csv", type=Path, default=DEFAULT_SPLIT_CSV)
    parser.add_argument("--video-root", type=Path, default=DEFAULT_VIDEO_ROOT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{description} not found: {resolved}")
    return resolved


def parse_bool(value: str, field_name: str, row_number: int) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(
        f"Row {row_number}: {field_name} must be True or False, got {value!r}."
    )


def selection_key(row: dict[str, str]) -> str:
    identity = "|".join(
        (SELECTION_SEED, row["label"], row["split"], row["video_path"])
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def load_samples(split_csv: Path, video_root: Path) -> list[Sample]:
    required_columns = {
        "split",
        "parent_label",
        "label",
        "video_path",
        "include_50",
        "local_exists",
    }
    rows_by_label: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_paths: set[str] = set()

    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            raise ValueError(
                f"Split CSV is missing columns: {sorted(missing_columns)}"
            )

        for row_number, row in enumerate(reader, start=2):
            if row["parent_label"].strip() != "Greetings":
                raise ValueError(
                    f"Row {row_number}: unexpected parent_label "
                    f"{row['parent_label']!r}."
                )
            if parse_bool(row["include_50"], "include_50", row_number):
                raise ValueError(
                    f"Row {row_number}: INCLUDE-50 entry found in full-INCLUDE CSV."
                )
            if not parse_bool(row["local_exists"], "local_exists", row_number):
                raise FileNotFoundError(
                    f"Row {row_number}: CSV marks video as unavailable: "
                    f"{row['video_path']}"
                )

            relative_path = Path(row["video_path"])
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    f"Row {row_number}: unsafe video_path: {row['video_path']!r}"
                )
            normalized_path = relative_path.as_posix()
            if normalized_path in seen_paths:
                raise ValueError(f"Duplicate video_path in CSV: {normalized_path}")
            seen_paths.add(normalized_path)
            rows_by_label[row["label"].strip()].append(row)

    if len(rows_by_label) != EXPECTED_CLASS_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_CLASS_COUNT} classes, found {len(rows_by_label)}."
        )

    samples: list[Sample] = []
    for label in sorted(rows_by_label, key=lambda item: int(item.split(".", 1)[0])):
        match = CLASS_ID_PATTERN.fullmatch(label)
        if not match:
            raise ValueError(f"Cannot parse class ID from label: {label!r}")
        class_id = int(match.group(1))
        ranked = sorted(rows_by_label[label], key=selection_key)
        selected = [ranked[0]]
        different_split = next(
            (row for row in ranked[1:] if row["split"] != ranked[0]["split"]),
            None,
        )
        if different_split is not None:
            selected.append(different_split)
        elif len(ranked) >= VIDEOS_PER_CLASS:
            selected.append(ranked[1])

        for row in selected[:VIDEOS_PER_CLASS]:
            relative_path = Path(row["video_path"])
            video_path = require_file(video_root / relative_path, "Sample video")
            samples.append(
                Sample(
                    class_id=class_id,
                    label=label,
                    filename=relative_path.name,
                    split=row["split"].strip(),
                    relative_video_path=relative_path.as_posix(),
                    video_path=video_path,
                )
            )

    expected_samples = EXPECTED_CLASS_COUNT * VIDEOS_PER_CLASS
    if len(samples) != expected_samples:
        raise ValueError(
            f"Expected {expected_samples} selected videos, found {len(samples)}."
        )
    return samples


def confidence_fields(scores: list[float]) -> tuple[str, str, str]:
    if not scores:
        return "", "", ""
    return (
        f"{min(scores):.6f}",
        f"{sum(scores) / len(scores):.6f}",
        f"{max(scores):.6f}",
    )


def audit_video(sample: Sample, model_path: Path) -> AuditResult:
    capture = cv2.VideoCapture(str(sample.video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {sample.video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        capture.release()
        raise RuntimeError(f"Invalid FPS {fps} for {sample.video_path}")
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_counts: Counter[int] = Counter({0: 0, 1: 0, 2: 0})
    label_counts: Counter[str] = Counter()
    confidence_scores: dict[str, list[float]] = defaultdict(list)
    same_handedness_conflicts = 0
    instability_events = 0
    previous_labels: tuple[str, ...] | None = None
    decoded_frames = 0

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
    )

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, bgr_frame = capture.read()
                if not success:
                    break

                timestamp_ms = int(round(decoded_frames * 1000.0 / fps))
                rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                result = landmarker.detect_for_video(image, timestamp_ms)
                hand_count = len(result.hand_landmarks)

                if hand_count > MAX_HANDS:
                    raise AssertionError(
                        f"{sample.filename} frame {decoded_frames}: detected "
                        f"{hand_count} hands; maximum is {MAX_HANDS}."
                    )
                if len(result.handedness) != hand_count:
                    raise AssertionError(
                        f"{sample.filename} frame {decoded_frames}: handedness "
                        "and landmark result counts differ."
                    )

                labels: list[str] = []
                for hand_index, (landmarks, categories) in enumerate(
                    zip(result.hand_landmarks, result.handedness), start=1
                ):
                    if len(landmarks) != EXPECTED_LANDMARKS_PER_HAND:
                        raise AssertionError(
                            f"{sample.filename} frame {decoded_frames}, hand "
                            f"{hand_index}: expected {EXPECTED_LANDMARKS_PER_HAND} "
                            f"landmarks, got {len(landmarks)}."
                        )
                    if not categories:
                        raise AssertionError(
                            f"{sample.filename} frame {decoded_frames}, hand "
                            f"{hand_index}: missing handedness classification."
                        )
                    best = categories[0]
                    label = best.category_name or "Unknown"
                    labels.append(label)
                    label_counts[label] += 1
                    confidence_scores[label].append(best.score)

                current_labels = tuple(sorted(labels))
                if hand_count == 2 and len(set(current_labels)) == 1:
                    same_handedness_conflicts += 1
                if (
                    previous_labels is not None
                    and len(previous_labels) == hand_count
                    and hand_count > 0
                    and previous_labels != current_labels
                ):
                    instability_events += 1

                previous_labels = current_labels
                frame_counts[hand_count] += 1
                decoded_frames += 1
    finally:
        capture.release()

    if decoded_frames == 0:
        raise RuntimeError(f"No frames decoded from {sample.video_path}")
    if reported_frames > 0 and decoded_frames != reported_frames:
        raise RuntimeError(
            f"{sample.filename}: decoded {decoded_frames} frames but OpenCV "
            f"reported {reported_frames}."
        )

    left_confidence = confidence_fields(confidence_scores["Left"])
    right_confidence = confidence_fields(confidence_scores["Right"])
    other_predictions = sum(
        count for label, count in label_counts.items() if label not in {"Left", "Right"}
    )
    detected_frames = frame_counts[1] + frame_counts[2]

    return AuditResult(
        class_id=sample.class_id,
        label=sample.label,
        filename=sample.filename,
        split=sample.split,
        video_path=sample.relative_video_path,
        total_decoded_frames=decoded_frames,
        zero_hand_frames=frame_counts[0],
        one_hand_frames=frame_counts[1],
        two_hand_frames=frame_counts[2],
        frame_detection_rate_percent=round(100.0 * detected_frames / decoded_frames, 6),
        left_predictions=label_counts["Left"],
        right_predictions=label_counts["Right"],
        other_handedness_predictions=other_predictions,
        same_handedness_conflict_frames=same_handedness_conflicts,
        handedness_instability_events=instability_events,
        left_confidence_min=left_confidence[0],
        left_confidence_mean=left_confidence[1],
        left_confidence_max=left_confidence[2],
        right_confidence_min=right_confidence[0],
        right_confidence_mean=right_confidence[1],
        right_confidence_max=right_confidence[2],
        all_hands_have_21_landmarks=True,
    )


def build_summary(samples: list[Sample], results: list[AuditResult]) -> str:
    total_frames = sum(result.total_decoded_frames for result in results)
    zero_frames = sum(result.zero_hand_frames for result in results)
    one_frames = sum(result.one_hand_frames for result in results)
    two_frames = sum(result.two_hand_frames for result in results)
    detected_frames = one_frames + two_frames
    conflicts = sum(result.same_handedness_conflict_frames for result in results)
    videos_with_conflicts = sum(
        result.same_handedness_conflict_frames > 0 for result in results
    )
    instabilities = sum(result.handedness_instability_events for result in results)

    lines = [
        "INCLUDE Greetings MediaPipe handedness audit",
        "=" * 52,
        f"Selection seed: {SELECTION_SEED}",
        f"Videos per class: {VIDEOS_PER_CLASS}",
        "Selection rule: lowest stable SHA-256 rank per class, then lowest-ranked "
        "video from a different split.",
        "Instability event: a consecutive-frame transition with equal nonzero "
        "hand counts but a changed sorted handedness-label multiset.",
        "Detection-count changes and output-order-only changes are excluded.",
        "Thresholds: MediaPipe defaults (unchanged)",
        "",
        "Selected sample",
        "-" * 52,
    ]
    lines.extend(
        f"{sample.class_id} | {sample.label} | {sample.split} | "
        f"{sample.relative_video_path}"
        for sample in samples
    )
    lines.extend(
        [
            "",
            "Aggregate results",
            "-" * 52,
            f"Total videos: {len(results)}",
            f"Total frames: {total_frames}",
            f"0-hand frames: {zero_frames}",
            f"1-hand frames: {one_frames}",
            f"2-hand frames: {two_frames}",
            f"Frame-level detection rate: {100.0 * detected_frames / total_frames:.2f}%",
            f"Same-handedness conflict frames: {conflicts}",
            f"Videos with conflicts: {videos_with_conflicts}/{len(results)} "
            f"({100.0 * videos_with_conflicts / len(results):.2f}%)",
            f"Handedness instability events: {instabilities}",
            f"All detected hands have 21 landmarks: "
            f"{all(result.all_hands_have_21_landmarks for result in results)}",
            "",
            "Class-wise summary",
            "-" * 52,
        ]
    )

    for class_id in sorted({result.class_id for result in results}):
        class_results = [result for result in results if result.class_id == class_id]
        class_frames = sum(result.total_decoded_frames for result in class_results)
        class_detected = sum(
            result.one_hand_frames + result.two_hand_frames for result in class_results
        )
        class_conflicts = sum(
            result.same_handedness_conflict_frames for result in class_results
        )
        class_instabilities = sum(
            result.handedness_instability_events for result in class_results
        )
        lines.append(
            f"{class_results[0].label}: videos={len(class_results)}, "
            f"frames={class_frames}, detection={100.0 * class_detected / class_frames:.2f}%, "
            f"conflicts={class_conflicts}, instabilities={class_instabilities}"
        )

    lines.extend(["", "Per-video results", "-" * 52])
    for result in results:
        lines.append(
            f"{result.label} | {result.filename} | {result.split} | "
            f"frames={result.total_decoded_frames}, "
            f"0/1/2={result.zero_hand_frames}/{result.one_hand_frames}/"
            f"{result.two_hand_frames}, detection={result.frame_detection_rate_percent:.2f}%, "
            f"L/R={result.left_predictions}/{result.right_predictions}, "
            f"conflicts={result.same_handedness_conflict_frames}, "
            f"instabilities={result.handedness_instability_events}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    samples: list[Sample],
    results: list[AuditResult],
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "handedness_audit.csv"
    summary_path = output_dir / "summary.txt"

    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=[field.name for field in fields(AuditResult)]
        )
        writer.writeheader()
        writer.writerows(asdict(result) for result in results)

    summary_path.write_text(build_summary(samples, results), encoding="utf-8")
    return csv_path, summary_path


def main() -> None:
    args = parse_args()
    split_csv = require_file(args.split_csv, "Clean split CSV")
    model_path = require_file(args.model, "Hand Landmarker model")
    video_root = args.video_root.expanduser().resolve()
    if not video_root.is_dir():
        raise NotADirectoryError(f"Video root not found: {video_root}")

    samples = load_samples(split_csv, video_root)
    print(f"Selected {len(samples)} videos across {EXPECTED_CLASS_COUNT} classes.")
    for index, sample in enumerate(samples, start=1):
        print(
            f"[{index:02d}/{len(samples)}] {sample.label} | {sample.split} | "
            f"{sample.relative_video_path}"
        )

    results: list[AuditResult] = []
    for index, sample in enumerate(samples, start=1):
        print(f"Auditing [{index:02d}/{len(samples)}] {sample.relative_video_path}")
        result = audit_video(sample, model_path)
        results.append(result)
        print(
            f"  frames={result.total_decoded_frames}, "
            f"0/1/2={result.zero_hand_frames}/{result.one_hand_frames}/"
            f"{result.two_hand_frames}, detection="
            f"{result.frame_detection_rate_percent:.2f}%, "
            f"conflicts={result.same_handedness_conflict_frames}, "
            f"instabilities={result.handedness_instability_events}"
        )

    csv_path, summary_path = write_outputs(
        args.output_dir.expanduser().resolve(), samples, results
    )
    print("\n" + build_summary(samples, results))
    print(f"Detailed CSV: {csv_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
