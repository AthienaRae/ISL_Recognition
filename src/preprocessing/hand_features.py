"""Assign detected hands to stable slots and build raw 126-D frame features.

The two slots are named ``Left`` and ``Right`` and are emitted in that order.
MediaPipe's valid, distinct labels are the strongest evidence. Ambiguous frames
are resolved using wrist continuity. Before tracking is initialized, ambiguous
two-hand detections use *image-side* order: the smaller normalized wrist x
(left side of the stored image) enters the Left slot. This spatial fallback is
deterministic; it is not a claim about anatomical handedness.

Coordinates are deliberately not normalized here. That is a later pipeline
stage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np


LANDMARKS_PER_HAND = 21
COORDINATES_PER_LANDMARK = 3
HAND_FEATURES = LANDMARKS_PER_HAND * COORDINATES_PER_LANDMARK
FRAME_FEATURES = 2 * HAND_FEATURES
VALID_LABELS = {"Left", "Right"}
SLOTS = ("Left", "Right")


@dataclass(frozen=True)
class AssignmentResult:
    """Feature vector plus diagnostics for one assigned frame."""

    features: np.ndarray
    strategy: str
    left_detection_index: int | None
    right_detection_index: int | None
    slot_switch_anomaly: bool = False


def _landmark_array(landmarks: Sequence[Any]) -> np.ndarray:
    if len(landmarks) != LANDMARKS_PER_HAND:
        raise ValueError(
            f"Expected {LANDMARKS_PER_HAND} landmarks, got {len(landmarks)}."
        )
    try:
        values = np.asarray(
            [(point.x, point.y, point.z) for point in landmarks], dtype=np.float32
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("Every landmark must provide numeric x, y, and z values.") from exc
    if values.shape != (LANDMARKS_PER_HAND, COORDINATES_PER_LANDMARK):
        raise ValueError(f"Invalid landmark array shape: {values.shape}.")
    if not np.isfinite(values).all():
        raise ValueError("Landmarks contain NaN or Inf values.")
    return values


def _label(value: Any) -> str | None:
    """Extract a valid label from a string or MediaPipe category collection."""
    if isinstance(value, str):
        candidate = value
    elif value:
        candidate = getattr(value[0], "category_name", None)
    else:
        candidate = None
    return candidate if candidate in VALID_LABELS else None


class HandSlotAssigner:
    """Stateful Left/Right assignment for sequential video frames."""

    def __init__(self, anomaly_margin: float = 0.05) -> None:
        if anomaly_margin < 0:
            raise ValueError("anomaly_margin must be non-negative.")
        self.anomaly_margin = float(anomaly_margin)
        self._wrists: dict[str, np.ndarray | None] = {slot: None for slot in SLOTS}

    def reset(self) -> None:
        """Clear temporal state before processing a new video."""
        self._wrists = {slot: None for slot in SLOTS}

    @staticmethod
    def _distance(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.linalg.norm(first - second))

    def _two_by_continuity(self, wrists: list[np.ndarray]) -> tuple[int, int]:
        left_previous = self._wrists["Left"]
        right_previous = self._wrists["Right"]
        if left_previous is not None and right_previous is not None:
            direct = self._distance(wrists[0], left_previous) + self._distance(
                wrists[1], right_previous
            )
            crossed = self._distance(wrists[1], left_previous) + self._distance(
                wrists[0], right_previous
            )
            return (0, 1) if direct <= crossed else (1, 0)
        if left_previous is not None:
            left_index = min(
                range(2), key=lambda index: self._distance(wrists[index], left_previous)
            )
            return left_index, 1 - left_index
        if right_previous is not None:
            right_index = min(
                range(2), key=lambda index: self._distance(wrists[index], right_previous)
            )
            return 1 - right_index, right_index
        left_index = 0 if wrists[0][0] <= wrists[1][0] else 1
        return left_index, 1 - left_index

    def _one_slot(self, wrist: np.ndarray, label: str | None) -> str:
        tracked = [slot for slot in SLOTS if self._wrists[slot] is not None]
        if tracked:
            distances = {
                slot: self._distance(wrist, self._wrists[slot])  # type: ignore[arg-type]
                for slot in tracked
            }
            minimum = min(distances.values())
            tied = [
                slot
                for slot, distance in distances.items()
                if np.isclose(distance, minimum, rtol=0.0, atol=1e-7)
            ]
            if label in tied:
                return label
            return tied[0]
        if label is not None:
            return label
        return "Left" if wrist[0] < 0.5 else "Right"

    def assign(
        self,
        hand_landmarks: Sequence[Sequence[Any]],
        handedness: Sequence[Any] | None = None,
    ) -> AssignmentResult:
        """Assign up to two detections and return exactly 126 float32 values."""
        count = len(hand_landmarks)
        if count > 2:
            raise ValueError(f"Expected at most 2 hands, got {count}.")
        if handedness is None:
            handedness = [None] * count
        if len(handedness) != count:
            raise ValueError("Hand landmark and handedness counts differ.")

        arrays = [_landmark_array(points) for points in hand_landmarks]
        wrists = [points[0] for points in arrays]
        labels = [_label(value) for value in handedness]
        assigned: dict[str, int] = {}
        anomaly = False

        if count == 0:
            strategy = "no_hands"
        elif count == 2 and set(labels) == VALID_LABELS:
            assigned = {labels[index]: index for index in range(2)}  # type: ignore[misc]
            strategy = "direct_distinct_labels"
            if all(self._wrists[slot] is not None for slot in SLOTS):
                direct_cost = sum(
                    self._distance(wrists[assigned[slot]], self._wrists[slot])  # type: ignore[arg-type]
                    for slot in SLOTS
                )
                swapped_cost = sum(
                    self._distance(
                        wrists[assigned[SLOTS[1 - index]]], self._wrists[slot]  # type: ignore[arg-type]
                    )
                    for index, slot in enumerate(SLOTS)
                )
                anomaly = swapped_cost + self.anomaly_margin < direct_cost
        elif count == 2:
            had_state = any(self._wrists[slot] is not None for slot in SLOTS)
            left_index, right_index = self._two_by_continuity(wrists)
            assigned = {"Left": left_index, "Right": right_index}
            strategy = "temporal_fallback" if had_state else "spatial_initialization"
        else:
            had_state = any(self._wrists[slot] is not None for slot in SLOTS)
            slot = self._one_slot(wrists[0], labels[0])
            assigned = {slot: 0}
            strategy = "temporal_fallback" if had_state else (
                "label_initialization" if labels[0] is not None else "spatial_initialization"
            )

        output = np.zeros(FRAME_FEATURES, dtype=np.float32)
        for slot_index, slot in enumerate(SLOTS):
            detection_index = assigned.get(slot)
            start = slot_index * HAND_FEATURES
            stop = start + HAND_FEATURES
            if detection_index is not None:
                flattened = arrays[detection_index].reshape(-1)
                if flattened.shape != (HAND_FEATURES,):
                    raise AssertionError(f"{slot} slot has shape {flattened.shape}.")
                output[start:stop] = flattened
                self._wrists[slot] = wrists[detection_index].copy()
            elif not np.count_nonzero(output[start:stop]) == 0:
                raise AssertionError(f"Missing {slot} slot was not zero-filled.")

        if output.shape != (FRAME_FEATURES,) or output.dtype != np.float32:
            raise AssertionError(
                f"Expected ({FRAME_FEATURES},) float32 output, got "
                f"{output.shape} {output.dtype}."
            )
        if not np.isfinite(output).all():
            raise AssertionError("Frame features contain NaN or Inf values.")
        return AssignmentResult(
            features=output,
            strategy=strategy,
            left_detection_index=assigned.get("Left"),
            right_detection_index=assigned.get("Right"),
            slot_switch_anomaly=anomaly,
        )
