"""Wrist-relative, scale-normalized hand landmark features."""

from __future__ import annotations

import numpy as np

from src.preprocessing.hand_features import FRAME_FEATURES, HAND_FEATURES


def normalize_hand(hand_features: np.ndarray) -> np.ndarray:
    """Normalize one 63-D hand independently, preserving a missing zero slot."""
    hand = np.asarray(hand_features, dtype=np.float32)
    if hand.shape != (HAND_FEATURES,):
        raise ValueError(f"Expected ({HAND_FEATURES},) hand features, got {hand.shape}.")
    if not np.isfinite(hand).all():
        raise ValueError("Hand features contain NaN or Inf values.")
    if not np.any(hand):
        return np.zeros(HAND_FEATURES, dtype=np.float32)

    points = hand.reshape(21, 3)
    relative = points - points[0]
    scale = float(np.max(np.linalg.norm(relative, axis=1)))
    if not np.isfinite(scale) or scale <= np.finfo(np.float32).eps:
        return np.zeros(HAND_FEATURES, dtype=np.float32)
    normalized = (relative / np.float32(scale)).astype(np.float32).reshape(-1)
    if normalized.shape != (HAND_FEATURES,) or not np.isfinite(normalized).all():
        raise ValueError("Normalization produced invalid hand features.")
    return normalized


def normalize_frame(frame_features: np.ndarray) -> np.ndarray:
    """Normalize Left and Right slots and return exactly 126 float32 values."""
    frame = np.asarray(frame_features, dtype=np.float32)
    if frame.shape != (FRAME_FEATURES,):
        raise ValueError(f"Expected ({FRAME_FEATURES},) frame features, got {frame.shape}.")
    if not np.isfinite(frame).all():
        raise ValueError("Frame features contain NaN or Inf values.")
    output = np.concatenate(
        (normalize_hand(frame[:HAND_FEATURES]), normalize_hand(frame[HAND_FEATURES:]))
    ).astype(np.float32, copy=False)
    if output.shape != (FRAME_FEATURES,) or not np.isfinite(output).all():
        raise AssertionError("Normalized frame contract was violated.")
    return output
