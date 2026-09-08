"""PyTorch dataset for fixed-length INCLUDE Greetings landmark sequences."""
from __future__ import annotations
import csv
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_METADATA = PROJECT_ROOT / "data/landmarks/include_greetings_landmark_metadata.csv"
SEQUENCE_LENGTH = 91
FEATURE_DIMENSION = 126
VALID_SPLITS = {"train", "val", "test"}

@dataclass(frozen=True)
class Sample:
    source_video_path: str
    landmark_path: Path
    label: str
    class_id: int
    split: str

def load_metadata(metadata_path: Path = DEFAULT_METADATA) -> list[dict[str, str]]:
    path = metadata_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Landmark metadata not found: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Landmark metadata is empty.")
    sources: set[str] = set()
    for number, row in enumerate(rows, 2):
        source = row.get("source_video_path", "").strip()
        if not source or source.casefold() in sources:
            raise ValueError(f"Row {number}: missing or duplicate source: {source!r}")
        sources.add(source.casefold())
        if row.get("split", "").strip() not in VALID_SPLITS:
            raise ValueError(f"Row {number}: invalid split.")
        if row.get("extraction_success", "").strip().lower() != "true":
            raise ValueError(f"Row {number}: unsuccessful extraction: {source}")
    return rows

def build_class_mapping(rows: list[dict[str, str]]) -> dict[str, int]:
    pairs = {(int(row["class_id"]), row["label"].strip()) for row in rows}
    if len(pairs) != 9:
        raise ValueError(f"Expected 9 classes, found {len(pairs)}.")
    if len({item[0] for item in pairs}) != 9 or len({item[1] for item in pairs}) != 9:
        raise ValueError("Class IDs and labels are not one-to-one.")
    return {label: index for index, (_, label) in enumerate(sorted(pairs))}

def resize_sequence(sequence: np.ndarray) -> tuple[np.ndarray, int]:
    if sequence.ndim != 2 or sequence.shape[1] != FEATURE_DIMENSION:
        raise ValueError(f"Expected (T, {FEATURE_DIMENSION}), got {sequence.shape}.")
    if sequence.dtype != np.float32 or not np.isfinite(sequence).all():
        raise ValueError(f"Sequence must be finite float32, got {sequence.dtype}.")
    length = sequence.shape[0]
    if length <= 0:
        raise ValueError("Empty sequence.")
    if length < SEQUENCE_LENGTH:
        output = np.zeros((SEQUENCE_LENGTH, FEATURE_DIMENSION), dtype=np.float32)
        output[:length] = sequence
        return output, length
    if length > SEQUENCE_LENGTH:
        indices = np.rint(np.linspace(0, length - 1, SEQUENCE_LENGTH)).astype(np.int64)
        if len(np.unique(indices)) != SEQUENCE_LENGTH:
            raise AssertionError("Uniform sampling produced duplicate indices.")
        return sequence[indices].copy(), SEQUENCE_LENGTH
    return sequence.copy(), SEQUENCE_LENGTH

class GreetingsDataset(Dataset):
    def __init__(self, split: str, metadata_path: Path = DEFAULT_METADATA) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"Invalid split {split!r}.")
        rows = load_metadata(metadata_path)
        self.class_to_index = build_class_mapping(rows)
        self.index_to_class = {index: label for label, index in self.class_to_index.items()}
        self.samples: list[Sample] = []
        for row in rows:
            if row["split"].strip() != split:
                continue
            relative = Path(row["landmark_file_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe landmark path: {relative}")
            self.samples.append(Sample(row["source_video_path"].strip(), PROJECT_ROOT / relative,
                                       row["label"].strip(), int(row["class_id"]), split))
        if not self.samples:
            raise ValueError(f"No samples for split {split!r}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]
        if not sample.landmark_path.is_file():
            raise FileNotFoundError(f"Missing sequence: {sample.landmark_path}")
        fixed, length = resize_sequence(np.load(sample.landmark_path, allow_pickle=False))
        return (torch.from_numpy(fixed),
                torch.tensor(self.class_to_index[sample.label], dtype=torch.long),
                torch.tensor(length, dtype=torch.long))
