"""Train and validate the first length-aware INCLUDE Greetings GRU baseline."""

from __future__ import annotations

import argparse
import copy
import csv
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from src.models.gru_classifier import GRUClassifier
from src.training.greetings_dataset import (
    DEFAULT_METADATA, FEATURE_DIMENSION, SEQUENCE_LENGTH, GreetingsDataset,
    load_metadata,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints/gru_greetings_best.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "results/gru_baseline"
EXPECTED_COUNTS = {"train": 132, "val": 17, "test": 41}


def set_determinism(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False


def verify_splits(metadata: Path) -> dict[str, int]:
    rows = load_metadata(metadata)
    counts = Counter(row["split"].strip() for row in rows)
    actual = {split: counts[split] for split in EXPECTED_COUNTS}
    if actual != EXPECTED_COUNTS:
        raise ValueError(f"Expected split counts {EXPECTED_COUNTS}, got {actual}.")
    sources = {split: set() for split in EXPECTED_COUNTS}
    for row in rows:
        split = row["split"].strip()
        source = row["source_video_path"].strip().casefold()
        sources[split].add(source)
    for first, second in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sources[first] & sources[second]
        if overlap:
            raise ValueError(f"Source overlap between {first} and {second}: {sorted(overlap)}")
    return actual


def run_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
    optimizer: Adam | None = None,
) -> tuple[float, float, list[int], list[int]]:
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = count = 0
    targets: list[int] = []
    predictions: list[int] = []
    for features, labels, lengths in loader:
        features, labels = features.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(features, lengths)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        batch = labels.shape[0]
        predicted = logits.argmax(dim=1)
        loss_sum += float(loss.item()) * batch
        correct += int((predicted == labels).sum().item())
        count += batch
        targets.extend(labels.detach().cpu().tolist())
        predictions.extend(predicted.detach().cpu().tolist())
    return loss_sum / count, correct / count, targets, predictions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) <= 0:
        raise ValueError("epochs, patience, and batch-size must be positive.")

    set_determinism(args.seed)
    metadata = args.metadata.resolve()
    counts = verify_splits(metadata)  # Test metadata is counted only; no test Dataset is built.
    train_data = GreetingsDataset("train", metadata)
    val_data = GreetingsDataset("val", metadata)
    if len(train_data) != counts["train"] or len(val_data) != counts["val"]:
        raise AssertionError("Dataset sizes disagree with verified metadata counts.")
    sample_features, _, sample_length = train_data[0]
    if sample_features.shape != (SEQUENCE_LENGTH, FEATURE_DIMENSION):
        raise AssertionError(f"Fixed input shape is invalid: {sample_features.shape}")
    if not 1 <= int(sample_length) <= SEQUENCE_LENGTH:
        raise AssertionError("Effective sequence length is invalid.")
    print(f"Split counts: train={counts['train']} val={counts['val']} test={counts['test']}")
    print("Source overlap: none")
    print(f"Fixed feature shape: ({SEQUENCE_LENGTH}, {FEATURE_DIMENSION})")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Device: {device}; parameters: {parameter_count}")

    history: list[dict[str, float | int]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, float | int] = {}
    best_loss = float("inf")
    stale_epochs = 0
    start_time = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy, _, _ = run_epoch(
            model, train_loader, criterion, device, optimizer
        )
        val_loss, val_accuracy, targets, predictions = run_epoch(
            model, val_loader, criterion, device
        )
        val_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
        record = {
            "epoch": epoch, "train_loss": train_loss, "train_accuracy": train_accuracy,
            "validation_loss": val_loss, "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_f1,
        }
        history.append(record)
        print(
            f"Epoch {epoch:02d}: train_loss={train_loss:.6f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.6f} val_acc={val_accuracy:.4f} val_f1={val_f1:.4f}"
        )
        if val_loss < best_loss:
            best_loss = val_loss
            stale_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
            best_metrics = record.copy()
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping after epoch {epoch}.")
                break
    training_seconds = time.perf_counter() - start_time
    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint.")

    model.load_state_dict(best_state)
    val_loss, val_accuracy, targets, predictions = run_epoch(model, val_loader, criterion, device)
    val_f1 = f1_score(targets, predictions, average="macro", zero_division=0)
    labels = list(range(len(train_data.class_to_index)))
    class_names = [train_data.index_to_class[index] for index in labels]
    matrix = confusion_matrix(targets, predictions, labels=labels)
    report = classification_report(
        targets, predictions, labels=labels, target_names=class_names,
        output_dict=True, zero_division=0,
    )

    checkpoint = args.checkpoint.resolve()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": best_state, "class_to_index": train_data.class_to_index,
        "model_config": {"input_size": 126, "hidden_size": 128, "num_layers": 1,
                         "num_classes": 9, "dropout": 0.3},
        "sequence_length": SEQUENCE_LENGTH, "best_epoch": int(best_metrics["epoch"]),
        "best_metrics": best_metrics, "seed": args.seed,
    }, checkpoint)
    results_dir = args.results_dir.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)
    history_path = results_dir / "training_history.csv"
    with history_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary_lines = [
        "INCLUDE Greetings GRU baseline validation", "=" * 44,
        f"Device: {device}", f"Train/validation/test counts: {counts['train']}/{counts['val']}/{counts['test']}",
        "Test split evaluated: No", f"Epochs trained: {len(history)}",
        f"Best epoch: {int(best_metrics['epoch'])}",
        f"Best train loss: {float(best_metrics['train_loss']):.6f}",
        f"Best train accuracy: {float(best_metrics['train_accuracy']):.6f}",
        f"Best validation loss: {val_loss:.6f}", f"Best validation accuracy: {val_accuracy:.6f}",
        f"Best validation macro-F1: {val_f1:.6f}", f"Training time seconds: {training_seconds:.3f}",
        f"Parameter count: {parameter_count}", f"Checkpoint bytes: {checkpoint.stat().st_size}",
        "", "Per-class validation metrics (precision, recall, F1, support)",
    ]
    for name in class_names:
        values = report[name]
        summary_lines.append(
            f"{name}: {values['precision']:.6f}, {values['recall']:.6f}, "
            f"{values['f1-score']:.6f}, {int(values['support'])}"
        )
    summary_lines.extend(["", "Validation confusion matrix (rows=true, columns=predicted)",
                          "Class order: " + " | ".join(class_names)])
    summary_lines.extend(" ".join(map(str, row)) for row in matrix.tolist())
    summary_path = results_dir / "training_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(summary_lines))
    print(f"Checkpoint: {checkpoint}")
    print(f"History: {history_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
