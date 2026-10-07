"""Train a 36-class MLP on normalized static ISL landmarks."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "landmarks" / "static_alphabet"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "mlp_static_best.pt"
SPLIT_PATH = PROJECT_ROOT / "data" / "landmarks" / "static_alphabet_splits.csv"

SEED = 42
NUM_CLASSES = 36
INPUT_DIM = 126
BATCH_SIZE = 128
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 12


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class StaticMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(INPUT_DIM, 128),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(64, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def load_dataset():
    features = []
    labels = []
    paths = []

    for class_id, label in enumerate("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        class_dir = DATA_ROOT / label

        for path in sorted(class_dir.glob("*.npy")):
            x = np.load(path, allow_pickle=False)

            if x.shape != (INPUT_DIM,):
                raise ValueError(
                    f"Unexpected shape {x.shape} in {path}"
                )

            features.append(x.astype(np.float32))
            labels.append(class_id)
            paths.append(path)

    X = np.stack(features)
    y = np.asarray(labels, dtype=np.int64)

    return X, y, paths


def save_splits(paths, y, train_idx, val_idx, test_idx):
    SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)

    split_map = {}

    for idx in train_idx:
        split_map[str(paths[idx])] = "train"

    for idx in val_idx:
        split_map[str(paths[idx])] = "val"

    for idx in test_idx:
        split_map[str(paths[idx])] = "test"

    with SPLIT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["landmark_path", "class_id", "label", "split"])

        for path, class_id in zip(paths, y):
            writer.writerow([
                path.relative_to(PROJECT_ROOT).as_posix(),
                int(class_id),
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"[class_id],
                split_map[str(path)],
            ])


def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    predictions = []
    targets = []

    with torch.no_grad():
        for x, y in loader:
            logits = model(x)
            loss = criterion(logits, y)

            total_loss += loss.item() * x.size(0)

            predictions.extend(logits.argmax(dim=1).cpu().numpy())
            targets.extend(y.cpu().numpy())

    loss = total_loss / len(loader.dataset)
    accuracy = accuracy_score(targets, predictions)
    macro_f1 = f1_score(
        targets,
        predictions,
        average="macro",
        zero_division=0,
    )

    return loss, accuracy, macro_f1, targets, predictions


def main():
    seed_everything(SEED)

    print("=" * 70)
    print("Static ISL Alphabet MLP Training")
    print("=" * 70)

    X, y, paths = load_dataset()

    print(f"\nSamples: {len(X)}")
    print(f"Features: {X.shape[1]}")
    print(f"Classes: {len(np.unique(y))}")

    indices = np.arange(len(X))

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=SEED,
    )

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        stratify=y[temp_idx],
        random_state=SEED,
    )

    print(f"\nTrain: {len(train_idx)}")
    print(f"Val:   {len(val_idx)}")
    print(f"Test:  {len(test_idx)}")

    save_splits(paths, y, train_idx, val_idx, test_idx)

    # Standardization parameters are computed ONLY from training data.
    mean = X[train_idx].mean(axis=0)
    std = X[train_idx].std(axis=0)

    std[std < 1e-8] = 1.0

    X = (X - mean) / std

    X_train = torch.from_numpy(X[train_idx])
    y_train = torch.from_numpy(y[train_idx])

    X_val = torch.from_numpy(X[val_idx])
    y_val = torch.from_numpy(y[val_idx])

    X_test = torch.from_numpy(X[test_idx])
    y_test = torch.from_numpy(y[test_idx])

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    test_loader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = StaticMLP()

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    best_val_f1 = -1.0
    best_epoch = 0
    patience_counter = 0

    print("\nStarting training...\n")

    for epoch in range(1, EPOCHS + 1):

        model.train()

        running_loss = 0.0

        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()

            logits = model(x_batch)
            loss = criterion(logits, y_batch)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * x_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_accuracy, val_f1, _, _ = evaluate(
            model,
            val_loader,
            criterion,
        )

        print(
            f"Epoch {epoch:03d} | "
            f"Train Loss {train_loss:.4f} | "
            f"Val Loss {val_loss:.4f} | "
            f"Val Acc {val_accuracy:.4f} | "
            f"Val F1 {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "input_dim": INPUT_DIM,
                    "num_classes": NUM_CLASSES,
                    "architecture": [126, 128, 64, 36],
                    "mean": mean.astype(np.float32),
                    "std": std.astype(np.float32),
                    "class_names": list(
                        "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    ),
                    "seed": SEED,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_val_f1,
                },
                CHECKPOINT_PATH,
            )

        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(
                f"\nEarly stopping at epoch {epoch}. "
                f"Best epoch: {best_epoch}"
            )
            break

    print("\nLoading best checkpoint...")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_accuracy, test_f1, targets, predictions = evaluate(
        model,
        test_loader,
        criterion,
    )

    print("\n" + "=" * 70)
    print("FINAL TEST RESULTS")
    print("=" * 70)

    print(f"Best epoch : {checkpoint['best_epoch']}")
    print(f"Test loss  : {test_loss:.4f}")
    print(f"Test acc   : {test_accuracy:.4f}")
    print(f"Test F1    : {test_f1:.4f}")

    print("\nClassification report:\n")

    print(
        classification_report(
            targets,
            predictions,
            target_names=list(
                "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            ),
            digits=4,
            zero_division=0,
        )
    )

    print(f"Checkpoint: {CHECKPOINT_PATH}")
    print(f"Splits:     {SPLIT_PATH}")


if __name__ == "__main__":
    main()
