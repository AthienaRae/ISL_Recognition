"""One final held-out evaluation of the frozen Greetings GRU checkpoint."""
from __future__ import annotations
import argparse, csv
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from torch import nn
from torch.utils.data import DataLoader
from src.models.gru_classifier import GRUClassifier
from src.training.greetings_dataset import DEFAULT_METADATA, GreetingsDataset, build_class_mapping, load_metadata

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "checkpoints/gru_greetings_best.pt"
OUTPUT = ROOT / "results/gru_test"
EXPECTED = {"train": 132, "val": 17, "test": 41}

def mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows])) if rows else float("nan")

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT)
    args = parser.parse_args()
    rows = load_metadata(args.metadata)
    grouped = {split: [r for r in rows if r["split"].strip() == split] for split in EXPECTED}
    counts = {split: len(value) for split, value in grouped.items()}
    if counts != EXPECTED:
        raise ValueError(f"Expected {EXPECTED}, got {counts}.")
    source_sets = {split: {r["source_video_path"].casefold() for r in value} for split, value in grouped.items()}
    if source_sets["test"] & source_sets["train"] or source_sets["test"] & source_sets["val"]:
        raise ValueError("Test source overlap detected.")

    checkpoint = torch.load(args.checkpoint.resolve(), map_location="cpu", weights_only=False)
    if checkpoint["class_to_index"] != build_class_mapping(rows):
        raise ValueError("Checkpoint class mapping differs from metadata.")
    expected_config = {"input_size": 126, "hidden_size": 128, "num_layers": 1,
                       "num_classes": 9, "dropout": 0.3}
    if checkpoint["model_config"] != expected_config or checkpoint["sequence_length"] != 91:
        raise ValueError("Checkpoint is not the frozen baseline contract.")
    dataset = GreetingsDataset("test", args.metadata)
    if len(dataset) != 41 or len(dataset.class_to_index) != 9:
        raise AssertionError("Expected 41 test samples and 9 classes.")
    for index in range(len(dataset)):
        features, _, length = dataset[index]
        if features.shape != (91, 126) or not 1 <= int(length) <= 91:
            raise AssertionError(f"Invalid test sample {index}: {features.shape}, {length}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GRUClassifier(**expected_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    targets, predictions, confidences, seconds, second_confidences = [], [], [], [], []
    loss_sum = 0.0
    with torch.no_grad():
        for features, labels, lengths in DataLoader(dataset, batch_size=16, shuffle=False):
            labels = labels.to(device)
            logits = model(features.to(device), lengths)
            loss_sum += float(nn.functional.cross_entropy(logits, labels, reduction="sum"))
            values, indices = torch.softmax(logits, 1).topk(2, dim=1)
            targets += labels.cpu().tolist(); predictions += indices[:, 0].cpu().tolist()
            confidences += values[:, 0].cpu().tolist(); seconds += indices[:, 1].cpu().tolist()
            second_confidences += values[:, 1].cpu().tolist()
    labels = list(range(9)); names = [dataset.index_to_class[i] for i in labels]
    loss = loss_sum / len(dataset); accuracy = float(np.mean(np.equal(targets, predictions)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets, predictions, labels=labels, average="macro", zero_division=0)
    report = classification_report(targets, predictions, labels=labels, target_names=names,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(targets, predictions, labels=labels)

    metadata = {r["source_video_path"]: r for r in grouped["test"]}
    records, errors, pairs = [], [], Counter()
    for i, sample in enumerate(dataset.samples):
        row = metadata[sample.source_video_path]; total = int(row["frame_count"])
        record = {"class_id": sample.class_id, "true_label": names[targets[i]],
                  "predicted_label": names[predictions[i]], "correctness": targets[i] == predictions[i],
                  "source_video_path": sample.source_video_path, "predicted_confidence": confidences[i],
                  "top2_label": names[seconds[i]], "top2_confidence": second_confidences[i],
                  "sequence_length": total, "zero_hand_frames": int(row["zero_hand_frames"]),
                  "zero_hand_rate": int(row["zero_hand_frames"]) / total,
                  "one_hand_frames": int(row["one_hand_frames"]),
                  "one_hand_rate": int(row["one_hand_frames"]) / total}
        records.append(record)
        if not record["correctness"]:
            errors.append(record); pairs[(record["true_label"], record["predicted_label"])] += 1
    correct_rows = [r for r in records if r["correctness"]]
    incorrect_rows = errors
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    with (out / "test_predictions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    with (out / "confusion_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle); writer.writerow(["true/predicted", *names])
        for name, row in zip(names, matrix.tolist()): writer.writerow([name, *row])

    summary = [f"Test samples: 41", f"Test loss: {loss:.6f}", f"Test accuracy: {accuracy:.6f}",
               f"Macro precision: {precision:.6f}", f"Macro recall: {recall:.6f}",
               f"Macro F1: {f1:.6f}", f"Correct/incorrect: {len(correct_rows)}/{len(errors)}",
               "", "Per-class results"]
    for name in names:
        r = report[name]; summary.append(f"{name}: precision={r['precision']:.6f}, recall={r['recall']:.6f}, f1={r['f1-score']:.6f}, support={int(r['support'])}")
    summary += ["", "Confusion matrix (rows=true, columns=predicted)", "Class order: " + " | ".join(names)]
    summary += [" ".join(map(str, row)) for row in matrix.tolist()]
    (out / "test_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    analysis = [f"Correct: {len(correct_rows)}", f"Incorrect: {len(errors)}",
                f"Mean correct confidence: {mean(correct_rows, 'predicted_confidence'):.6f}",
                f"Mean incorrect confidence: {mean(incorrect_rows, 'predicted_confidence'):.6f}",
                f"High-confidence errors (>=0.8): {sum(float(r['predicted_confidence']) >= .8 for r in errors)}",
                "Confusion pairs: " + (", ".join(f"{a} -> {b}: {n}" for (a,b),n in pairs.most_common()) or "none"),
                "", "Misclassified samples"]
    for r in errors:
        analysis.append(f"{r['source_video_path']}: {r['true_label']} -> {r['predicted_label']}, confidence={r['predicted_confidence']:.6f}, length={r['sequence_length']}, zero={r['zero_hand_rate']:.2%}, one={r['one_hand_rate']:.2%}")
    for title, group in (("correct", correct_rows), ("incorrect", incorrect_rows)):
        analysis.append(f"Mean {title}: length={mean(group,'sequence_length'):.2f}, zero-rate={mean(group,'zero_hand_rate'):.2%}, one-rate={mean(group,'one_hand_rate'):.2%}")
    (out / "error_analysis.txt").write_text("\n".join(analysis) + "\n", encoding="utf-8")
    print("\n".join(summary + ["", *analysis, f"Device: {device}", "Weights updated: No"]))

if __name__ == "__main__": main()
