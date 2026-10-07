"""Export the Greetings GRU checkpoint to a fixed-length ONNX model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = PROJECT_ROOT / "checkpoints/gru_greetings_best.pt"
OUTPUT = PROJECT_ROOT / "models/gru_greetings.onnx"

SEQUENCE_LENGTH = 91
FEATURE_DIMENSION = 126


class MobileGRUClassifier(nn.Module):
    """Fixed-length version of the trained Greetings GRU for ONNX."""

    def __init__(
        self,
        input_size: int = 126,
        hidden_size: int = 128,
        num_layers: int = 1,
        num_classes: int = 9,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size,
            hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,
        )

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(features)
        return self.classifier(self.dropout(hidden[-1]))


def main() -> None:
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {CHECKPOINT}")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint: {CHECKPOINT}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "model_state_dict",
        "model_config",
        "sequence_length",
        "class_to_index",
    }

    missing = required - checkpoint.keys()
    if missing:
        raise ValueError(f"Checkpoint is missing keys: {sorted(missing)}")

    if checkpoint["sequence_length"] != SEQUENCE_LENGTH:
        raise ValueError(
            f"Expected sequence length {SEQUENCE_LENGTH}, "
            f"got {checkpoint['sequence_length']}"
        )

    config = checkpoint["model_config"]

    print(f"Model configuration: {config}")
    print(f"Class mapping: {checkpoint['class_to_index']}")

    model = MobileGRUClassifier(**config)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Fixed mobile input: exactly 91 frames × 126 features.
    dummy_input = torch.zeros(
        1,
        SEQUENCE_LENGTH,
        FEATURE_DIMENSION,
        dtype=torch.float32,
    )

    with torch.no_grad():
        pytorch_output = model(dummy_input)

    print(f"PyTorch output shape: {tuple(pytorch_output.shape)}")

    print(f"Exporting ONNX model: {OUTPUT}")

    torch.onnx.export(
        model,
        dummy_input,
        str(OUTPUT),
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        dynamo=False,
    )

    print("ONNX export complete.")

    if not OUTPUT.is_file():
        raise RuntimeError("ONNX file was not created.")

    print(f"ONNX file: {OUTPUT}")
    print(f"ONNX size: {OUTPUT.stat().st_size / 1024:.1f} KB")

    # Optional ONNX Runtime verification.
    try:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(OUTPUT),
            providers=["CPUExecutionProvider"],
        )

        onnx_output = session.run(
            ["logits"],
            {"features": dummy_input.numpy()},
        )[0]

        difference = np.max(
            np.abs(pytorch_output.numpy() - onnx_output)
        )

        print(f"PyTorch vs ONNX max difference: {difference:.8f}")

        if difference > 1e-4:
            raise RuntimeError(
                f"ONNX verification failed. Maximum difference: {difference}"
            )

        print("ONNX verification: PASS")

    except ImportError:
        print(
            "ONNX Runtime is not installed, so runtime verification was skipped."
        )

    print()
    print("Export finished successfully.")
    print(f"Model saved to: {OUTPUT}")


if __name__ == "__main__":
    main()
