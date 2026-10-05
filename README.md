# Indian Sign Language Recognition

A planned computer-vision application for recognizing Indian Sign Language (ISL)
gestures from camera input. The repository is currently at the project-setup
stage: the Python environment and dependency lock exist, but the data pipeline,
model, training workflow, and user interface have not yet been implemented.

## Problem Statement

The project aims to translate visual ISL gestures into machine-readable labels
in near real time. The initial scope is supervised recognition from hand and
body landmarks extracted from video frames. The exact vocabulary, whether
gestures are static or dynamic, and the target accuracy and latency still need
to be defined against the selected dataset and deployment hardware.

## Assumptions

- Input will come from prerecorded video or a local webcam.
- MediaPipe will provide hand, pose, and/or face landmarks.
- PyTorch will be used to train the recognition model.
- ONNX and ONNX Runtime will provide a portable inference path.
- OpenCV will handle frame capture and basic image processing.
- PyQt5 is the planned desktop interface toolkit.
- The current environment uses CPU-only PyTorch; GPU support is not configured.
- Raw data, derived landmarks, model checkpoints, and exported ONNX models are
  local artifacts and are not committed to Git.

## Planned Architecture

```text
Camera or video
      |
      v
Frame capture (OpenCV)
      |
      v
Landmark extraction (MediaPipe)
      |
      v
Normalization and sequence construction
      |
      +------------------------+
      |                        |
      v                        v
Training (PyTorch)       Real-time inference
      |                        |
      v                        v
Checkpoints / ONNX       Smoothing and label output
                               |
                               v
                         Desktop UI (PyQt5)
```

The intended source layout is:

- `src/preprocessing/`: capture, landmark extraction, normalization, and splits
- `src/models/`: model definitions and shared inference code
- `src/training/`: training configuration and entry points
- `src/evaluation/`: metrics, reports, and model comparison
- `src/realtime/`: webcam inference and desktop UI
- `data/`: raw and processed local datasets (ignored by Git)
- `checkpoints/`: local training checkpoints (ignored by Git)
- `results/`: evaluation outputs

## Environment Setup

Python 3.14 is the currently tested interpreter. On Windows PowerShell:

```powershell
python3 --version
python3 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements-lock.txt
python -m pip check
```

For development against only the direct dependency set, use
`requirements.txt`. For a reproducible environment, prefer
`requirements-lock.txt`, which pins transitive dependencies as well.

To verify the core runtime imports:

```powershell
python -c "import cv2, mediapipe, onnx, onnxruntime, torch; print('environment OK')"
```

## Planned Milestones

1. Select and document the ISL vocabulary and dataset.
2. Implement landmark extraction and deterministic dataset splits.
3. Establish a baseline classifier and evaluation metrics.
4. Train and compare sequence-aware models for dynamic gestures.
5. Export the selected model to ONNX and validate output parity.
6. Build real-time inference with prediction smoothing.
7. Add the PyQt5 desktop workflow and end-to-end tests.

## Current Status

- Python 3.14 virtual environment created.
- Direct and transitive dependencies pinned.
- Repository initialized with Git.
- Application code, tests, datasets, and trained models are not yet present.
