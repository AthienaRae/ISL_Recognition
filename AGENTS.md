# ISL Recognition Project Agent Instructions

## Project Goal

Build a real-time Indian Sign Language recognition and speech translation
Android application.

Core pipeline:

Camera
→ MediaPipe landmarks
→ landmark normalization
→ temporal sequence
→ PyTorch-trained model
→ on-device inference
→ predicted ISL sign
→ text
→ speech

## Current Development Environment

Project root:
D:\projects\ISL_Recognition

Python:
3.11

Virtual environment:
venv

Activate using:
.\venv\Scripts\Activate.ps1

Important:
Never recreate or replace the virtual environment unless explicitly required.

## Python ML Stack

- PyTorch
- MediaPipe Tasks API
- OpenCV
- NumPy
- Pandas
- scikit-learn

Do NOT use legacy MediaPipe mp.solutions APIs unless explicitly requested.

## Dataset

Primary dynamic dataset:
INCLUDE

Current pilot:
Greetings category

Current pilot contains:
- 9 classes
- 190 physical videos

Classes:
48. Hello
49. How are you
50. Alright
51. Good Morning
52. Good afternoon
53. Good evening
54. Good night
55. Thank you
56. Pleased

Raw dataset location:

data/raw/include/extracted/Greetings/

Clean split file:

data/splits/include_greetings_full_clean.csv

The clean split file uses:
include_50 = False

Do NOT combine INCLUDE and INCLUDE-50 split definitions.

There must be:
- one physical video per metadata record
- exactly one split per video
- no train/test leakage

## Existing Verified Pipeline

OpenCV successfully decodes INCLUDE .MOV files.

Verified sample:

data/raw/include/extracted/Greetings/48. Hello/MVI_0029.MOV

Properties:
1920x1080
25 FPS
63 frames
2.52 seconds

## MediaPipe

Installed version:
1.0.1

Use:
MediaPipe Tasks Hand Landmarker

Target representation:

21 landmarks per hand
x, y, z coordinates
2 hands maximum

Feature vector per frame:

21 × 3 × 2 = 126 values

Convention:

Left hand first
Right hand second

If one hand is absent:
fill that hand's 63 values with zeros.

All features must ultimately be float32.

## Landmark Processing

The preprocessing pipeline should be:

video
→ frames
→ BGR to RGB
→ MediaPipe Hand Landmarker
→ handedness detection
→ fixed left/right ordering
→ landmark extraction
→ normalization
→ temporal sequence
→ save processed features

Normalization must make landmarks less sensitive to:
- screen position
- distance from camera
- signer size

Do not destroy relative finger geometry.

## Model Research Plan

Start with small baselines before advanced models.

Experiment order:

1. MLP baseline where appropriate
2. GRU
3. LSTM
4. TCN + GRU
5. Lightweight Transformer only if justified

Primary dynamic model candidate:
TCN + GRU

Evaluation:

- Accuracy
- Macro F1
- Per-class precision/recall/F1
- Confusion matrix
- Inference latency
- Model size

Never evaluate on training data.

## Sequence Handling

Do not arbitrarily assume all videos have the same frame count.

First inspect the dataset distribution:
- FPS
- duration
- number of frames

Then choose:
- temporal resampling
or
- padding/truncation

Document the chosen sequence length and reasoning.

## Android Target

Native Android application.

Language:
Kotlin

Mobile pipeline:

CameraX
→ MediaPipe Hand Landmarker
→ normalization
→ temporal buffer
→ exported ML model
→ prediction
→ text
→ Android TextToSpeech

Core recognition should work offline.

Do not implement server-side inference unless explicitly requested.

## Mobile Model Runtime

Training remains in PyTorch.

Deployment runtime should be selected based on current compatibility.
ExecuTorch is the preferred candidate.

Do not integrate mobile runtime until the offline Python model works.

## Development Priorities

Current priority:

1. Verify MediaPipe landmark extraction on one video.
2. Inspect detection quality.
3. Determine whether hands-only landmarks are sufficient.
4. Implement stable 126-feature extraction.
5. Process all 190 Greetings videos.
6. Train baseline model.
7. Evaluate held-out performance.
8. Only then expand dataset.
9. Only after ML proof-of-concept, integrate Android inference.

## Code Quality Rules

Use pathlib for filesystem paths.

Never hard-code:
C:\
D:\

Shared project code must work regardless of teammate drive letter.

Prefer:
PROJECT_ROOT = Path(__file__).resolve().parents[...]

Add clear logs and validation checks.

Scripts should fail loudly for:
- missing files
- invalid metadata
- duplicate samples
- split leakage
- failed video decoding
- invalid tensor shapes

Do not silently skip corrupt data.

## Git Rules

Never commit:

venv/
large raw datasets
model caches
temporary outputs

Do commit:

scripts
source code
configs
metadata definitions
split CSVs if small
README
model contract documentation

Do not make destructive Git operations unless explicitly instructed.

## Current Objective

Build the first complete offline proof-of-concept:

INCLUDE video
→ MediaPipe landmarks
→ temporal model
→ predicted Greetings sign

Do not jump ahead to:
- adaptive user learning
- context-aware sentence formation
- Play Store publishing

until the core recognizer works reliably.