# ISL Recognition

A real-time Indian Sign Language (ISL) recognition system that combines hand-landmark-based computer vision, deep learning, and Android deployment for recognizing static and dynamic signs.

## Project Overview

This project aims to develop an AI-powered Indian Sign Language recognition system capable of recognizing both static hand signs and dynamic sign sequences.

The system uses MediaPipe Hand Landmarker for extracting hand landmarks and deep learning models for sign classification. The trained models are exported to ONNX and integrated into an Android application using ONNX Runtime for on-device inference.

The overall architecture consists of two recognition paths:

* **Static Path:** Multi-Layer Perceptron (MLP) for single-frame fingerspelling recognition.
* **Dynamic Path:** Bidirectional LSTM (Bi-LSTM) / GRU-based sequence models for temporal sign recognition.

The Android application provides the real-time camera interface and performs model inference directly on the device.

## Current Project Status

### Implemented

* MediaPipe-based hand landmark extraction
* Two-hand landmark representation
* Wrist-relative landmark normalization
* Hand-slot assignment for consistent left/right representation
* 126-dimensional landmark feature representation
* Dynamic landmark sequence preprocessing
* Variable-length sequence handling
* Padding and masking
* GRU-based dynamic recognition model
* Greetings-category dataset pipeline
* Model evaluation scripts
* ONNX model export
* Android application with CameraX
* MediaPipe Hand Landmarker integration on Android
* ONNX Runtime inference on Android
* Real-time dynamic sign recognition
* Recognition confidence display
* Physical Android-device testing

### In Progress

* Complete 36-class static MLP training
* Static A-Z and 0-9 recognition
* Full 263-class dynamic recognition model
* Prediction smoothing and temporal stabilization
* Context-aware sentence formation
* Adaptive user learning
* Complete Android evaluation
* Final performance analysis

## System Architecture

```text
                    Camera Input
                         │
                         ▼
              MediaPipe Hand Landmarker
                         │
                         ▼
             Hand Landmark Extraction
                         │
                         ▼
             Landmark Normalization
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Static Recognition     Dynamic Recognition
              │                     │
              ▼                     ▼
        MLP Classifier       GRU / Bi-LSTM
              │                     │
              │                     ▼
              │              Temporal Processing
              │                     │
              └──────────┬──────────┘
                         ▼
                 Sign Prediction
                         │
                         ▼
               Android Application
```

## Landmark Representation

Each detected hand contains 21 landmarks. Each landmark consists of three coordinates:

```text
21 landmarks × 3 coordinates = 63 features per hand
```

For two hands:

```text
63 × 2 = 126 features per frame
```

The landmark representation is normalized before being provided to the classification models.

The preprocessing includes:

* Wrist-relative coordinate normalization
* Reference-length scaling
* Consistent left/right hand slot assignment
* Missing-hand handling
* Temporal padding and masking for dynamic sequences

## Recognition Models

### Static MLP

The static recognition path is designed for fingerspelling and other single-frame hand configurations.

**Target classes:** 36

```text
A-Z = 26 classes
0-9 = 10 classes
Total = 36 classes
```

The model receives a single normalized 126-dimensional landmark vector.

Planned architecture:

```text
Input: 126
    ↓
Linear: 128
    ↓
ReLU
    ↓
Dropout
    ↓
Linear: 64
    ↓
ReLU
    ↓
Dropout
    ↓
Linear: 36
    ↓
Output
```

The complete 36-class training and evaluation pipeline is currently being developed.

### Dynamic Recognition

The dynamic path processes a sequence of normalized landmark frames rather than a single frame.

The current Android prototype uses a GRU-based dynamic recognition model trained on the ISL Greetings category.

The broader project architecture is intended to support a Bi-LSTM model for the complete dynamic class set.

Dynamic processing includes:

* Variable-length sequences
* Sequence padding
* Masking
* Packed sequence processing
* Temporal classification

## Current Dynamic Prototype

The current prototype focuses on the Greetings category from the INCLUDE ISL dataset.

The dataset contains:

* 9 greeting-related sign classes
* 190 videos

The preprocessing pipeline successfully processes the dataset into normalized landmark sequences for model training and evaluation.

The trained GRU model has been exported to:

```text
ONNX
```

and integrated into the Android application.

## Android Application

The Android application is built using native Android development tools and performs on-device inference.

### Main Components

* Kotlin
* Jetpack Compose
* CameraX
* MediaPipe Hand Landmarker
* ONNX Runtime
* Android SDK

The Android application currently performs the following pipeline:

```text
Camera
  ↓
CameraX
  ↓
MediaPipe Hand Landmarker
  ↓
126D Landmark Features
  ↓
Normalization
  ↓
ONNX Model
  ↓
Prediction
  ↓
Recognized Sign + Confidence
```

The model and MediaPipe task file are bundled with the Android application.

## Project Structure

```text
ISL_Recognition/
│
├── app/
│   └── src/
│       └── main/
│           ├── assets/
│           │   ├── gru_greetings.onnx
│           │   └── hand_landmarker.task
│           │
│           ├── java/
│           │   └── com/athiena/islrecognition/
│           │       ├── HandLandmarkerHelper.kt
│           │       ├── MainActivity.kt
│           │       ├── OnnxClassifier.kt
│           │       └── ui/
│           │
│           └── AndroidManifest.xml
│
├── data/
│   └── splits/
│
├── scripts/
│   ├── audit_hand_detection.py
│   ├── extract_greetings_landmarks.py
│   ├── realtime_greetings_demo.py
│   └── ...
│
├── src/
│   ├── evaluation/
│   ├── models/
│   ├── preprocessing/
│   └── training/
│
├── requirements.txt
├── requirements-lock.txt
├── build.gradle.kts
├── settings.gradle.kts
└── README.md
```

## Technologies Used

| Component               | Technology                 |
| ----------------------- | -------------------------- |
| Programming Language    | Python, Kotlin             |
| Computer Vision         | MediaPipe                  |
| Deep Learning           | PyTorch                    |
| Static Model            | MLP                        |
| Dynamic Model           | GRU / Bi-LSTM              |
| Model Deployment        | ONNX                       |
| Android Inference       | ONNX Runtime               |
| Android UI              | Jetpack Compose            |
| Camera                  | CameraX                    |
| Dataset                 | INCLUDE ISL / ISL Alphabet |
| Development Environment | Android Studio, Python     |

## Dataset

The project uses Indian Sign Language datasets for training and evaluation.

The static recognition path is designed around the ISL Alphabet dataset containing:

* A-Z
* 0-9

The dynamic recognition path uses the INCLUDE ISL dataset for sentence/sign sequence recognition.

Dataset files are not redistributed through this repository unless their respective licenses permit redistribution.

## Reproducibility

The Python environment can be installed using:

```bash
pip install -r requirements.txt
```

The project contains scripts for:

* Dataset preparation
* Dataset validation
* Hand-detection auditing
* Landmark extraction
* Landmark normalization
* Model training
* Model evaluation
* Real-time testing

## Model Deployment

Trained PyTorch models are exported to ONNX for deployment.

The Android application uses ONNX Runtime to execute inference locally on the device.

This allows recognition without requiring continuous communication with a remote server.

## Research Objective

The long-term objective is to develop a hybrid ISL recognition system that combines:

1. Static sign recognition
2. Dynamic sign recognition
3. User-adaptive learning
4. Temporal prediction stabilization
5. Context-aware sentence formation
6. Real-time on-device inference

The final system is intended to provide a practical bridge between Indian Sign Language gestures and readable digital communication.

## Authors

**Athiena Rachel J**|
**Abishek S**|
**Arun Prakash M**|
**Ashwin Kumar AP** 


Final Year Project - Phase I
Computer Science and Engineering

---

## License

This repository is intended for academic and research purposes.
