# 🛒 Shopper Activity Monitoring

A real-time shopper behavior classification system using pose keypoints extracted from MediaPipe and action recognition via a BiLSTM model with attention. This system identifies shopper actions like *reaching to shelf*, *retracting*, *inspecting*, and *keeping hand on shelf* based on video input.

---

## 🎯 Objective

To build an end-to-end, pose-based human action classification pipeline for retail environments. This helps in monitoring shopper interactions with shelves to enhance retail analytics and automate behavior-based insights.

---

## 📌 Features

- ✅ Keypoint extraction using MediaPipe Holistic (Pose + Hand landmarks)
- ✅ BiLSTM model with attention mechanism for action classification
- ✅ Real-time inference and prediction every 30 frames
- ✅ Model trained on custom dataset with 4 shopper action classes
- ✅ Automatically saves extracted keypoints to `.txt` file

---

## 🧠 Action Classes

1. **Reach to shelf**
2. **Retract from shelf**
3. **Hand on shelf**
4. **Inspecting**

---


## setup_and_installation:
  steps:
    - step: "Clone this repo"
      commands:
        - git clone https://github.com/your-username/shopper-activity-monitoring.git
        - cd shopper-activity-monitoring

    - step: "Create environment & install dependencies"
      commands:
        - pip install -r requirements.txt

    - step: "Ensure PyTorch & MediaPipe are working properly"
      notes:
        - "Visit https://pytorch.org/get-started/locally/ to install the appropriate PyTorch version for your system (CPU/GPU)."
        - "Make sure MediaPipe is correctly installed. You may use: pip install mediapipe"

## training:
  description: "Train the BiLSTM + Attention model using pose keypoints"
  command: python train_model.py
  notes:
    - "Edit `train_model.py` to configure paths and class names"
    - "Input shape: [30 frames, 44 features]"
    - "Architecture: 3-layer BiLSTM + Attention + FC layers"
    - "Loss: CrossEntropyLoss"
    - "Optimizer: Adam"

## inference:
  description: "Run inference on video to predict shopper actions"
  command: python inference.py
  behavior:
    - "Loads trained model checkpoint"
    - "Extracts pose and hand keypoints with MediaPipe"
    - "Predicts action every 30 frames"
    - "Saves keypoints to .txt file"
    - "Displays FPS and predicted action per window"

## sample_output:
  example_terminal_output:
    - "Frame 30: Predicted action - Inspecting"
    - "Frame 60: Predicted action - Reach to shelf"

## dependencies:
  required_packages:
    - torch
    - torchmetrics
    - opencv-python
    - mediapipe
    - numpy
  source: requirements.txt

