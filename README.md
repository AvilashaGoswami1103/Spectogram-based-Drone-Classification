# Spectrogram-based Drone Classification

## 📌 Overview
This project classifies drones using RF (Radio Frequency) signals by converting them into spectrogram images and applying deep learning models.

## 🔄 Pipeline

IQ Signals  
↓  
Spectrogram Generation (spectogram.py)  
↓  
Train/Validation Split (train_val_split.py)  
↓  
Data Augmentation (augment.py)  
↓  
Model Training (ResNet18 / ResNet50)  
↓  
Inference & Prediction (inference.py)

## 💡 Key Highlights
- Converts raw RF IQ signals into spectrogram images  
- Uses CNN architectures (ResNet18 & ResNet50) for classification  
- Applies physically meaningful augmentation (no unrealistic flips/rotations)  
- Demonstrates transfer learning for RF signal classification  
- End-to-end pipeline from raw data to prediction

## 📂 Dataset
Training data:  
https://huggingface.co/datasets/kitofrank/RFUAV/tree/main

## 1️⃣ spectogram.py: Data Preprocessing & Spectrogram Generation

This module converts raw IQ (In-phase and Quadrature) signal data into spectrogram images suitable for CNN training.
### 🔹 Functionality
- Reads .iq files containing RF signal samples  
- Converts IQ → complex signal (I + jQ)  
- Applies DC offset removal and normalization  
### 🔹 Signal Processing
- Uses Short-Time Fourier Transform (STFT)  
- Applies:
  - 1024-point Hamming window  
  - FFT shift (center frequency at 0 Hz)  
### 🔹 Output
- Generates clean spectrogram images:
  - Size: 224 × 224 pixels  
  - No axes/labels (CNN-ready)  
- Uses sliding window to generate multiple images per signal  
IQ Signal → Spectrogram Images → CNN Input

## 2️⃣ train_val_split.py: Dataset Organization & Splitting

This module prepares the dataset for training.
### 🔹 Features
- Organizes images into class-based folders:
  - DEVENTION  
  - FATUBA  
  - FLYSKY  
  - YUNZHOU  
- Performs 80:20 split:
  - 80% → Training  
  - 20% → Validation  
- Uses fixed random seed for reproducibility  
### 🔹 Output Structure
dataset_new/  
 ├── train/  
 └── val/  
Produces a structured dataset compatible with PyTorch loaders.

## 3️⃣ augment.py: Data Augmentation for Model Generalization

This module enhances the training dataset to improve model robustness.
### 🔹 Key Insight
Unlike conventional image augmentation, transformations are carefully selected to preserve the physical meaning of RF spectrograms.
### 🔹 Techniques Used
- Gaussian Noise  
- Advanced Blur  
- CLAHE  
- ISO Noise  
- Sharpening  
- Brightness Adjustment  
### 🔹 Output
- Training set:
  - Each image → 7 versions (1 original + 6 augmented)  
- Validation set:
  - Only original images  
Improves robustness to noise and real-world signal variations.

## 4️⃣ train_resnet18.py & train_resnet50.py: Model Training

These modules implement deep learning models using transfer learning.
### 🔹 Models Used
- ResNet18 → lightweight, faster  
- ResNet50 → deeper, higher accuracy potential  
### 🔹 Training Details
- Input size: 224 × 224  
- Loss: CrossEntropyLoss  
- Optimizer: Adam  
- Learning Rate: 0.0001  
- Batch Size: 16  
### 🔹 Transfer Learning
- Uses pretrained ImageNet weights  
- Final layer modified → 4 drone classes  
Demonstrates effectiveness of transfer learning in RF signal classification.

## 5️⃣ inference.py: Model Evaluation & Prediction

This module performs prediction on unseen spectrogram images.
### 🔹 Workflow
For each test image:
1. Load and preprocess image  
2. Pass through trained model  
3. Apply Softmax  
4. Output prediction and confidence  
### 🔹 Output
- Predicted class  
- Confidence score  
- Class-wise probabilities  
- CSV file with all results  
Final step converting model knowledge into real-world predictions.


   
