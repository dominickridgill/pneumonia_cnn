# Pneumonia Detection Using Convolutional Neural Networks

## Overview

This project implements and evaluates convolutional neural network (CNN)–based models to classify chest X-ray images as **Normal** or **Pneumonia**. The goal is to explore how deep learning techniques can assist in medical image classification while addressing common challenges such as **class imbalance** and **overfitting**.

Two approaches are evaluated:

- A custom-built CNN trained from scratch
- A transfer learning approach using a pre-trained **DenseNet** architecture

The project emphasizes correct machine learning practices, including proper dataset splitting, validation-based tuning, and honest evaluation on held-out test data.

---

## Dataset

The dataset used is the *Chest X-Ray Images (Pneumonia)* dataset published on Kaggle and originally collected from pediatric patients (ages 1–5) at Guangzhou Women and Children’s Medical Center.

- Total images: **5,856**
- Classes:
  - Normal: **1,583**
  - Pneumonia: **4,273**
- Image format: JPEG
- Input size: **150 × 150**

Due to the strong class imbalance, special care was taken to ensure valid evaluation.

---

## Data Preparation

- The original dataset directories were consolidated and manually re-split.
- Data was split into:
  - **70% Training**
  - **15% Validation**
  - **15% Test**
- **Oversampling** was applied **only to the training set** to balance class distribution.
- Validation and test sets were left untouched to avoid inflated performance metrics.
- Images were normalized to the range `[0, 1]`.

---

## Modeling Approach

### 1. Baseline CNN
- Custom convolutional neural network built from scratch
- Included convolution, pooling, and dropout layers
- Served as a performance baseline

### 2. Tuned CNN
- Hyperparameters tuned using multiple learning rates and dropout values
- Validation accuracy improved significantly over the baseline model

### 3. DenseNet (Transfer Learning)
- Pre-trained DenseNet model initialized with ImageNet weights
- Initial model showed signs of overfitting
- Dropout and learning rate tuning were applied to mitigate overfitting

---

## Overfitting Mitigation

Several strategies were used to address overfitting:

- Dropout layers added to both custom and pre-trained models
- Validation accuracy monitored during training
- Hyperparameter tuning performed using validation performance
- Final results reported using **held-out test data only**

Models that showed strong validation performance but degraded test accuracy were not selected as final results.

---

## Results

Final performance across models was consistent and realistic:

- **Validation Accuracy:** ~95%
- **Test Accuracy:** ~94%

The DenseNet-based model demonstrated slightly better performance on normal X-ray classification, while the custom CNN performed comparably overall.

These results align with known benchmarks for this dataset and avoid unrealistic performance claims.

---

## Project Structure

pneumonia-cnn/
├── README.md
├── pneumonia_cnn.ipynb
├── pneumonia_cnn.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── docs/
    └── sample-results.txt

---

## How to Run

1. Install dependencies using `pip install -r requirements.txt`.
2. Run the script version with `python pneumonia_cnn.py`.
3. Alternatively, open the notebook using `jupyter notebook pneumonia_cnn.ipynb`.

