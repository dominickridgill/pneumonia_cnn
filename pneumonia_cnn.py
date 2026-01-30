"""
Pneumonia Detection Using Convolutional Neural Networks

Author: Dominick Ridgill
Focus Areas: Machine Learning, Deep Learning, Medical Image Classification

This script trains and evaluates CNN-based models to classify chest X-ray
images as Normal or Pneumonia. It emphasizes proper dataset splitting,
overfitting mitigation, and validation-driven model selection.

This code is intended for educational and demonstration purposes only.
"""

import os
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121


# ----------------------------
# Configuration
# ----------------------------

IMAGE_SIZE: Tuple[int, int] = (150, 150)
BATCH_SIZE: int = 32
EPOCHS: int = 10
SEED: int = 42

DATA_DIR = "data"  # expected structure: data/train, data/val, data/test


# ----------------------------
# Data Generators
# ----------------------------

def create_data_generators():
    """
    Create ImageDataGenerators for training, validation, and testing.
    Training data includes augmentation; validation and test do not.
    """

    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=15,
        zoom_range=0.1,
        horizontal_flip=True
    )

    eval_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "train"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=True,
        seed=SEED
    )

    val_generator = eval_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "val"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    test_generator = eval_datagen.flow_from_directory(
        os.path.join(DATA_DIR, "test"),
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, val_generator, test_generator


# ----------------------------
# Model Definitions
# ----------------------------

def build_custom_cnn() -> tf.keras.Model:
    """
    Build a custom CNN model from scratch.
    """

    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(*IMAGE_SIZE, 3)),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


def build_densenet_model() -> tf.keras.Model:
    """
    Build a DenseNet-based model using transfer learning.
    """

    base_model = DenseNet121(
        weights="imagenet",
        include_top=False,
        input_shape=(*IMAGE_SIZE, 3)
    )

    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    return model


# ----------------------------
# Training & Evaluation
# ----------------------------

def train_and_evaluate(model: tf.keras.Model,
                       train_gen,
                       val_gen,
                       test_gen,
                       model_name: str):
    """
    Train the model and evaluate performance on validation and test data.
    """

    print(f"\nTraining {model_name}...\n")

    model.fit(
        train_gen,
        epochs=EPOCHS,
        validation_data=val_gen
    )

    print(f"\nEvaluating {model_name} on test data...\n")

    test_loss, test_accuracy = model.evaluate(test_gen)
    print(f"{model_name} Test Accuracy: {test_accuracy:.4f}")


# ----------------------------
# Main Execution
# ----------------------------

def main():
    """
    Main execution function.
    """

    train_gen, val_gen, test_gen = create_data_generators()

    custom_cnn = build_custom_cnn()
    train_and_evaluate(custom_cnn, train_gen, val_gen, test_gen, "Custom CNN")

    densenet_model = build_densenet_model()
    train_and_evaluate(densenet_model, train_gen, val_gen, test_gen, "DenseNet CNN")


if __name__ == "__main__":
    main()
