"""
train.py
--------
CNN model definition, training with callbacks, and model saving
for the Hand Gesture Recognition task.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint,
)
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import models_path, outputs_path, ensure_dir, save_figure, set_plot_style


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE     = 64
NUM_CHANNELS = 1
BATCH_SIZE   = 32
EPOCHS       = 30
LEARNING_RATE = 1e-3


# ---------------------------------------------------------------------------
# Model architecture
# ---------------------------------------------------------------------------

def build_cnn(num_classes: int,
              img_size: int    = IMG_SIZE,
              channels: int    = NUM_CHANNELS) -> tf.keras.Model:
    """
    Build a compact CNN for gesture classification.

    Architecture
    ------------
    Block 1 : Conv(32) → BN → ReLU → Conv(32) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 2 : Conv(64) → BN → ReLU → Conv(64) → BN → ReLU → MaxPool → Dropout(0.25)
    Block 3 : Conv(128)→ BN → ReLU → MaxPool → Dropout(0.25)
    Head    : Flatten → Dense(256) → BN → ReLU → Dropout(0.50) → Softmax(num_classes)

    Parameters
    ----------
    num_classes : number of gesture classes
    img_size    : spatial dimension of input images
    channels    : 1 (grayscale) or 3 (RGB)

    Returns
    -------
    Uncompiled tf.keras.Model
    """
    input_shape = (img_size, img_size, channels)

    model = models.Sequential(name="GestureNet", layers=[
        # Input
        layers.Input(shape=input_shape),

        # Block 1
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(32, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Conv2D(64, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), padding="same"),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Fully connected head
        layers.Flatten(),
        layers.Dense(256, kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Activation("relu"),
        layers.Dropout(0.50),

        # Output
        layers.Dense(num_classes, activation="softmax"),
    ])

    return model


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_model(model: tf.keras.Model,
                  lr: float = LEARNING_RATE) -> tf.keras.Model:
    """Compile with Adam optimiser + categorical cross-entropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

def get_callbacks() -> list:
    """
    Build training callbacks.

    Callbacks
    ---------
    EarlyStopping       : stops after 7 epochs without val_accuracy improvement
    ReduceLROnPlateau   : halves LR after 4 epochs of no val_loss improvement
    ModelCheckpoint     : saves best model weights (by val_accuracy)

    Returns
    -------
    list of tf.keras.callbacks.Callback
    """
    ensure_dir(models_path())

    return [
        EarlyStopping(
            monitor="val_accuracy", patience=7,
            restore_best_weights=True, verbose=1
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=4,
            min_lr=1e-6, verbose=1
        ),
        ModelCheckpoint(
            filepath=models_path("best_model.keras"),
            monitor="val_accuracy", save_best_only=True,
            verbose=1
        ),
    ]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(model: tf.keras.Model,
                X_train: np.ndarray, y_train: np.ndarray,
                X_val:   np.ndarray, y_val:   np.ndarray,
                datagen: ImageDataGenerator) -> tf.keras.callbacks.History:
    """
    Train the CNN using the augmentation generator for the training data.

    Returns
    -------
    history : Keras training history
    """
    callbacks = get_callbacks()
    steps_per_epoch = len(X_train) // BATCH_SIZE

    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=RANDOM_SEED),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    print(f"[INFO] Training complete  →  {len(history.epoch)} epochs run")
    return history


RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(history: tf.keras.callbacks.History) -> None:
    """
    Plot accuracy and loss curves for training and validation sets.

    Saves → outputs/training_curves.png
    """
    set_plot_style()
    hist = history.history

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Training History", fontsize=14, fontweight="bold")

    # Accuracy
    axes[0].plot(hist["accuracy"],     label="Train",      color="#4C72B0", linewidth=2)
    axes[0].plot(hist["val_accuracy"], label="Validation", color="#DD8452", linewidth=2, linestyle="--")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    # Loss
    axes[1].plot(hist["loss"],     label="Train",      color="#4C72B0", linewidth=2)
    axes[1].plot(hist["val_loss"], label="Validation", color="#DD8452", linewidth=2, linestyle="--")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    save_figure(fig, "training_curves.png")


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def save_model(model: tf.keras.Model, filename: str = "gesture_cnn.keras") -> None:
    ensure_dir(models_path())
    filepath = models_path(filename)
    model.save(filepath)
    print(f"[INFO] CNN model saved → {filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_training(num_classes: int,
                 X_train: np.ndarray, y_train: np.ndarray,
                 X_val:   np.ndarray, y_val:   np.ndarray,
                 datagen: ImageDataGenerator) -> tuple:
    """
    Full training workflow.

    Returns
    -------
    model   : trained tf.keras.Model
    history : Keras History object
    """
    print("[STEP] Building CNN …")
    model = build_cnn(num_classes)
    model = compile_model(model)

    print("[STEP] Training CNN …")
    history = train_model(model, X_train, y_train, X_val, y_val, datagen)

    print("[STEP] Plotting training curves …")
    plot_training_curves(history)

    print("[STEP] Saving final model …")
    save_model(model)

    return model, history
