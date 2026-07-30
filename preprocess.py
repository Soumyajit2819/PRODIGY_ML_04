"""
preprocess.py
-------------
Image loading, preprocessing, augmentation setup, and dataset splitting
for the Hand Gesture Recognition CNN task.
"""

import os
import shutil
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from utils import (
    data_path,
    outputs_path,
    ensure_dir,
    save_figure,
    set_plot_style,
    build_label_map,
)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE     = 64        # resize images to 64 × 64
NUM_CHANNELS = 1         # grayscale
MAX_PER_CLASS = 500      # max images per gesture class (keep memory manageable)
DATASET_DIR  = data_path("leapGestRecog")


# ---------------------------------------------------------------------------
# Dataset download
# ---------------------------------------------------------------------------

def download_dataset() -> str:
    """
    Download the LeapGestRecog dataset via kagglehub.
    Places it at data/leapGestRecog/ and returns that path.
    """
    ensure_dir(data_path())

    if os.path.isdir(DATASET_DIR) and len(os.listdir(DATASET_DIR)) > 0:
        print(f"[INFO] Dataset already present → {DATASET_DIR}")
        return DATASET_DIR

    print("[INFO] Downloading LeapGestRecog dataset from Kaggle …")
    dl_path = kagglehub.dataset_download("gti-upm/leapgestrecog")
    print(f"[INFO] Downloaded to cache → {dl_path}")

    # Walk to find the folder that contains subject subdirs with gesture images
    # Structure: <dl_path>/leapGestRecog/<subject>/<gesture>/frame_*.png
    target = None
    for root, dirs, files in os.walk(dl_path):
        # Look for a folder whose children contain image files
        png_children = []
        for d in dirs:
            sub = os.path.join(root, d)
            for _, subdirs2, _ in os.walk(sub):
                for sd2 in subdirs2:
                    sp2 = os.path.join(sub, sd2)
                    if any(f.endswith('.png') for f in os.listdir(sp2) if os.path.isfile(os.path.join(sp2,f))):
                        png_children.append(d)
                        break
                break
        if len(png_children) >= 5:
            target = root
            break

    if target is None:
        target = dl_path

    if not os.path.isdir(DATASET_DIR):
        print(f"[INFO] Copying dataset from {target} → {DATASET_DIR}")
        shutil.copytree(target, DATASET_DIR)
        print(f"[INFO] Dataset ready → {DATASET_DIR}")

    return DATASET_DIR
TEST_SIZE    = 0.15
VAL_SIZE     = 0.15
RANDOM_SEED  = 42


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_images_from_dataset(dataset_dir: str, max_per_class: int = MAX_PER_CLASS) -> tuple:
    """
    Walk leapGestRecog directory structure:
      leapGestRecog/<subject_num>/<gesture_name>/frame_*.png

    Discovers all unique gesture names, assigns integer class IDs,
    and loads up to max_per_class images per gesture.

    Returns
    -------
    X         : np.ndarray  shape (N, IMG_SIZE, IMG_SIZE, 1)  float32
    y         : np.ndarray  shape (N,)                         int
    label_map : dict  int → gesture name
    """
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"[ERROR] Dataset directory not found: {dataset_dir}")

    # Discover all unique gesture folder names across subjects
    # Gesture dirs have names like "01_palm", "02_l" etc (contain underscore)
    # Subject dirs are pure numbers like "00", "01" etc — skip those as class labels
    gesture_names = set()
    for subject_dir in os.listdir(dataset_dir):
        subject_path = os.path.join(dataset_dir, subject_dir)
        if os.path.isdir(subject_path):
            for gesture_dir in os.listdir(subject_path):
                gesture_path = os.path.join(subject_path, gesture_dir)
                # Only include dirs that look like gesture names (contain underscore)
                if os.path.isdir(gesture_path) and "_" in gesture_dir:
                    gesture_names.add(gesture_dir)

    gesture_names = sorted(gesture_names)   # e.g. ['01_palm','02_l','03_fist',...]
    label_map = {i: name for i, name in enumerate(gesture_names)}
    name_to_id = {name: i for i, name in enumerate(gesture_names)}

    print(f"[INFO] Gesture classes found: {len(gesture_names)}")
    for i, name in label_map.items():
        print(f"         Class {i:2d} → {name}")

    images, labels = [], []
    class_counts = {name: 0 for name in gesture_names}

    # Walk: subject → gesture
    for subject_dir in sorted(os.listdir(dataset_dir)):
        subject_path = os.path.join(dataset_dir, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        for gesture_dir in sorted(os.listdir(subject_path)):
            gesture_path = os.path.join(subject_path, gesture_dir)
            if not os.path.isdir(gesture_path) or "_" not in gesture_dir or gesture_dir not in name_to_id:
                continue
            class_id = name_to_id[gesture_dir]
            if class_counts[gesture_dir] >= max_per_class:
                continue
            for fname in sorted(os.listdir(gesture_path)):
                if class_counts[gesture_dir] >= max_per_class:
                    break
                if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                fpath = os.path.join(gesture_path, fname)
                img = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(resized)
                labels.append(class_id)
                class_counts[gesture_dir] += 1

    for name, count in class_counts.items():
        print(f"  {name:<20}: {count} images loaded")

    X = np.array(images, dtype=np.float32) / 255.0
    X = X[..., np.newaxis]   # (N, 64, 64, 1)
    y = np.array(labels, dtype=np.int32)

    print(f"\n[INFO] Dataset shape: X={X.shape}  y={y.shape}")
    return X, y, label_map


# ---------------------------------------------------------------------------
# EDA
# ---------------------------------------------------------------------------

def visualise_samples(X: np.ndarray, y: np.ndarray, label_map: dict, n_per_class: int = 5) -> None:
    """
    Display sample images for each gesture class.

    Saves → outputs/sample_gestures.png
    """
    set_plot_style()
    n_classes = len(label_map)
    fig, axes = plt.subplots(n_classes, n_per_class, figsize=(n_per_class * 2, n_classes * 2))
    fig.suptitle("Sample Gesture Images", fontsize=13, fontweight="bold")

    for cls in range(n_classes):
        indices = np.where(y == cls)[0][:n_per_class]
        for col, idx in enumerate(indices):
            ax = axes[cls, col]
            ax.imshow(X[idx, :, :, 0], cmap="gray")
            ax.axis("off")
            if col == 0:
                ax.set_ylabel(label_map[cls], fontsize=8, rotation=0,
                              labelpad=40, va="center")

    plt.tight_layout()
    save_figure(fig, "sample_gestures.png")


def plot_class_distribution(y: np.ndarray, label_map: dict) -> None:
    """Bar chart of class counts. Saves → outputs/class_distribution.png"""
    set_plot_style()
    unique, counts = np.unique(y, return_counts=True)
    names = [label_map[u] for u in unique]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(names, counts, color=sns.color_palette("tab10", n_colors=len(names)))
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=9)
    ax.set_title("Class Distribution", fontsize=13)
    ax.set_ylabel("Count")
    plt.xticks(rotation=30, ha="right")
    save_figure(fig, "class_distribution.png")


# ---------------------------------------------------------------------------
# Train / Val / Test split
# ---------------------------------------------------------------------------

def split_dataset(X: np.ndarray, y: np.ndarray, num_classes: int) -> tuple:
    """
    Split into train (70%), validation (15%), test (15%) sets.
    One-hot encode labels.

    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
    """
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(TEST_SIZE + VAL_SIZE), stratify=y, random_state=RANDOM_SEED
    )
    # Second split: val vs test
    rel_val = VAL_SIZE / (TEST_SIZE + VAL_SIZE)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - rel_val), stratify=y_temp, random_state=RANDOM_SEED
    )

    print(f"[INFO] Train: {len(X_train)}  Val: {len(X_val)}  Test: {len(X_test)}")

    # One-hot encode
    y_train_oh = to_categorical(y_train, num_classes)
    y_val_oh   = to_categorical(y_val,   num_classes)
    y_test_oh  = to_categorical(y_test,  num_classes)

    return X_train, X_val, X_test, y_train_oh, y_val_oh, y_test_oh, y_test


# ---------------------------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------------------------

def build_augmentation_generator(X_train: np.ndarray, y_train: np.ndarray) -> ImageDataGenerator:
    """
    Build a Keras ImageDataGenerator with standard augmentation for gesture images.

    Augmentation applied
    --------------------
    - Rotation ±15°
    - Horizontal / vertical shift ±10%
    - Zoom ±10%
    - Horizontal flip

    Returns
    -------
    ImageDataGenerator (fitted on training data)
    """
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.10,
        height_shift_range=0.10,
        zoom_range=0.10,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    datagen.fit(X_train)
    print("[INFO] Augmentation generator built and fitted on training data")
    return datagen


def plot_augmented_samples(X_train: np.ndarray, y_train: np.ndarray,
                           datagen: ImageDataGenerator, label_map: dict) -> None:
    """
    Show a few augmented versions of a single sample.

    Saves → outputs/augmented_samples.png
    """
    set_plot_style()
    sample = X_train[0:1]

    fig, axes = plt.subplots(1, 8, figsize=(16, 3))
    fig.suptitle("Data Augmentation Examples", fontsize=12, fontweight="bold")

    axes[0].imshow(sample[0, :, :, 0], cmap="gray")
    axes[0].set_title("Original", fontsize=9)
    axes[0].axis("off")

    for i, batch in enumerate(datagen.flow(sample, batch_size=1, seed=42)):
        if i >= 7:
            break
        axes[i + 1].imshow(batch[0, :, :, 0], cmap="gray")
        axes[i + 1].set_title(f"Aug {i+1}", fontsize=9)
        axes[i + 1].axis("off")

    save_figure(fig, "augmented_samples.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_preprocessing() -> tuple:
    """
    Full preprocessing pipeline.

    Returns
    -------
    X_train, X_val, X_test,
    y_train, y_val, y_test,     ← one-hot encoded
    y_test_raw,                 ← integer labels for evaluation
    datagen,
    num_classes,
    label_map
    """
    ensure_dir(outputs_path())

    print("[STEP] Downloading / locating dataset …")
    download_dataset()

    print("[STEP] Loading images …")
    X, y, label_map = load_images_from_dataset(DATASET_DIR)
    num_classes = len(label_map)

    print("[STEP] EDA — visualising samples …")
    visualise_samples(X, y, label_map)
    plot_class_distribution(y, label_map)

    print("[STEP] Splitting dataset …")
    X_train, X_val, X_test, y_train, y_val, y_test, y_test_raw = split_dataset(X, y, num_classes)

    print("[STEP] Building augmentation generator …")
    datagen = build_augmentation_generator(X_train, y_train)
    plot_augmented_samples(X_train, y_train, datagen, label_map)

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            y_test_raw, datagen, num_classes, label_map)
