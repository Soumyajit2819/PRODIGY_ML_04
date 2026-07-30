"""
predict.py
----------
Load the saved CNN model and predict the gesture class of new images.
Optionally use OpenCV webcam for real-time prediction.
"""

import os
import sys
import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import tensorflow as tf

from utils import models_path, outputs_path, ensure_dir, save_figure, set_plot_style
from preprocess import IMG_SIZE


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_model(filename: str = "gesture_cnn.keras") -> tf.keras.Model:
    filepath = models_path(filename)
    if not os.path.exists(filepath):
        # Try best_model.keras fallback
        best = models_path("best_model.keras")
        if os.path.exists(best):
            filepath = best
        else:
            raise FileNotFoundError(
                f"[ERROR] Model not found: {filepath}\n"
                "Run main.py first to train and save the model."
            )
    model = tf.keras.models.load_model(filepath)
    print(f"[INFO] CNN model loaded ← {filepath}")
    return model


# ---------------------------------------------------------------------------
# Preprocessing for prediction
# ---------------------------------------------------------------------------

def preprocess_single_image(image_path: str) -> np.ndarray:
    """
    Load and preprocess a single image for prediction.

    Returns
    -------
    np.ndarray  shape (1, IMG_SIZE, IMG_SIZE, 1)  float32
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"[ERROR] Could not read image: {image_path}")
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    norm    = resized.astype(np.float32) / 255.0
    return norm[np.newaxis, :, :, np.newaxis]      # (1, 64, 64, 1)


def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Preprocess a single OpenCV BGR frame for prediction.

    Returns
    -------
    np.ndarray  shape (1, IMG_SIZE, IMG_SIZE, 1)
    """
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    norm    = resized.astype(np.float32) / 255.0
    return norm[np.newaxis, :, :, np.newaxis]


# ---------------------------------------------------------------------------
# Single image prediction
# ---------------------------------------------------------------------------

def predict_image(image_path: str, model: tf.keras.Model,
                  label_map: dict) -> dict:
    """
    Predict gesture class from a file path.

    Returns
    -------
    dict with keys: label, class_id, confidence, all_probs
    """
    X         = preprocess_single_image(image_path)
    probs     = model.predict(X, verbose=0)[0]
    class_id  = int(np.argmax(probs))
    label     = label_map.get(class_id, str(class_id))
    confidence = float(probs[class_id])
    return {
        "label":      label,
        "class_id":   class_id,
        "confidence": confidence,
        "all_probs":  {label_map.get(i, str(i)): float(p) for i, p in enumerate(probs)},
    }


# ---------------------------------------------------------------------------
# Visualise prediction
# ---------------------------------------------------------------------------

def visualise_prediction(image_path: str, result: dict, filename: str) -> None:
    """
    Display the image, predicted label, and top-5 probability bar chart.

    Saves → outputs/<filename>
    """
    set_plot_style()
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Top-5 probabilities
    all_probs = result["all_probs"]
    top5      = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)[:5]
    labels_5  = [t[0] for t in top5]
    scores_5  = [t[1] for t in top5]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(
        f"Predicted: {result['label'].upper()}\n({result['confidence']*100:.1f}%)",
        fontsize=11, color="#4C72B0", fontweight="bold"
    )
    axes[0].axis("off")

    colors = ["#4C72B0" if l == result["label"] else "#AAAAAA" for l in labels_5]
    axes[1].barh(labels_5[::-1], scores_5[::-1], color=colors[::-1])
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title("Top-5 Class Probabilities")

    plt.tight_layout()
    save_figure(fig, filename)


# ---------------------------------------------------------------------------
# Demo predictions
# ---------------------------------------------------------------------------

def run_demo_predictions(model: tf.keras.Model, label_map: dict,
                          test_image_paths: list = None) -> None:
    """Predict gestures on sample images from the dataset."""
    import random
    from preprocess import DATASET_DIR

    if test_image_paths is None:
        # Collect a few random images from the dataset
        all_images = []
        if os.path.isdir(DATASET_DIR):
            for root, _, files in os.walk(DATASET_DIR):
                for f in files:
                    if f.lower().endswith((".png", ".jpg")):
                        all_images.append(os.path.join(root, f))
        random.seed(42)
        test_image_paths = random.sample(all_images, min(8, len(all_images)))

    if not test_image_paths:
        print("[WARN] No images found for demo. Skipping.")
        return

    ensure_dir(outputs_path())
    print("\n" + "=" * 65)
    print("  Demo Predictions — Hand Gesture Recognition")
    print("=" * 65)

    for i, path in enumerate(test_image_paths):
        try:
            result  = predict_image(path, model, label_map)
            # Extract true label from folder name
            parts   = path.replace("\\", "/").split("/")
            true_label = parts[-3] if len(parts) >= 3 else "unknown"
            true_gesture = label_map.get(int(true_label), true_label) if true_label.isdigit() else true_label
            correct = "✓" if result["label"].startswith(true_gesture[:4]) else "✗"
            print(
                f"  [{i+1}] {os.path.basename(path):<25}  "
                f"Predicted: {result['label']:<15}  "
                f"Conf: {result['confidence']*100:5.1f}%  {correct}"
            )
            visualise_prediction(path, result, f"demo_prediction_{i+1}.png")
        except Exception as e:
            print(f"  [WARN] Skipped {path}: {e}")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Optional: Real-time OpenCV webcam prediction
# ---------------------------------------------------------------------------

def run_webcam_prediction(model: tf.keras.Model, label_map: dict) -> None:
    """
    Optional real-time gesture prediction using the webcam.

    Press 'q' to quit.

    NOTE: This requires a working webcam and a windowed display environment.
          Not available in headless / server environments.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] Webcam not available. Skipping real-time prediction.")
        return

    print("[INFO] Starting webcam prediction. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Crop a central ROI for the hand
        h, w = frame.shape[:2]
        roi = frame[h//4 : 3*h//4, w//4 : 3*w//4]

        X         = preprocess_frame(roi)
        probs     = model.predict(X, verbose=0)[0]
        class_id  = int(np.argmax(probs))
        label     = label_map.get(class_id, str(class_id))
        conf      = float(probs[class_id])

        # Draw bounding box and label on original frame
        cv2.rectangle(frame, (w//4, h//4), (3*w//4, 3*h//4), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label}  {conf*100:.1f}%",
            (w//4, h//4 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9,
            (0, 255, 0), 2
        )
        cv2.imshow("Hand Gesture Recognition — press Q to quit", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Webcam prediction stopped.")
