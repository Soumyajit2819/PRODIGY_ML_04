"""
main.py
-------
Entry point for the Hand Gesture Recognition CNN pipeline.

Run
---
    python main.py

    # For real-time webcam prediction (optional):
    python main.py --webcam

Steps executed
--------------
1. Preprocessing  — load images, augmentation, split
2. Training       — build CNN, train, save model
3. Evaluation     — metrics, confusion matrix, training curves
4. Prediction     — demo predictions on sample images
"""

import sys
import joblib

from preprocess import run_preprocessing
from train      import run_training
from evaluate   import run_evaluation
from predict    import (
    run_demo_predictions,
    run_webcam_prediction,
    load_model,
)
from utils import models_path, outputs_path, ensure_dir


def main() -> None:
    webcam_mode = "--webcam" in sys.argv

    print("=" * 65)
    print("  PRODIGY_ML_04 — Hand Gesture Recognition (CNN)")
    print("=" * 65)

    # ------------------------------------------------------------------
    # Step 1: Preprocessing
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 1/4 — Preprocessing …")
    ensure_dir(outputs_path())
    ensure_dir(models_path())

    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     y_test_raw, datagen,
     num_classes, label_map) = run_preprocessing()

    # Persist label map for standalone predict.py usage
    joblib.dump(label_map, models_path("label_map.pkl"))
    print(f"[INFO] Label map saved → {models_path('label_map.pkl')}")

    # ------------------------------------------------------------------
    # Step 2: Training
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 2/4 — Training CNN …")
    model, history = run_training(
        num_classes, X_train, y_train, X_val, y_val, datagen
    )

    # ------------------------------------------------------------------
    # Step 3: Evaluation
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 3/4 — Evaluation …")
    metrics = run_evaluation(model, X_test, y_test, y_test_raw, label_map)

    # ------------------------------------------------------------------
    # Step 4: Demo Predictions
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Step 4/4 — Demo Predictions …")
    run_demo_predictions(model, label_map)

    # ------------------------------------------------------------------
    # Optional: Webcam real-time prediction
    # ------------------------------------------------------------------
    if webcam_mode:
        print("\n[PIPELINE] Optional — Webcam Real-Time Prediction …")
        run_webcam_prediction(model, label_map)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 65)
    print("  Pipeline complete!")
    print(f"  Accuracy         : {metrics['Accuracy']}")
    print(f"  Precision (macro): {metrics['Precision (macro)']}")
    print(f"  Recall (macro)   : {metrics['Recall (macro)']}")
    print(f"  F1 Score (macro) : {metrics['F1 Score (macro)']}")
    print(f"  Num Classes      : {num_classes}")
    print("\n  Saved artefacts:")
    print("    models/gesture_cnn.keras")
    print("    models/best_model.keras")
    print("    models/label_map.pkl")
    print("    outputs/ — plots + evaluation report + metrics JSON")
    print("=" * 65)
    if not webcam_mode:
        print("\n  Tip: Run `python main.py --webcam` for real-time prediction")


if __name__ == "__main__":
    main()
