"""
evaluate.py
-----------
Evaluation metrics, confusion matrix, and per-class analysis
for the Hand Gesture Recognition CNN.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from utils import (
    outputs_path,
    ensure_dir,
    save_metrics,
    print_metrics,
    save_figure,
    set_plot_style,
)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    num_classes: int) -> dict:
    """
    Compute Accuracy, Precision, Recall, F1 (macro and per-class).

    Parameters
    ----------
    y_true      : integer ground-truth labels
    y_pred      : integer predicted labels
    num_classes : total number of gesture classes

    Returns
    -------
    dict
    """
    metrics = {
        "Accuracy":            round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision (macro)":   round(float(precision_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "Recall (macro)":      round(float(recall_score(y_true, y_pred,    average="macro", zero_division=0)), 4),
        "F1 Score (macro)":    round(float(f1_score(y_true, y_pred,        average="macro", zero_division=0)), 4),
        "Precision (weighted)":round(float(precision_score(y_true, y_pred, average="weighted", zero_division=0)), 4),
        "Recall (weighted)":   round(float(recall_score(y_true, y_pred,    average="weighted", zero_division=0)), 4),
        "F1 Score (weighted)": round(float(f1_score(y_true, y_pred,        average="weighted", zero_division=0)), 4),
    }
    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          label_map: dict) -> None:
    """
    Normalised + raw confusion matrix.

    Saves → outputs/confusion_matrix.png
    """
    set_plot_style()
    class_names = [label_map[i] for i in range(len(label_map))]
    cm          = confusion_matrix(y_true, y_pred)
    cm_norm     = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle("Confusion Matrix — Hand Gesture Recognition", fontsize=13, fontweight="bold")

    for ax, data, title, fmt in [
        (axes[0], cm,      "Raw Counts",     "d"),
        (axes[1], cm_norm, "Normalised (%)", ".2%"),
    ]:
        sns.heatmap(
            data, annot=True, fmt=fmt, cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            linewidths=0.4, ax=ax,
        )
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("True", fontsize=10)
        ax.set_title(title)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=9)
        plt.setp(ax.get_yticklabels(), rotation=0,  fontsize=9)

    plt.tight_layout()
    save_figure(fig, "confusion_matrix.png")


def plot_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                           label_map: dict) -> None:
    """
    Bar chart of per-class F1 scores.

    Saves → outputs/per_class_f1.png
    """
    set_plot_style()
    class_names = [label_map[i] for i in range(len(label_map))]
    f1_scores   = f1_score(y_true, y_pred, average=None, zero_division=0)

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(class_names, f1_scores,
                  color=sns.color_palette("tab10", n_colors=len(class_names)))
    for bar, score in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-Class F1 Score", fontsize=13, fontweight="bold")
    ax.set_ylabel("F1 Score")
    plt.xticks(rotation=30, ha="right")
    save_figure(fig, "per_class_f1.png")


def plot_top_misclassifications(y_true: np.ndarray, y_pred: np.ndarray,
                                X_test: np.ndarray, label_map: dict,
                                n: int = 10) -> None:
    """
    Show misclassified test images (wrong prediction shown in title).

    Saves → outputs/misclassified_samples.png
    """
    set_plot_style()
    wrong = np.where(y_true != y_pred)[0][:n]
    if len(wrong) == 0:
        print("[INFO] No misclassifications found!")
        return

    cols = min(n, len(wrong))
    fig, axes = plt.subplots(1, cols, figsize=(cols * 2.5, 3))
    if cols == 1:
        axes = [axes]
    fig.suptitle("Misclassified Samples", fontsize=12, fontweight="bold")

    for ax, idx in zip(axes, wrong):
        ax.imshow(X_test[idx, :, :, 0], cmap="gray")
        ax.set_title(
            f"T:{label_map[y_true[idx]]}\nP:{label_map[y_pred[idx]]}",
            fontsize=8, color="#C44E52"
        )
        ax.axis("off")

    plt.tight_layout()
    save_figure(fig, "misclassified_samples.png")


# ---------------------------------------------------------------------------
# Text report
# ---------------------------------------------------------------------------

def save_evaluation_report(metrics: dict, y_true: np.ndarray,
                            y_pred: np.ndarray, label_map: dict) -> None:
    ensure_dir(outputs_path())
    filepath    = outputs_path("evaluation_report.txt")
    class_names = [label_map[i] for i in range(len(label_map))]
    report      = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)

    with open(filepath, "w") as fh:
        fh.write("=" * 60 + "\n")
        fh.write("  Hand Gesture Recognition CNN — Evaluation Report\n")
        fh.write("=" * 60 + "\n\n")
        fh.write("Test Set Metrics\n")
        fh.write("-" * 40 + "\n")
        for k, v in metrics.items():
            fh.write(f"  {k:<30}: {v}\n")
        fh.write("\nClassification Report\n")
        fh.write("-" * 40 + "\n")
        fh.write(report)
    print(f"[INFO] Evaluation report saved → {filepath}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_evaluation(model, X_test: np.ndarray,
                   y_test_oh: np.ndarray, y_test_raw: np.ndarray,
                   label_map: dict) -> dict:
    """
    Full evaluation workflow.

    Returns
    -------
    metrics : dict
    """
    print("\n[STEP] Generating predictions …")
    y_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    num_classes = len(label_map)
    metrics     = compute_metrics(y_test_raw, y_pred, num_classes)
    print_metrics(metrics, title="Test Set Evaluation Metrics")
    save_metrics(metrics)

    print("[STEP] Generating plots …")
    plot_confusion_matrix(y_test_raw, y_pred, label_map)
    plot_per_class_metrics(y_test_raw, y_pred, label_map)
    plot_top_misclassifications(y_test_raw, y_pred, X_test, label_map)

    save_evaluation_report(metrics, y_test_raw, y_pred, label_map)

    return metrics
