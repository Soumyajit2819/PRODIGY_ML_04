"""
utils.py
--------
Shared helper functions for the Hand Gesture Recognition project.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

def get_project_root() -> str:
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(directory: str) -> None:
    os.makedirs(directory, exist_ok=True)


def data_path(*parts: str) -> str:
    return os.path.join(get_project_root(), "data", *parts)


def models_path(*parts: str) -> str:
    return os.path.join(get_project_root(), "models", *parts)


def outputs_path(*parts: str) -> str:
    return os.path.join(get_project_root(), "outputs", *parts)


# ---------------------------------------------------------------------------
# Dataset check
# ---------------------------------------------------------------------------

def check_dataset_exists(directory: str) -> None:
    if not os.path.isdir(directory):
        raise FileNotFoundError(
            f"\n[ERROR] Dataset not found: {directory}\n"
            "Please download the dataset from Kaggle:\n"
            "  https://www.kaggle.com/datasets/gti-upm/leapgestrecog\n"
            "Extract and place the folder as: data/leapGestRecog/\n"
            "Expected structure:\n"
            "  data/leapGestRecog/00/\n"
            "  data/leapGestRecog/01/\n"
            "  ... (10 gesture folders, each with subject subfolders)\n"
        )


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, filename: str) -> None:
    ensure_dir(outputs_path())
    filepath = outputs_path(filename)
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[INFO] Figure saved → {filepath}")


def set_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "#f9f9f9",
        "axes.grid": True,
        "grid.color": "#dddddd",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "font.size": 11,
    })


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def save_metrics(metrics: dict, filename: str = "metrics.json") -> None:
    ensure_dir(outputs_path())
    filepath = outputs_path(filename)
    with open(filepath, "w") as fh:
        json.dump(metrics, fh, indent=4)
    print(f"[INFO] Metrics saved → {filepath}")


def print_metrics(metrics: dict, title: str = "Evaluation Metrics") -> None:
    border = "=" * 50
    print(f"\n{border}")
    print(f"  {title}")
    print(border)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:<30}: {value:.4f}")
        else:
            print(f"  {key:<30}: {value}")
    print(f"{border}\n")


# ---------------------------------------------------------------------------
# Label map builder
# ---------------------------------------------------------------------------

def build_label_map(class_dirs: list) -> dict:
    """
    Build an integer → gesture name mapping from sorted class directory names.

    The leapGestRecog dataset uses folder names like:
      00/ → palm
      01/ → l
      02/ → fist
      ...

    Returns
    -------
    dict  { int_class_id : gesture_name_string }
    """
    GESTURE_NAMES = {
        "00": "palm",
        "01": "l",
        "02": "fist",
        "03": "fist_moved",
        "04": "thumb",
        "05": "index",
        "06": "ok",
        "07": "palm_moved",
        "08": "c",
        "09": "down",
    }
    label_map = {}
    for idx, dirname in enumerate(sorted(class_dirs)):
        key = dirname.zfill(2)
        label_map[idx] = GESTURE_NAMES.get(key, dirname)
    return label_map
