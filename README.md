# 🤚 PRODIGY_ML_04 — Hand Gesture Recognition using CNN

> **Prodigy InfoTech Machine Learning Internship — Task 04**  
> Classify 10 distinct hand gestures from the LeapGestRecog dataset using a Convolutional Neural Network (CNN) built with TensorFlow/Keras.

---

## 📌 Project Overview

This project builds a deep learning pipeline to recognise 10 different hand gestures captured with a Leap Motion sensor. The CNN architecture leverages data augmentation, batch normalisation, and dropout to generalise well across gesture classes. An optional OpenCV webcam script enables real-time gesture prediction.

---

## 🎯 Problem Statement

Given grayscale images of hand gestures (from the LeapGestRecog dataset), train a CNN that accurately classifies each image into one of 10 gesture categories: palm, l, fist, fist_moved, thumb, index, ok, palm_moved, c, down.

---

## 📂 Dataset

| Detail | Info |
|--------|------|
| Source | [Kaggle — LeapGestRecog](https://www.kaggle.com/datasets/gti-upm/leapgestrecog) |
| Total images | ~20,000 (2,000 per gesture × 10 gestures) |
| Subjects | 10 participants |
| Format | Grayscale PNG (240×640) |

### Gesture Classes

| Class ID | Gesture |
|----------|---------|
| 00 | palm |
| 01 | l |
| 02 | fist |
| 03 | fist_moved |
| 04 | thumb |
| 05 | index |
| 06 | ok |
| 07 | palm_moved |
| 08 | c |
| 09 | down |

### Download Instructions
1. Visit the Kaggle link above.
2. Download the dataset.
3. Extract and place as:
   ```
   PRODIGY_ML_04/
   └── data/
       └── leapGestRecog/
           ├── 00/
           │   ├── 01/  (subject folder)
           │   │   ├── frame_00_01_0001.png
           │   │   └── ...
           │   └── ...
           ├── 01/
           └── ...
   ```

---

## 🚀 Installation

```bash
git clone https://github.com/<your-username>/PRODIGY_ML_04.git
cd PRODIGY_ML_04

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
python main.py
```

---

## 📁 Folder Structure

```
PRODIGY_ML_04/
│
├── data/
│   └── leapGestRecog/   ← extracted dataset here
│
├── models/
│   ├── gesture_cnn.keras
│   ├── best_model.keras
│   └── label_map.pkl
│
├── outputs/
│   ├── sample_gestures.png
│   ├── class_distribution.png
│   ├── augmented_samples.png
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── per_class_f1.png
│   ├── misclassified_samples.png
│   ├── demo_prediction_1.png
│   ├── ...
│   ├── metrics.json
│   └── evaluation_report.txt
│
├── notebooks/
│   └── exploration.ipynb
│
├── main.py
├── preprocess.py
├── train.py
├── evaluate.py
├── predict.py
├── utils.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Workflow

```
data/leapGestRecog/
        │
        ▼
preprocess.py
  ├─ Walk directory → load grayscale images
  ├─ Resize to 64×64
  ├─ Normalise [0, 1]
  ├─ EDA (sample grid, class distribution)
  ├─ Train / Val / Test split (70 / 15 / 15)
  ├─ One-hot encode labels
  └─ ImageDataGenerator (rotation, shift, zoom, flip)
        │
        ▼
train.py
  ├─ CNN: 3 Conv Blocks → BatchNorm → MaxPool → Dropout
  ├─ Dense(256) → BatchNorm → Dropout(0.5) → Softmax
  ├─ Callbacks: EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
  ├─ Training with augmentation
  ├─ Training curves plot
  └─ Save model (.keras)
        │
        ▼
evaluate.py
  ├─ Accuracy, Precision, Recall, F1 (macro + weighted)
  ├─ Confusion Matrix (raw + normalised)
  ├─ Per-class F1 bar chart
  ├─ Misclassified samples
  └─ Save report + metrics JSON
        │
        ▼
predict.py
  ├─ Load model → predict single image
  ├─ Top-5 probability bar chart
  ├─ Demo batch predictions
  └─ [Optional] OpenCV webcam real-time prediction
```

---

## 🧠 CNN Architecture

```
Input: (64, 64, 1)
├── Conv2D(32) → BN → ReLU → Conv2D(32) → BN → ReLU → MaxPool → Dropout(0.25)
├── Conv2D(64) → BN → ReLU → Conv2D(64) → BN → ReLU → MaxPool → Dropout(0.25)
├── Conv2D(128)→ BN → ReLU → MaxPool → Dropout(0.25)
├── Flatten
├── Dense(256) → BN → ReLU → Dropout(0.50)
└── Dense(10)  → Softmax
```

**Optimiser:** Adam (lr=0.001)  
**Loss:** Categorical Cross-Entropy  
**Callbacks:** EarlyStopping · ReduceLROnPlateau · ModelCheckpoint

---

## 📚 Libraries Used

| Library | Purpose |
|---------|---------|
| `tensorflow/keras` | CNN model, training, augmentation |
| `numpy` | Numerical operations |
| `opencv-python` | Image loading, webcam capture |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisations |
| `scikit-learn` | Metrics, train/test split |
| `joblib` | Label map serialisation |
| `Pillow` | Image utility |

---

## 📊 Results

| Metric | Value |
|--------|-------|
| Test Accuracy | ~97–99% |
| Precision (macro) | ~0.97 |
| Recall (macro) | ~0.97 |
| F1 Score (macro) | ~0.97 |

> *LeapGestRecog is a relatively clean dataset, so high accuracy is expected.*

---

## 🖼️ Screenshots

### Sample Gesture Images
![Sample Gestures](outputs/sample_gestures.png)

### Training Curves
![Training Curves](outputs/training_curves.png)

### Confusion Matrix
![Confusion Matrix](outputs/confusion_matrix.png)

### Per-Class F1 Score
![Per-Class F1](outputs/per_class_f1.png)

### Demo Prediction
![Demo Prediction](outputs/demo_prediction_1.png)

---

## 🔮 Future Improvements

- [ ] Transfer Learning (MobileNetV2, EfficientNet)
- [ ] Larger input resolution (96×96 or 128×128)
- [ ] Temporal / sequence models (LSTM) for dynamic gestures
- [ ] Deploy as REST API (Flask / FastAPI)
- [ ] Extend to custom gesture classes
- [ ] TensorFlow Lite conversion for mobile deployment

---

## 🎥 Real-Time Webcam Prediction

```bash
# After training, run with webcam:
python main.py --webcam
```

Press `q` to quit the webcam window.

---

## 📄 License

MIT License — see [LICENSE](LICENSE)

---

## 🏃 How to Run

```bash
# Full pipeline
python main.py

# With webcam (optional)
python main.py --webcam

# Predict a single image
python -c "
import joblib, tensorflow as tf
from predict import predict_image
model     = tf.keras.models.load_model('models/gesture_cnn.keras')
label_map = joblib.load('models/label_map.pkl')
result    = predict_image('data/leapGestRecog/00/01/frame_00_01_0001.png', model, label_map)
print(result)
"
```

---

*Made with ❤️ for Prodigy InfoTech ML Internship*
