# Breast-cancer

### Cancer Detection using Deep Learning (Adapted from Potato Disease Classifier)

## 📌 Overview

This project is an adaptation of the **Potato Disease Classification Deep Learning Project** by [CodeBasics](https://www.youtube.com/playlist?list=PLeo1K3hjS3ut49PskOfLnE6WUoOp_2lsD), originally designed to detect diseases in potato leaves using Convolutional Neural Networks (CNNs) with TensorFlow.

Instead of classifying potato diseases, we restructured and trained the model to **classify breast cancer images** using histopathological scans. The primary objective is to detect whether a given tissue image indicates **malignant** or **non-malignant** cancer.

---

## 🔍 Problem Statement

Breast cancer is one of the most common cancers among women globally. Early detection through histopathological image analysis can significantly improve patient outcomes. This project uses deep learning to automate the classification process.

---

## 🧠 Original Project

* **Tutorial Source**: [Potato Disease Classification by CodeBasics](https://www.youtube.com/playlist?list=PLeo1K3hjS3ut49PskOfLnE6WUoOp_2lsD)
* **Framework**: TensorFlow / Keras
* **Techniques Used**:

  * Image Preprocessing
  * CNNs for classification
  * Transfer Learning (optional)
  * Model Evaluation and Visualization

---

## 🧬 Modified Dataset

* **Dataset Used**: [Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
* **Description**: Microscopic images of breast tissue, labeled as:

  * `0`: No cancer (non-malignant)
  * `1`: Cancer (malignant)
* **Image Format**: `.png`, 50x50 patches

---

<!--
## 🔧 How We Adapted the Project

| Component              | Original (Potato)                  | Modified (Cancer)                       |
| ---------------------- | ---------------------------------- | --------------------------------------- |
| **Dataset**            | Potato leaf disease images         | Breast cancer histopathology images     |
| **Classes**            | Healthy, Early Blight, Late Blight | Malignant, Non-Malignant                |
| **Preprocessing**      | Image resizing, normalization      | Image patch selection, normalization    |
| **Model Architecture** | CNN using TensorFlow/Keras         | Same CNN architecture with minor tuning |
| **Evaluation**         | Accuracy, Confusion Matrix         | Accuracy, Confusion Matrix, ROC-AUC     |

## 📁 Directory Structure

```bash
.
├── data/                         # Dataset directory (after preprocessing)
├── notebooks/                   # Jupyter notebooks for EDA & modeling
├── src/                         # Source code files
│   ├── model.py                 # CNN architecture
│   ├── train.py                 # Training script
│   └── predict.py               # Prediction & inference
├── README.md
├── requirements.txt
└── cancer_detection.ipynb       # Main notebook (if not modularized)
```



## 🛠️ Installation

1. Clone the repository:

   ```bash
   git clone <your-repo-url>
   cd <your-repo-folder>
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Prepare the dataset:

   * Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)
   * Extract and place into the `data/` directory

---

## 🚀 How to Run

```bash
# Train the model
python src/train.py

# Predict on a new image
python src/predict.py --image path/to/image.png
```

Or use the notebook `cancer_detection.ipynb` to interactively train and test.

---



## 📈 Results

| Metric           | Value              |
| ---------------- | ------------------ |
| Accuracy         | \~XX%              |
| ROC-AUC          | \~XX               |
| Precision/Recall | Included in report |

*(Fill in actual metrics after training your model.)*
-->

## 🙌 Acknowledgements

* [CodeBasics YouTube Playlist](https://www.youtube.com/playlist?list=PLeo1K3hjS3ut49PskOfLnE6WUoOp_2lsD)
* [Kaggle Dataset - Breast Histopathology Images](https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images)


