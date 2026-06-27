# 🛣️ Pothole Detection System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-SVM%20Classifier-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Tkinter](https://img.shields.io/badge/Tkinter-Desktop%20GUI-1E90FF?style=for-the-badge&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-Feature%20Engineering-013243?style=for-the-badge&logo=numpy&logoColor=white)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

**An AI-powered desktop application for automated pothole detection in road surface images using SVM classification, HOG feature descriptors, and a production-grade dark-themed Tkinter GUI — no cloud, no deep learning infrastructure required.**

</div>

---

## 📖 Overview

The **Pothole Detection System** is a standalone desktop application that leverages classical machine learning and computer vision techniques to classify road images as either **POTHOLE** or **CLEAR**. Built entirely in Python, it is designed for road maintenance authorities, transportation agencies, and infrastructure researchers who need an accessible, offline AI tool without requiring specialized data science expertise.

The system addresses three critical real-world problems:
- 🕳️ **Manual road inspections** are slow and expensive — solved via automated image-based AI classification
- 💸 **Billions lost annually** in vehicle damage caused by undetected potholes — solved with early automated detection
- 🖥️ **Deep learning infrastructure is inaccessible** to local authorities — solved via a lightweight SVM-based pipeline that runs on any Python machine

---

## ✨ Core Features

| Feature | Description |
|---------|-------------|
| 🧠 SVM Classifier | RBF-kernel Support Vector Machine with balanced class weights |
| 🔍 HOG Feature Extraction | Histogram of Oriented Gradients — shape and texture gradient encoding |
| 📐 Multi-Signal Feature Vector | 6 feature groups combined: HOG, Laplacian, Canny edges, intensity stats, dark ratio, variance grid |
| ⚙️ One-Click Training | Folder-based dataset loader — auto-detects class labels from subfolder names |
| 💾 Model Persistence | Trained model + scaler + label map saved as `.pkl` for instant reuse |
| 🖥️ Dark-Themed Desktop UI | GitHub-inspired Tkinter dashboard with live clock, stat cards, and progress bar |
| ⚡ Multi-Threaded Training | Background thread keeps the UI fully responsive during long training runs |
| 📊 Live Statistics Dashboard | 4-stat card row: model status, images processed, classes detected, scans count |

---

## 🧠 Feature Engineering Pipeline

Each image is resized to **128×128 pixels** and converted to grayscale before the following 6 feature groups are extracted and concatenated into a single feature vector of **~2937 dimensions**:

| S.No | Feature Group | Description | Dimensions |
|---|---------------|-------------|-----------|
| 1 | **HOG Descriptor** | Histogram of Oriented Gradients (16×16 blocks, 8×8 stride, 9 bins) — captures shape and texture gradients, L2-normalized | ~2916 |
| 2 | **Laplacian Variance** | Second-derivative variance of pixel intensity — measures surface roughness | 1 |
| 3 | **Canny Edge Density** | Fraction of Canny edge pixels (thresholds 50–150) — potholes produce strong irregular boundaries | 1 |
| 4 | **Intensity Stats** | Mean and standard deviation of pixel intensity (normalized 0–1) | 2 |
| 5 | **Dark Region Ratio** | Proportion of pixels below intensity threshold 80 — captures pothole shadow regions | 1 |
| 6 | **Local Variance Grid** | 4×4 spatial grid of per-cell variance — detects spatially uneven surface distribution | 16 |

---

## 🛠️ Tech Stack

### Machine Learning & Computer Vision
| Technology | Purpose |
|------------|---------|
| scikit-learn | SVM classifier (`SVC`, RBF kernel) + `StandardScaler` |
| OpenCV (cv2) | HOG extraction, Canny edge detection, Laplacian, image I/O |
| NumPy | Array operations, feature vector concatenation |
| Pillow (PIL) | Image preview and thumbnail rendering in the UI |

### GUI & Runtime
| Technology | Purpose |
|------------|---------|
| Tkinter + ttk | Desktop GUI framework with dark theme |
| Python threading | Background training thread (non-blocking UI) |
| ttk.Progressbar | Real-time training progress display |
| pickle | Model serialization and reload |

---

## 🖥️ Application UI

The desktop dashboard is organized into four functional zones:

| Zone | Component | Functionality |
|------|-----------|---------------|
| **Header Bar** | Title + Live Clock + Badge | System branding, live date/time, `● SYSTEM LIVE` indicator |
| **Stats Row** | 4 Stat Cards | Model status, images processed, class count, scan count |
| **Train Model Panel** | Card 1 | Dataset folder browser, Train button, progress bar, status label |
| **Analyze Image Panel** | Card 2 | Image upload, road preview thumbnail, Choose Image button |
| **Detection Result Panel** | Card 3 | Result icon, CLEAR / POTHOLE label, image name, status badge |
| **Status Bar** | Footer Label | Global operation feedback with color-coded messages |

---

## 🚀 Installation Guide

### 1. Clone the Repository
```bash
git clone https://github.com/EswariSankar/pothole-detection-system
cd pothole-detection-system
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Application
```bash
python sourcecode.py
```

---

## 📦 Requirements

Create a `requirements.txt` with the following:

```
opencv-python
scikit-learn
numpy
Pillow
```

> **Note:** `tkinter` is bundled with standard Python installations. No separate install needed.

---

## 📂 Project Structure

```
pothole-detection-system/
├── sourcecode.py            # Main application (GUI + ML pipeline)
├── pothole_model.pkl        # Trained model artifact (generated after first training)
├── requirements.txt         # Python dependencies
└── README.md
```

### Dataset Folder Format

The training dataset must be organized as follows — the app **auto-detects class labels from subfolder names**:

```
dataset/
├── Pothole/              # Images of road potholes
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
└── NoPothole/            # Images of clear road surfaces
    ├── img_001.jpg
    ├── img_002.jpg
    └── ...
```

> Folder names are flexible — the app uses keyword matching (`pothole` in name) to assign labels automatically.

---

## ⚙️ How It Works

```
Dataset Folder → Auto-detect Classes → Extract Features (6 groups) → StandardScaler
     → Train SVM (RBF, balanced) → Save .pkl → Load Image → Extract Features
     → Predict: POTHOLE / CLEAR → Display Result
```

1. **Load Dataset** — Select a folder with class subfolders via the Browse button
2. **Extract Features** — Each image is processed through the 6-group feature pipeline
3. **Train SVM** — RBF SVM trained in a background thread; progress bar updates in real time
4. **Save Model** — Model, scaler, and label map pickled to `pothole_model.pkl`
5. **Detect Potholes** — Upload any road image for instant POTHOLE / CLEAR classification

---

## 🎨 UI Design Tokens

| Token | Color | Usage |
|-------|-------|-------|
| `BG_MAIN` | `#0D1117` | Main window background |
| `BG_CARD` | `#161B22` | Card and panel backgrounds |
| `ACCENT` | `#F78166` | Icon and highlight accents |
| `SUCCESS` | `#3FB950` | CLEAR result, ready state |
| `DANGER` | `#F85149` | POTHOLE result, error state |
| `INFO` | `#58A6FF` | Analyze panel icon, clock |
| `TEXT_PRIMARY` | `#E6EDF3` | Main text |
| `TEXT_MUTED` | `#8B949E` | Secondary text, labels |

---

## ⚠️ Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Imbalanced dataset | High | `class_weight='balanced'` in SVM |
| Overfitting on small datasets | Medium | StandardScaler normalization + RBF kernel with `gamma='scale'` |
| Slow training on large datasets | Medium | Multi-threading keeps UI responsive |
| Missing or corrupt model file | Low | Graceful error dialogs; model is re-trainable at any time |

---

## 👩‍💻 Author

**Eswari Sankar**  
B.E. Information Technology  
Annamalai University, Chidambaram

[![GitHub](https://img.shields.io/badge/GitHub-EswariSankar-181717?style=flat&logo=github)](https://github.com/EswariSankar)

---

## 📄 License

This project is developed for **academic and learning purposes**.

---

