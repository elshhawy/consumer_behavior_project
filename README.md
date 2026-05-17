# 🛒 Consumer Behavior Prediction & Precision Marketing

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange?logo=scikit-learn)
![TensorFlow](https://img.shields.io/badge/DL-TensorFlow-red?logo=tensorflow)
![XGBoost](https://img.shields.io/badge/Boosting-XGBoost-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)

> A full end-to-end Machine Learning project that predicts online shoppers'
> purchase intention and enables precision marketing — replicating and extending
> Lin (2025), published in PLoS ONE.

---

## 📌 Overview

This project is built in **two phases**:

**Phase 1 — Baseline ML Models (Replication of Lin, 2025)**
Reproducing the original paper's pipeline using four machine learning models:
SVM, XGBoost, CatBoost, and BPANN on the UCI Online Shoppers dataset.

**Phase 2 — Hybrid Adaptive Framework (Novel Contribution)**
A three-layer intelligent system that goes beyond static prediction:
- Deep Learning for complex behavioral pattern extraction
- K-means Clustering for automated customer segmentation  
- Q-learning (Reinforcement Learning) for real-time adaptive
  marketing decisions

---

## 🧠 Why This Project?

Traditional marketing models like the 4P framework struggle with
real-time, high-dimensional consumer data. This project addresses
that gap by building a dynamic system that:

- ✅ Predicts purchase intent with >93% accuracy
- ✅ Segments customers automatically based on behavior
- ✅ Adapts marketing decisions in real time using RL
- ✅ Validates all results with 10-fold Cross-Validation

---

## 🗂️ Project Structure
```
consumer-behavior-ml/
│
├── data/
│   └── online_shoppers_intention.csv   # UCI Dataset
│
├── src/
│   ├── preprocessing.py      # Cleaning, SMOTE, encoding, scaling
│   ├── models.py             # SVM, XGBoost, CatBoost, BPANN
│   ├── deep_learning.py      # Deep Neural Network (feature extraction)
│   ├── segmentation.py       # K-means behavioral clustering
│   └── reinforcement_learning.py  # Q-learning marketing engine
│
├── outputs/
│   ├── figures/              # All plots (correlation, confusion, ROC, etc.)
│   └── reports/              # Classification reports & comparison table
│
├── main.py                   # Run the entire pipeline
├── requirements.txt          # All dependencies
└── README.md
```
---

## 📊 Dataset

**UCI Online Shoppers Purchasing Intention Dataset**

| Property | Value |
|---|---|
| Source | UCI Machine Learning Repository |
| Sessions | 12,330 |
| Features | 18 |
| Target | Revenue (Purchase: Yes/No) |
| Class Imbalance | ~84% No / ~16% Yes → handled with SMOTE |

**Key Features:**

| Feature | Type | Description |
|---|---|---|
| Administrative | Numeric | Admin pages visited |
| Administrative_Duration | Numeric | Time on admin pages |
| ProductRelated | Numeric | Product pages visited |
| ProductRelated_Duration | Numeric | Time on product pages |
| BounceRates | Numeric | % single-page visits |
| ExitRates | Numeric | % exits from page |
| PageValues | Numeric | Average page value |
| Month | Categorical | Month of visit |
| VisitorType | Categorical | New/Returning visitor |
| Revenue | Boolean | **Target variable** |

---

## ⚙️ Preprocessing Pipeline
```
Raw Data (12,330 rows)
↓
Handle Missing Values
↓
Encode Categorical Variables (Label Encoding)
↓
Feature Scaling (StandardScaler)
↓
SMOTE (Balance classes)
↓
Train/Test Split (80% / 20%)
↓
Ready for Modeling
```
---

## 🤖 Phase 1 — ML Models

### Models & Hyperparameters

| Model | Key Parameters |
|---|---|
| SVM | kernel=RBF, C=50.12, gamma=0.120 |
| XGBoost | max_depth=4, learning_rate=0.11, subsample=0.8 |
| CatBoost | iterations=245, learning_rate=0.09, max_depth=8 |
| BPANN | hidden_layers=3, epochs=50 |

### Results

| Model | Precision | Recall | ROC AUC |
|---|---|---|---|
| **CatBoost** | 93.4% | 93.5% | **0.985** ⭐ |
| XGBoost | 93.5% | 92.5% | 0.984 |
| SVM | 95.4% | 88.6% | 0.977 |
| BPANN | 90.1% | 90.2% | 0.955 |

**Key Finding:** CatBoost achieves the best overall performance with
the highest ROC AUC (0.985), excelling at handling complex categorical
features in e-commerce data.

---

## 🔬 Phase 2 — Hybrid Framework
```
Raw Consumer Data
↓
┌─────────────────────────────┐
│   Layer 1: Deep Learning    │  → Extracts hidden behavioral patterns
│   (Multi-layer DNN)         │    from raw interaction data
└─────────────────────────────┘
↓
┌─────────────────────────────┐
│ Layer 2: K-means Clustering │  → Groups customers into behavioral
│ (Behavioral Segmentation)   │    segments automatically
└─────────────────────────────┘
↓
┌─────────────────────────────┐
│  Layer 3: Q-learning (RL)   │  → Makes real-time personalized
│  (Reinforcement Learning)   │    marketing decisions per segment
└─────────────────────────────┘
↓
Personalized Marketing Actions
(Dynamic Pricing / Targeted Offers / Churn Prevention)
```
### RL Components

| Component | Description |
|---|---|
| State (S) | Current user context + behavioral segment |
| Action (A) | Marketing decision (offer / price / retention) |
| Reward (R) | User engagement (click / convert / retain) |
| Strategy | ε-greedy exploration/exploitation |

---

## 📈 Outputs
```
All results are saved automatically in `outputs/`:
outputs/
├── figures/
│   ├── correlation_matrix.png
│   ├── feature_importance.png
│   ├── confusion_matrix_catboost.png
│   ├── confusion_matrix_xgboost.png
│   ├── confusion_matrix_svm.png
│   ├── confusion_matrix_bpann.png
│   └── roc_curves_comparison.png
│
└── reports/
├── classification_report_catboost.txt
├── classification_report_xgboost.txt
├── classification_report_svm.txt
├── classification_report_bpann.txt
└── final_comparison_table.csv
```
---

## ✅ Validation

- **10-fold Stratified Cross-Validation** on all models
- **Statistical Testing:** Paired t-tests (p < 0.05)
- **Baseline Comparison:** All 4 ML models vs Hybrid Framework
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC AUC

---

## ⚙️ Installation

```bash
# 1. Clone the repository
git clone https://github.com/elshhawy/consumer-behavior-ml.git
cd consumer-behavior-ml

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add the dataset
# Download from UCI and place in data/
# https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset
```

---

## 🚀 Usage

```bash
# Run the full pipeline (Phase 1 + Phase 2)
python main.py

# Run Phase 1 only (4 ML Models)
python src/models.py

# Run Phase 2 only (Hybrid Framework)
python src/deep_learning.py
python src/segmentation.py
python src/reinforcement_learning.py
```

---

## 📦 Requirements
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
scikit-learn>=1.2.0
xgboost>=1.7.0
catboost>=1.1.0
imbalanced-learn>=0.10.0
tensorflow>=2.11.0

---

## 📚 Reference

> Lin, J. (2025). Application of machine learning in predicting consumer
> behavior and precision marketing. *PLoS ONE*, 20(5), e0321854.
> https://doi.org/10.1371/journal.pone.0321854

---

## 👤 Author

**Abdelrahman Elshhawy**

[![GitHub](https://img.shields.io/badge/GitHub-elshhawy-black?logo=github)](https://github.com/elshhawy)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-elshhawy95-blue?logo=linkedin)](https://www.linkedin.com/in/elshhawy95)

---

## 📄 License

This project is licensed under the MIT License.
