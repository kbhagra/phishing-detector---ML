# Phishing Website Detector Machine Learning Project
**CS 171 Group Project**

An end-to-end Machine Learning and Deep Learning system designed to detect malicious website URLs using tabular feature extraction and raw character-level sequence modeling.

---

## Checklist & Core Requirements Compliance

| # | Requirement | Implementation & Location |
|---|---|---|
| **1** | **Problem Framing** | Binary classification task (Phishing vs Legitimate). Target: `CLASS_LABEL`. 10,000 samples, 48 features. Source: Kaggle (`shashwatwork/phishing-dataset-for-machine-learning`). License: CC BY 4.0. |
| **2** | **EDA & Preprocessing** | EDA handled in `eda.py`. Zero missing values, 50/50 class balance checked, Plots saved in `plots/eda_*.png`.-- Data Preprocessing found in the file for each model  |
| **3** | **Proper Evaluation Split** | 80/20 train/test split for all models except LSTM, which uses 60/20/20 train/validation/test |
| **4** | **Simple Baseline** | Linear Model Implemented as `logistic_regression.py`. |
| **5** | **Multiple Model Families** | **4 Distinct Families Trained:**<br>1. Linear: Logistic Regression (`logistic_regression.py`)<br>2. Tree-Based: Random Forest (`random_forest.py`)<br>3. Deep Neural Net: 3-Layer ReLU MLP (`mlp_neural_network.py`)<br>4. Recurrent NN: LSTM (`lstm_model.py`) |
| **6** | **Loss & Optimization** | Log Loss (Logistic Regression), Binary Cross-Entropy (MLP, LSTM), Adam Optimizer (`lr=0.001-0.003`), Batch Size = 64 and Epochs = 50 for MLP, Training loss/accuracy curves generated in `plots/`. |
| **7** | **Tuning & Regularization** | 5-Fold CV hyperparameter search (`hyperparameter_tuning.py`): L1/L2 penalty & C search for Logistic Regression, tree depth for Random Forest, and Dropout & L2 weight decay for MLP. |
| **8** | **Class Imbalance** | Dataset is 50% Phishing / 50% Legitimate (perfectly balanced). No class imbalance to address. |
| **9** | **Rigorous Metrics** | Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix. Some metrics found in `plots/`, or others are found by running each model's file. |
| **10** | **Error Analysis & Insight** | On report. |
| **11** | **Cloud Compute** | Benchmarked via executable Jupyter Notebook `colab_phishing_nn.ipynb` running on Google Colab. |
| **12** | **Reproducibility** | Seeded runs (`seed=42`), listed environment (`requirements.txt`), and single script to run all files (except the notebook)(`run_pipeline.py`). |

---

## How to Execute these Files:

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Pipeline
```bash
python run_pipeline.py
```
This single command executes all 7 pipeline steps top-to-bottom:
1. Downloads Kaggle dataset (`get_dataset.py`)
2. Generates EDA plots and gives a Sample Datapoint (`eda.py`)
3. Trains Logistic Regression (`logistic_regression.py`)
4. Trains Random Forest (`random_forest.py`)
5. Trains 3-Layer Deep ReLU MLP (`mlp_neural_network.py`)
6. Trains Feature LSTM (`lstm_model.py`)
7. Runs 5-Fold Stratified CV Hyperparameter Search (`hyperparameter_tuning.py`)