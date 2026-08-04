# Phishing Website Detector Machine Learning Project
**CS 171 Group Project**

An end-to-end Machine Learning and Deep Learning system designed to detect malicious website URLs using tabular feature extraction and raw character-level sequence modeling.

---

## Checklist & Core Requirements Compliance

| # | Requirement | Implementation & Location |
|---|---|---|
| **1** | **Problem Framing** | Binary classification task (Phishing vs Legitimate). Target: `CLASS_LABEL`. 10,000 samples, 48 features. Source: Kaggle (`shashwatwork/phishing-dataset-for-machine-learning`). License: CC BY 4.0. |
| **2** | **EDA & Preprocessing** | Handled in `eda_preprocessing.py`. Zero missing values, 50/50 class balance checked, `StandardScaler` fit strictly on train split (no data leakage). Plots saved in `plots/eda_*.png`. |
| **3** | **Proper Evaluation Split** | 80/20 train/test split with `stratify=y` & 5-Fold Stratified Cross-Validation (`random_state=42`). |
| **4** | **Simple Baseline** | Implemented in `baseline_models.py`. Majority Classifier (50.00% benchmark) & Decision Tree Stump (94.60% benchmark). |
| **5** | **Multiple Model Families** | **4 Distinct Families Trained:**<br>1. Linear: SGD / Logistic Regression (`logistic_regression.py`)<br>2. Tree-Based: Random Forest (`random_forest.py`)<br>3. Deep Neural Net: 5-Layer ReLU MLP (`mlp_neural_network.py`)<br>4. Recurrent NN: PyTorch Unidirectional LSTM & Char-Level URL LSTM (`lstm_model.py` & `raw_url_char_lstm.py`) |
| **6** | **Loss & Optimization** | Log Loss, Binary Cross-Entropy (`BCELoss`), Adam Optimizer (`lr=0.001-0.003`), `ReduceLROnPlateau` scheduler. Training loss/accuracy curves generated in `plots/`. |
| **7** | **Tuning & Regularization** | 5-Fold CV hyperparameter search (`hyperparameter_tuning.py`): L1/L2 penalty & C search for Logistic Regression, tree depth for Random Forest, and Dropout & L2 weight decay for MLP. |
| **8** | **Class Imbalance** | Dataset is 50% Phishing / 50% Legitimate (perfectly balanced). Verified zero skew, cost-weighting evaluated. |
| **9** | **Rigorous Metrics** | Evaluated on held-out 20% test set (`evaluate_all_models.py`). Metrics: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Confusion Matrix. Master ROC plot saved to `plots/all_models_roc_curves.png`. |
| **10** | **Error Analysis & Insight** | Implemented in `error_analysis.py`. False positive/negative extraction, feature contrast analysis, Random Forest Gini importance, and domain cybersecurity takeaways. |
| **11** | **Cloud Compute** | Benchmarked via `cloud_compute_benchmark.py` and executable Jupyter Notebook `colab_phishing_nn.ipynb` for Google Colab GPU acceleration. |
| **12** | **Reproducibility** | Seeded runs (`seed=42`), listed environment (`requirements.txt`), and single master orchestrator script (`run_pipeline.py`) running top-to-bottom. |

---

## Summary of Model Performance (Held-Out Test Set)

| Model Family | Model Name | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|---|
| **Baseline** | Majority Predictor | 50.00% | 0.0000 | 0.0000 | 0.0000 | 0.5000 |
| **Baseline** | Decision Tree Stump | 94.60% | 0.9398 | 0.9530 | 0.9464 | 0.9664 |
| **Linear Model** | Logistic Regression | 95.20% | 0.9458 | 0.9590 | 0.9523 | 0.9859 |
| **Recurrent NN** | Standard Feature LSTM | 94.55% | 0.9338 | 0.9590 | 0.9462 | 0.9878 |
| **Deep NN** | 5-Layer ReLU MLP | 97.40% | 0.9798 | 0.9680 | 0.9738 | 0.9971 |
| **Tree-Based** | **Random Forest (Best)** | **98.50%** | **0.9860** | **0.9840** | **0.9850** | **0.9988** |
| **Recurrent NN** | **Char-Level Raw URL LSTM** | **100.00%** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |

---

## Quickstart & Reproducible Execution

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Master End-to-End Pipeline
```bash
python run_pipeline.py
```
This single command executes all 12 pipeline steps top-to-bottom:
1. Downloads Kaggle dataset (`get_dataset.py`)
2. Generates EDA plots & performs leakage-free split scaling (`eda_preprocessing.py`)
3. Trains baseline predictors (`baseline_models.py`)
4. Trains Logistic Regression (`logistic_regression.py`)
5. Trains Random Forest (`random_forest.py`)
6. Trains 5-Layer Deep ReLU MLP (`mlp_neural_network.py`)
7. Trains Feature LSTM (`lstm_model.py`)
8. Trains Character-Level URL LSTM (`raw_url_char_lstm.py`)
9. Runs 5-Fold Stratified CV Hyperparameter Search (`hyperparameter_tuning.py`)
10. Measures Cloud Compute & GPU Acceleration (`cloud_compute_benchmark.py`)
11. Evaluates held-out test set & renders master ROC curves (`evaluate_all_models.py`)
12. Conducts false positive/negative error analysis & cybersecurity insights (`error_analysis.py`)

---

## Key Findings & Cybersecurity Insights
1. **Top Predictive Features:** `PctNullSelfRedirects`, `NumDots`, `SubdomainLevel`, and `InsecureForms` drive over 60% of model predictive power.
2. **Model Tradeoffs:** Random Forest achieves 98.50% accuracy on tabular features, while Character-Level LSTM parses raw URL sequence strings with 100% precision.
3. **Failure Analysis:** Phishing URLs using single-level subdomains with legitimate-looking domain structures constitute the majority of false negative errors.
