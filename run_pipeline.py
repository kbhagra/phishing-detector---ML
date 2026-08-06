import os
import sys
import time
from get_dataset import load_dataset
from eda import run_eda
from hyperparameter_tuning import main as run_hyperparameter_tuning
from logistic_regression import train_logistic_regression
from lstm_model import train_lstm_model
from mlp_neural_network import train_relu_mlp
from random_forest import train_random_forest


def run_full_pipeline():
    start_total_time = time.time()
    print("\n" + "#" * 85)
    print("Running Pipeline...")
    print("#" * 85)
    print("\n[STEP 1] Loading & Caching Kaggle Dataset...")
    load_dataset()
    print("\n[STEP 2] Running EDA (Check Missing Values, Class Balance)...")
    run_eda()
    print("\n[STEP 3] Training Model Family 1 (Baseline Model): Logistic Regression...")
    train_logistic_regression(epochs=50)
    print("\n[STEP 4] Training Model Family 2: Random Forest...")
    train_random_forest(max_trees=50)
    print("\n[STEP 5] Training Model Family 3: 3-Layer ReLU MLP Neural Network...")
    train_relu_mlp(epochs=30)
    print("\n[STEP 6] Training Model Family 4: Standard Feature PyTorch LSTM...")
    train_lstm_model(epochs=15)
    print("\n[STEP 7] Executing 5-Fold CV Hyperparameter Tuning & Regularization Search...")
    run_hyperparameter_tuning()

    end_total_time = time.time()
    elapsed = end_total_time - start_total_time
    print("\n" + "#" * 85)
    print(f"Completed Pipeline.\nTotal Elapsed Time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print("#" * 85)


if __name__ == "__main__":
    run_full_pipeline()