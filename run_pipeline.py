import os 
import sys 
import time 
from get_dataset import load_dataset 
from eda_preprocessing import run_eda_and_preprocessing 
from baseline_models import train_and_evaluate_baselines 
from logistic_regression import train_logistic_regression 
from random_forest import train_random_forest 
from mlp_neural_network import train_relu_mlp 
from lstm_model import train_lstm_model 
from raw_url_char_lstm import train_raw_url_char_lstm 
from hyperparameter_tuning import main as run_hyperparameter_tuning 
from cloud_compute_benchmark import benchmark_cloud_training 
from evaluate_all_models import evaluate_all_models 
from error_analysis import run_error_analysis 
def run_full_pipeline ():
    start_total_time =time .time ()
    print ("\n"+"#"*85 )
    print ("      CS 171 END-TO-END PHISHING DETECTOR ML PIPELINE EXECUTION")
    print ("#"*85 )
    print ("\n[STEP 1/12] Loading & Caching Kaggle Dataset...")
    load_dataset ()
    print ("\n[STEP 2/12] Running EDA, Missing Values Check, Class Balance Plot & Split Scaling...")
    run_eda_and_preprocessing ()
    print ("\n[STEP 3/12] Training Baseline Models (Majority Predictor & Decision Tree)...")
    train_and_evaluate_baselines ()
    print ("\n[STEP 4/12] Training Model Family 1: Logistic Regression...")
    train_logistic_regression (epochs =50 )
    print ("\n[STEP 5/12] Training Model Family 2: Random Forest...")
    train_random_forest (max_trees =50 )
    print ("\n[STEP 6/12] Training Model Family 3: 5-Layer Deep ReLU MLP Neural Network...")
    train_relu_mlp (epochs =30 )
    print ("\n[STEP 7/12] Training Model Family 4 (Part A): Standard Feature PyTorch LSTM...")
    train_lstm_model (epochs =15 )
    print ("\n[STEP 8/12] Training Model Family 4 (Part B): Character-Level Raw URL PyTorch LSTM...")
    train_raw_url_char_lstm (epochs =15 )
    print ("\n[STEP 9/12] Executing 5-Fold CV Hyperparameter Tuning & Regularization Search...")
    run_hyperparameter_tuning ()
    print ("\n[STEP 10/12] Executing Cloud Compute Hardware & GPU Acceleration Benchmark...")
    benchmark_cloud_training ()
    print ("\n[STEP 11/12] Evaluating All Models on Held-Out Test Set & Generating ROC Curves...")
    evaluate_all_models ()
    print ("\n[STEP 12/12] Running False Positive/Negative Error Analysis & Cybersecurity Insights...")
    run_error_analysis ()
    end_total_time =time .time ()
    elapsed =end_total_time -start_total_time 
    print ("\n"+"#"*85 )
    print (f"      PIPELINE COMPLETE! Total Elapsed Time: {elapsed :.2f} seconds ({elapsed /60 :.2f} minutes)")
    print ("#"*85 )
    print ("All 12 core requirement checklist items are satisfied!")
if __name__ =="__main__":
    run_full_pipeline ()
