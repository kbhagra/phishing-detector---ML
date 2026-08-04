import os 
import joblib 
import pandas as pd 
import numpy as np 
from sklearn .dummy import DummyClassifier 
from sklearn .tree import DecisionTreeClassifier 
from sklearn .metrics import accuracy_score ,precision_score ,recall_score ,f1_score ,roc_auc_score ,confusion_matrix 
from eda_preprocessing import run_eda_and_preprocessing 
def train_and_evaluate_baselines ():
    print ("="*70 )
    print ("      BASELINE MODELS EVALUATION (REQUIREMENT #4)")
    print ("="*70 )
    df ,X_train ,X_test ,y_train ,y_test ,X_train_scaled ,X_test_scaled ,scaler =run_eda_and_preprocessing ()
    target_names =['Class 0: Phishing','Class 1: Legitimate']
    dummy =DummyClassifier (strategy ="most_frequent")
    dummy .fit (X_train ,y_train )
    dummy_preds =dummy .predict (X_test )
    dummy_probs =dummy .predict_proba (X_test )[:,1 ]
    dummy_acc =accuracy_score (y_test ,dummy_preds )
    dummy_prec =precision_score (y_test ,dummy_preds ,zero_division =0 )
    dummy_rec =recall_score (y_test ,dummy_preds ,zero_division =0 )
    dummy_f1 =f1_score (y_test ,dummy_preds ,zero_division =0 )
    dummy_auc =roc_auc_score (y_test ,dummy_probs )
    print ("\n--- BASELINE 1: Majority Class Predictor (Dummy) ---")
    print (f"Accuracy  : {dummy_acc *100 :.2f}%")
    print (f"Precision : {dummy_prec :.4f}")
    print (f"Recall    : {dummy_rec :.4f}")
    print (f"F1-Score  : {dummy_f1 :.4f}")
    print (f"ROC-AUC   : {dummy_auc :.4f}")
    dt_stump =DecisionTreeClassifier (max_depth =3 ,random_state =42 )
    dt_stump .fit (X_train ,y_train )
    dt_preds =dt_stump .predict (X_test )
    dt_probs =dt_stump .predict_proba (X_test )[:,1 ]
    dt_acc =accuracy_score (y_test ,dt_preds )
    dt_prec =precision_score (y_test ,dt_preds )
    dt_rec =recall_score (y_test ,dt_preds )
    dt_f1 =f1_score (y_test ,dt_preds )
    dt_auc =roc_auc_score (y_test ,dt_probs )
    print ("\n--- BASELINE 2: Decision Tree Stump (max_depth=3) ---")
    print (f"Accuracy  : {dt_acc *100 :.2f}%")
    print (f"Precision : {dt_prec :.4f}")
    print (f"Recall    : {dt_rec :.4f}")
    print (f"F1-Score  : {dt_f1 :.4f}")
    print (f"ROC-AUC   : {dt_auc :.4f}")
    models_dir ="models"
    os .makedirs (models_dir ,exist_ok =True )
    joblib .dump (dummy ,os .path .join (models_dir ,"majority_baseline.pt"))
    joblib .dump (dt_stump ,os .path .join (models_dir ,"decision_tree_baseline.pt"))
    print ("="*70 )
    return {
    "Majority Classifier":{
    "Accuracy":dummy_acc ,"Precision":dummy_prec ,"Recall":dummy_rec ,"F1":dummy_f1 ,"ROC_AUC":dummy_auc 
    },
    "Decision Tree (Stump)":{
    "Accuracy":dt_acc ,"Precision":dt_prec ,"Recall":dt_rec ,"F1":dt_f1 ,"ROC_AUC":dt_auc 
    }
    }
if __name__ =="__main__":
    train_and_evaluate_baselines ()
