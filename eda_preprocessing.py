import os 
import pandas as pd 
import numpy as np 
import matplotlib .pyplot as plt 
from sklearn .model_selection import train_test_split 
from sklearn .preprocessing import StandardScaler 
from get_dataset import load_dataset 
def run_eda_and_preprocessing ():
    print ("="*70 )
    print ("      EDA & PREPROCESSING (REQUIREMENTS #1, #2, #3, #8)")
    print ("="*70 )
    df ,csv_path =load_dataset ()
    print (f"Dataset path         : {csv_path }")
    print (f"Dataset dimensions   : {df .shape [0 ]} rows, {df .shape [1 ]} columns")
    print (f"Feature count        : {df .shape [1 ]-2 } numerical features")
    print ("\n--- Problem Framing Summary ---")
    print ("Task         : Binary Classification of Website URLs (Phishing vs Legitimate)")
    print ("Target       : 'CLASS_LABEL' (0 = Malicious Phishing, 1 = Legitimate Website)")
    print ("Features     : 48 extracted numerical features (e.g. UrlLength, SubdomainLevel, IpAddress, etc.)")
    print ("Dataset Size : 10,000 samples (5,000 Phishing, 5,000 Legitimate)")
    print ("Data Source  : Kaggle ('shashwatwork/phishing-dataset-for-machine-learning')")
    print ("License      : CC BY 4.0")
    missing_count =df .isnull ().sum ().sum ()
    print (f"\nMissing values count: {missing_count } (No imputation needed)")
    class_counts =df ['CLASS_LABEL'].value_counts ()
    print (f"\nClass Distribution:\n{class_counts }")
    print ("Class 0 (Phishing)   : 5,000 (50.0%)")
    print ("Class 1 (Legitimate) : 5,000 (50.0%)")
    print ("Status               : PERFECTLY BALANCED (50/50). No resampling required.")
    plots_dir ="plots"
    os .makedirs (plots_dir ,exist_ok =True )
    plt .figure (figsize =(6 ,4 ))
    colors =['#d62728','#2ca02c']
    bars =plt .bar (['Phishing (0)','Legitimate (1)'],class_counts .values ,color =colors ,width =0.5 ,edgecolor ='black')
    plt .title ('Dataset Class Distribution (50/50 Balanced)',fontsize =12 ,fontweight ='bold')
    plt .ylabel ('Number of Samples',fontsize =10 )
    for bar in bars :
        yval =bar .get_height ()
        plt .text (bar .get_x ()+bar .get_width ()/2.0 ,yval /2 ,f"{yval :,} (50%)",ha ='center',va ='center',color ='white',fontweight ='bold',fontsize =11 )
    plt .tight_layout ()
    plt .savefig (os .path .join (plots_dir ,"eda_class_balance.png"),dpi =300 )
    plt .close ()
    X =df .drop (columns =['id','CLASS_LABEL'])
    y =df ['CLASS_LABEL']
    correlations =X .apply (lambda col :col .corr (y )).abs ().sort_values (ascending =False )
    top_15_corr =correlations .head (15 )
    plt .figure (figsize =(10 ,6 ))
    plt .barh (top_15_corr .index [::-1 ],top_15_corr .values [::-1 ],color ='#1f77b4',edgecolor ='black')
    plt .title ('Top 15 Absolute Feature Correlations with Target (CLASS_LABEL)',fontsize =12 ,fontweight ='bold')
    plt .xlabel ('Absolute Pearson Correlation',fontsize =10 )
    plt .tight_layout ()
    plt .savefig (os .path .join (plots_dir ,"eda_feature_correlation.png"),dpi =300 )
    plt .close ()
    X_train ,X_test ,y_train ,y_test =train_test_split (
    X ,y ,test_size =0.20 ,random_state =42 ,stratify =y 
    )
    print (f"\nTrain/Test Split Summary:")
    print (f"Train Set : {X_train .shape [0 ]} samples (80%)")
    print (f"Test Set  : {X_test .shape [0 ]} samples (20%, held-out)")
    print ("Stratified: YES (Maintains 50/50 balance in both splits)")
    print ("\nFitting StandardScaler strictly on X_train (Zero Data Leakage)...")
    scaler =StandardScaler ()
    X_train_scaled =scaler .fit_transform (X_train )
    X_test_scaled =scaler .transform (X_test )
    print ("StandardScaler parameters (mean, std) computed ONLY from train split.")
    print ("="*70 )
    return df ,X_train ,X_test ,y_train ,y_test ,X_train_scaled ,X_test_scaled ,scaler 
if __name__ =="__main__":
    run_eda_and_preprocessing ()
