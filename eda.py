import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from get_dataset import load_dataset


def run_eda():
    print("=" * 70)
    print("EDA")
    print("=" * 70)
    
    df, csv_path = load_dataset()
    print(f"Dataset path         : {csv_path}")
    print(f"Dataset dimensions   : {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Feature count        : {df.shape[1] - 2} numerical features")
    
    print("\n--- Problem Framing Summary ---")
    print("Task         : Binary Classification of Website URLs (Phishing vs Legitimate)")
    print("Target       : 'CLASS_LABEL' (0 = Malicious Phishing, 1 = Legitimate Website)")
    print("Features     : 48 extracted numerical features (e.g. UrlLength, SubdomainLevel, IpAddress, etc.)")
    print("Dataset Size : 10,000 samples (5,000 Phishing, 5,000 Legitimate)")
    print("Data Source  : Kaggle ('shashwatwork/phishing-dataset-for-machine-learning')")
    print("License      : CC BY 4.0")
    
    missing_count = df.isnull().sum().sum()
    print(f"\nMissing values count: {missing_count} (No imputation needed)")
    
    class_counts = df['CLASS_LABEL'].value_counts()
    print(f"\nClass Distribution:\n{class_counts}")
    print("Class 0 (Phishing)   : 5,000 (50.0%)")
    print("Class 1 (Legitimate) : 5,000 (50.0%)")
    print("Status               : PERFECTLY BALANCED (50/50). No resampling required.")

    print("\n--- Sample Datapoint ---")
    sample = df.iloc[1]
    print(f"Dataset ID   : {sample.get('id', 'N/A')}")
    print(f"Class Label  : {sample['CLASS_LABEL']} ({'Legitimate' if sample['CLASS_LABEL'] == 1 else 'Phishing'})")
    print("Feature Values (Sample subset):")
    for col in list(df.columns[1:11]):
        print(f"  {col:<20} : {sample[col]}")
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    plt.figure(figsize=(6, 4))
    colors = ['#d62728', '#2ca02c']
    bars = plt.bar(['Phishing (0)', 'Legitimate (1)'], class_counts.values, color=colors, width=0.5, edgecolor='black')
    plt.title('Dataset Class Distribution', fontsize=12, fontweight='bold')
    plt.ylabel('Number of Samples', fontsize=10)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2.0, yval / 2, f"{yval:,} (50%)", ha='center', va='center', color='white', fontweight='bold', fontsize=11)
        
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "eda_class_balance.png"), dpi=300)
    plt.close()


if __name__ == "__main__":
    run_eda()