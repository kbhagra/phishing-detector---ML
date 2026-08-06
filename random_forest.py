import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from get_dataset import load_dataset


def train_random_forest():
    print("Loading dataset...")
    df, csv_path = load_dataset()
    X = df.drop(columns=['id', 'CLASS_LABEL'])
    y = df['CLASS_LABEL']
    target_names = ['Class 0: Malicious Phishing', 'Class 1: Legitimate']

    print("\nDataset Info:")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    rf_model = RandomForestClassifier(n_estimators=5, warm_start=True, oob_score=True, random_state=42, n_jobs=-1)
    
    # oob error
    # Note: 5,10,15 give warnings because not every input value has an OOB score
    param_range = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    print(f"\nCalculating OOB scores for n_estimators = {param_range}...")
    
    train_mean = []
    oob_mean = []
    
    for n in param_range:
        rf_model.n_estimators = n
        rf_model.fit(X_train, y_train)
        
        # Training accuracy
        tr_acc = accuracy_score(y_train, rf_model.predict(X_train)) * 100
        train_mean.append(tr_acc)
        
        # OOB accuracy
        oob_acc = rf_model.oob_score_ * 100
        oob_mean.append(oob_acc)
        
        print(f"Trees: {n:3d} | Train Acc: {tr_acc:.2f}% | OOB Val Acc: {oob_acc:.2f}%")

    # final test set    
    y_pred = rf_model.predict(X_test)
    y_prob = rf_model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=target_names)

    # ---------------------------
    # Printing and Saving Results
    
    print("\n" + "=" * 60)
    print("Random Forest Results (100 estimators)")
    print("=" * 60)
    print(f"Test Accuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"OOB Score: {rf_model.oob_score_ * 100:.2f}%")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=target_names, columns=['Pred 0 (Phishing)', 'Pred 1 (Legitimate)']))
    print("\nClassification Report:")
    print(report)
    
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # Plot OOB validation curves
    plt.figure(figsize=(10, 5))
    plt.plot(param_range, train_mean, label='Training Accuracy', color='#1f77b4', marker='o', linewidth=2)
    plt.plot(param_range, oob_mean, label='OOB Accuracy', color='#ff7f0e', marker='s', linestyle='--', linewidth=2)
    plt.title('Random Forest: Effect of n_estimators on OOB Accuracy', fontsize=12, fontweight='bold')
    plt.xlabel('Number of Trees (n_estimators)', fontsize=10)
    plt.ylabel('Accuracy (%)', fontsize=10)
    plt.legend(loc='lower right')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    
    curve_plot_path = os.path.join(plots_dir, "random_forest_oob_curve.png")
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    print(f"\nSaved OOB curve plot to: {curve_plot_path}")

    # feature importance
    importances = rf_model.feature_importances_
    feature_names = X.columns
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    top_n = 15
    top_features = feature_importance_df.head(top_n)
    
    plt.figure(figsize=(10, 6))
    plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1], color='#2ca02c', edgecolor='black')
    plt.title(f'Random Forest: Top {top_n} Feature Importances', fontsize=12, fontweight='bold')
    plt.xlabel('Gini Importance', fontsize=10)
    plt.tight_layout()
    
    fi_plot_path = os.path.join(plots_dir, "random_forest_feature_importance.png")
    plt.savefig(fi_plot_path, dpi=300)
    plt.close()
    print(f"Saved feature importances plot to  : {fi_plot_path}")

    # confusion matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Greens)
    plt.title('Random Forest: Confusion Matrix', fontsize=12, fontweight='bold')
    plt.colorbar()
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, ['Phishing (0)', 'Legitimate (1)'], rotation=0)
    plt.yticks(tick_marks, ['Phishing (0)', 'Legitimate (1)'])
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black",
                     fontsize=12, fontweight='bold')
            
    plt.ylabel('True Class')
    plt.xlabel('Predicted Class')
    plt.tight_layout()
    
    cm_plot_path = os.path.join(plots_dir, "random_forest_confusion_matrix.png")
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # save trained model
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest_model.joblib")
    joblib.dump(rf_model, model_path)
    print(f"Random Forest model saved to: {model_path}")
    print("=" * 60)
    
    return rf_model


if __name__ == "__main__":
    train_random_forest()