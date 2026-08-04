import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from get_dataset import load_dataset


def train_logistic_regression():
    print("Loading dataset...")
    df, csv_path = load_dataset()

    X = df.drop(columns=["id", "CLASS_LABEL"])
    y = df["CLASS_LABEL"]

    target_names = [
        "Class 0: Malicious Phishing",
        "Class 1: Legitimate",
    ]
    classes = np.unique(y)

    print("\nDataset Info:")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print("Class 0 -> Malicious Phishing Links")
    print("Class 1 -> Legitimate Links")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print(
        f"\nTrain set: {X_train.shape[0]} samples | "
        f"Test set: {X_test.shape[0]} samples"
    )

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    print("\nTraining Logistic Regression...")

    model.fit(X_train_scaled, y_train)

    # training metrics
    train_prob = model.predict_proba(X_train_scaled)
    train_pred = model.predict(X_train_scaled)

    tr_loss = log_loss(y_train, train_prob)  # (log loss / cross entropy)
    tr_acc = accuracy_score(y_train, train_pred)

    # test metrics
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    test_loss = log_loss(y_test, model.predict_proba(X_test_scaled))
    test_acc = accuracy_score(y_test, y_pred)

    print("Training Complete")
    print(f"\nTrain Loss      : {tr_loss:.4f}")
    print(f"Test Loss : {test_loss:.4f}")
    print(f"Train Accuracy  : {tr_acc * 100:.2f}%")
    print(f"Test Accuracy : {test_acc * 100:.2f}%")

    # final evaluation metrics
    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(
        y_test,
        y_pred,
        target_names=target_names
    )

    # ---------------------------
    # Printing and Saving Results

    print("\n" + "=" * 60)
    print("Logistic Regression Results")
    print("=" * 60)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC  : {roc_auc:.4f}")

    print("\nConfusion Matrix:")
    print(
        pd.DataFrame(
            cm,
            index=target_names,
            columns=[
                "Pred 0 (Phishing)",
                "Pred 1 (Legitimate)",
            ],
        )
    )

    print("\nClassification Report:")
    print(report)

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    plt.figure(figsize=(6, 5))
    plt.imshow(
        cm,
        interpolation="nearest",
        cmap=plt.cm.Blues,
    )
    plt.title(
        "Confusion Matrix",
        fontsize=12,
        fontweight="bold",
    )
    plt.colorbar()

    tick_marks = np.arange(len(target_names))
    plt.xticks(
        tick_marks,
        ["Phishing (0)", "Legitimate (1)"],
        rotation=0,
    )
    plt.yticks(
        tick_marks,
        ["Phishing (0)", "Legitimate (1)"],
    )

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(  
                j,
                i,
                format(cm[i, j], "d"),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=12,
                fontweight="bold",
            )

    plt.ylabel("True Class")
    plt.xlabel("Predicted Class")
    plt.tight_layout()

    cm_plot_path = os.path.join(
        plots_dir,
        "logistic_regression_confusion_matrix.png",
    )
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix plot to: {cm_plot_path}")

    # saved trained model and associated scaler
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)

    model_path = os.path.join(
        models_dir,
        "logistic_regression_model.joblib",
    )
    scaler_path = os.path.join(
        models_dir,
        "logistic_regression_scaler.joblib",
    )

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    print(f"Model saved to: {model_path}")
    print(f"Scaler saved to: {scaler_path}")
    print("=" * 60)

    return model, scaler


if __name__ == "__main__":
    train_logistic_regression()