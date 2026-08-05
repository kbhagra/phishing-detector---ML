import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
from get_dataset import load_dataset

torch.manual_seed(42)
np.random.seed(42)

class MLPmodel(nn.Module):
    """
    3-Layer MLP
    Activation: ReLU Activation + Sigmoid Output
    Weight Initialization: He Normal Initialization
    """
    def __init__(self, input_dim, dropout_rate=0.1):
        super(MLPmodel, self).__init__()
        self.net = nn.Sequential(
            # layer 1
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # layer 2
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            # layer 3
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),

            # output layer
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.apply(self._init_weights)

    def forward(self, x):
        return self.net(x)

    # he initialization for weights
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

def train_relu_mlp(epochs=50, batch_size=64, learning_rate=0.001, weight_decay=1e-4):
    print("Loading dataset...")
    df, csv_path = load_dataset()
    X = df.drop(columns=['id', 'CLASS_LABEL'])
    y = df['CLASS_LABEL']
    target_names = ['Class 0: Malicious Phishing', 'Class 1: Legitimate']

    print("\nDataset Info:")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print("Class 0 -> Malicious Phishing Links")
    print("Class 1 -> Legitimate Links")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")

    # scale data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # transform data
    X_tr_t = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    X_va_t = torch.tensor(X_test_scaled, dtype=torch.float32)
    y_va_t = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)
    
    train_dataset = TensorDataset(X_tr_t, y_tr_t)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # config
    model = MLPmodel(input_dim=X_train.shape[1])
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5) # reduces learning rate after seeing no change in loss
    
    print(f"\n--- 3-Layer ReLU MLP Architecture ---")
    print(model)
    print(f"\nTraining ReLU MLP over {epochs} epochs...")
    
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    for epoch in range(1, epochs + 1):
        # train the model for an epoch
        model.train()
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

        # test the model's results after an epoch
        model.eval()
        with torch.no_grad():
            tr_outputs = model(X_tr_t)
            va_outputs = model(X_va_t)
            tr_loss = criterion(tr_outputs, y_tr_t).item()
            va_loss = criterion(va_outputs, y_va_t).item()
            tr_preds = (tr_outputs.numpy() >= 0.5).astype(int)
            va_preds = (va_outputs.numpy() >= 0.5).astype(int)
            tr_acc = accuracy_score(y_train, tr_preds) * 100
            va_acc = accuracy_score(y_test, va_preds) * 100
            
        scheduler.step(va_loss)
        train_losses.append(tr_loss)
        val_losses.append(va_loss)
        train_accs.append(tr_acc)
        val_accs.append(va_acc)

        # print progress at certain milestones
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"Epoch {epoch:2d}/{epochs} | "
                  f"Train Loss: {tr_loss:.4f} | Val Loss: {va_loss:.4f} | "
                  f"Train Acc: {tr_acc:.2f}% | Val Acc: {va_acc:.2f}%")

    # final test
    model.eval()
    with torch.no_grad():
        final_probs = model(X_va_t).numpy()
        final_preds = (final_probs >= 0.5).astype(int)
        
    acc = accuracy_score(y_test, final_preds)
    roc_auc = roc_auc_score(y_test, final_probs)
    cm = confusion_matrix(y_test, final_preds)
    report = classification_report(y_test, final_preds, target_names=target_names)

    # ---------------------------
    # Printing and Saving Results
    
    print("\n" + "=" * 60)
    print("MLP Results")
    print("=" * 60)
    print(f"Accuracy : {acc * 100:.2f}%")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm, index=target_names, columns=['Pred 0 (Phishing)', 'Pred 1 (Legitimate)']))
    print("\nClassification Report:")
    print(report)

    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)

    # loss and accuracy curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(range(1, epochs + 1), train_losses, label='Train Loss', color='#1f77b4', linewidth=2)
    axes[0].plot(range(1, epochs + 1), val_losses, label='Validation Loss', color='#d62728', linewidth=2, linestyle='--')
    axes[0].set_title('MLP: Loss over Epochs', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=10)
    axes[0].set_ylabel('Binary Cross-Entropy Loss', fontsize=10)
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    axes[1].plot(range(1, epochs + 1), train_accs, label='Train Accuracy', color='#1f77b4', linewidth=2)
    axes[1].plot(range(1, epochs + 1), val_accs, label='Validation Accuracy', color='#2ca02c', linewidth=2, linestyle='--')
    axes[1].set_title('MLP: Accuracy over Epochs', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=10)
    axes[1].set_ylabel('Accuracy (%)', fontsize=10)
    axes[1].legend(loc='lower right')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    curve_plot_path = os.path.join(plots_dir, "mlp_curves.png")
    plt.savefig(curve_plot_path, dpi=300)
    plt.close()
    
    print(f"\nSaved training curves plot to      : {curve_plot_path}")

    # confusion matrix
    cm_plot_path = os.path.join(plots_dir, "mlp_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0: Phishing', 'Class 1: Legitimate'])
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d', colorbar=True)
    
    ax.set_title('MLP: Confusion Matrix', fontsize=13, fontweight='bold', pad=15)
    
    plt.tight_layout()
    plt.savefig(cm_plot_path, dpi=300)
    plt.close()

    print(f"Saved confusion matrix plot to    : {cm_plot_path}")

    # save model and associated scaler
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "mlp_model.pt")
    scaler_path = os.path.join(models_dir, "mlp_scaler.joblib")
    
    torch.save(model.state_dict(), model_path)
    joblib.dump(scaler, scaler_path)
    
    print(f"ReLU MLP Model weights saved to    : {model_path}")
    print(f"ReLU MLP Scaler saved to           : {scaler_path}")
    print("=" * 60)
    
    return model, scaler

if __name__ == "__main__":
    train_relu_mlp(epochs=50)