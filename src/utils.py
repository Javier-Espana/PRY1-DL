import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def set_seed(seed: int = 42):
    """
    Ensures reproducibility across numpy, random, torch, and CUDA.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def calculate_rmse(y_true, y_pred):
    """
    Root Mean Squared Error in original dollar scale.
    """
    return np.sqrt(mean_squared_error(y_true, y_pred))

def calculate_metrics(y_true, y_pred):
    """
    Comprehensive regression metrics suite.
    """
    rmse = calculate_rmse(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {
        'RMSE': float(rmse),
        'MAE': float(mae),
        'R2': float(r2),
        'MAPE': float(mape)
    }

def plot_learning_curves(train_losses, val_losses, title="Training and Validation Loss Curve", save_path=None):
    """
    Plots training loss and validation loss over epochs.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss (MSE Log Scale)', color='#2563EB', linewidth=2)
    plt.plot(val_losses, label='Val Loss (MSE Log Scale)', color='#DC2626', linewidth=2)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss (Smooth L1 / MSE)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_actual_vs_predicted(y_true, y_pred, title="Actual vs Predicted SalePrice ($)", save_path=None):
    """
    Scatter plot comparing true vs predicted house prices with reference diagonal.
    """
    plt.figure(figsize=(9, 7))
    plt.scatter(y_true, y_pred, alpha=0.5, color='#3B82F6', edgecolors='k', linewidth=0.5)
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Ideal 1:1 Line')
    plt.xlabel('Actual SalePrice ($)', fontsize=12)
    plt.ylabel('Predicted SalePrice ($)', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_residuals(y_true, y_pred, title="Residual Analysis (Errors)", save_path=None):
    """
    Plot residual errors (Predicted - Actual) vs Actual prices and residual histogram.
    """
    residuals = y_pred - y_true
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Scatter of residuals vs actual
    axes[0].scatter(y_true, residuals, alpha=0.5, color='#8B5CF6', edgecolors='k', linewidth=0.5)
    axes[0].axhline(0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Actual SalePrice ($)', fontsize=12)
    axes[0].set_ylabel('Residual ($)', fontsize=12)
    axes[0].set_title('Residuals vs Actual Values', fontsize=13, fontweight='bold')
    axes[0].grid(True, linestyle='--', alpha=0.6)

    # Residual distribution histogram
    sns.histplot(residuals, kde=True, ax=axes[1], color='#10B981', bins=30)
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_xlabel('Residual ($)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    axes[1].set_title('Residual Error Distribution', fontsize=13, fontweight='bold')
    axes[1].grid(True, linestyle='--', alpha=0.6)

    plt.suptitle(title, fontsize=15, fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()
