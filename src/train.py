import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold
import optuna

from src.data_processing import TabularPreprocessor
from src.dataset import create_dataloaders
from src.models import get_model
from src.utils import set_seed, calculate_rmse

optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(X_batch)
    return total_loss / len(dataloader.dataset)

def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            total_loss += loss.item() * len(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.vstack(all_preds).ravel()
    all_targets = np.vstack(all_targets).ravel()
    
    rmse_dollars = calculate_rmse(np.expm1(all_targets), np.expm1(all_preds))
    return total_loss / len(dataloader.dataset), rmse_dollars, all_preds

def train_kfold(df: pd.DataFrame, config: dict, save_dir: str = 'saved_models', verbose: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    set_seed(config.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"--- Training using device: {device} ---")

    y_all_log = np.log1p(df['SalePrice'].values)
    X_all = df.drop(columns=['SalePrice'])

    # Fit overall preprocessor to establish fixed feature category dictionary
    full_preprocessor = TabularPreprocessor()
    X_full_proc = full_preprocessor.fit_transform(X_all)
    full_preprocessor.save(os.path.join(save_dir, 'pipeline.joblib'))

    input_dim = X_full_proc.shape[1]
    if verbose:
        print(f"Processed input feature dimension: {input_dim}")

    # Extract categories_dict for fold consistency
    categories_dict = {}
    if full_preprocessor.nominal_cols_ and full_preprocessor.encoder_ is not None:
        for col, cats in zip(full_preprocessor.nominal_cols_, full_preprocessor.encoder_.categories_):
            categories_dict[col] = cats

    n_splits = config.get('n_splits', 10)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=config.get('seed', 42))

    oof_preds_log = np.zeros(len(df))
    fold_rmse_list = []
    history = {'train_losses': [], 'val_losses': [], 'val_rmses': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all_log)):
        if verbose:
            print(f"\n========== Fold {fold + 1}/{n_splits} ==========")

        X_train_df, y_train_log = X_all.iloc[train_idx], y_all_log[train_idx]
        X_val_df, y_val_log = X_all.iloc[val_idx], y_all_log[val_idx]

        # Fit preprocessor strictly on train fold using fixed categories_dict
        fold_preprocessor = TabularPreprocessor(categories_dict=categories_dict)
        X_train_proc = fold_preprocessor.fit_transform(X_train_df)
        X_val_proc = fold_preprocessor.transform(X_val_df)

        train_loader, val_loader = create_dataloaders(
            X_train_proc, y_train_log, X_val_proc, y_val_log,
            batch_size=config.get('batch_size', 32)
        )

        model = get_model(
            architecture=config.get('architecture', 'resnet'),
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_blocks=config.get('num_blocks', 3),
            dropout=config.get('dropout', 0.15),
            activation=config.get('activation', 'silu')
        ).to(device)

        criterion = nn.SmoothL1Loss()
        optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 1e-3),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=config.get('epochs', 150), eta_min=1e-6)

        best_val_loss = float('inf')
        best_val_rmse = float('inf')
        best_preds = None
        patience = config.get('patience', 30)
        patience_counter = 0

        fold_train_losses, fold_val_losses = [], []

        for epoch in range(1, config.get('epochs', 150) + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_rmse, val_preds = validate_epoch(model, val_loader, criterion, device)
            scheduler.step()

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_rmse = val_rmse
                best_preds = val_preds
                patience_counter = 0
                torch.save(model.state_dict(), os.path.join(save_dir, f'model_fold_{fold}.pt'))
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch}. Best Val Loss: {best_val_loss:.5f}, Best Val RMSE: ${best_val_rmse:,.2f}")
                break

        oof_preds_log[val_idx] = best_preds
        fold_rmse_list.append(best_val_rmse)
        if verbose:
            print(f"Fold {fold + 1} Best RMSE: ${best_val_rmse:,.2f}")

        history['train_losses'].append(fold_train_losses)
        history['val_losses'].append(fold_val_losses)

    oof_preds_dollars = np.expm1(oof_preds_log)
    true_dollars = np.expm1(y_all_log)
    overall_oof_rmse = calculate_rmse(true_dollars, oof_preds_dollars)

    if verbose:
        print(f"\n==========================================")
        print(f"Overall Out-of-Fold RMSE: ${overall_oof_rmse:,.2f}")
        print(f"Mean Fold RMSE: ${np.mean(fold_rmse_list):,.2f} +/- ${np.std(fold_rmse_list):,.2f}")
        print(f"==========================================")

    config_to_save = config.copy()
    config_to_save['input_dim'] = input_dim
    config_to_save['overall_oof_rmse'] = float(overall_oof_rmse)
    config_to_save['mean_fold_rmse'] = float(np.mean(fold_rmse_list))

    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=4)

    return overall_oof_rmse, oof_preds_dollars, history


def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 12, save_dir: str = 'saved_models'):
    print(f"Starting Optuna Hyperparameter Optimization ({n_trials} trials)...")

    def objective(trial):
        config = {
            'architecture': trial.suggest_categorical('architecture', ['resnet', 'standard', 'wide_deep']),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'num_blocks': trial.suggest_int('num_blocks', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.05, 0.25, step=0.05),
            'activation': trial.suggest_categorical('activation', ['silu', 'gelu']),
            'lr': trial.suggest_float('lr', 5e-4, 3e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32]),
            'epochs': 80,
            'patience': 20,
            'n_splits': 5,
            'seed': 42
        }

        oof_rmse, _, _ = train_kfold(df, config, save_dir=os.path.join(save_dir, f"trial_{trial.number}"), verbose=False)
        return oof_rmse

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials)

    print("\nBest Optuna Trial:")
    print(f"  OOF RMSE: ${study.best_value:,.2f}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    return study.best_params
