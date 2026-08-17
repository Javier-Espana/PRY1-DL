import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR
from sklearn.model_selection import KFold
import optuna

from src.data_processing import TabularPreprocessor
from src.dataset import create_dataloaders
from src.models import get_model
from src.utils import set_seed, calculate_rmse

optuna.logging.set_verbosity(optuna.logging.WARNING)

def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in dataloader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        if use_mixup and np.random.rand() < 0.25:
            lam = np.random.beta(0.3, 0.3)
            perm = torch.randperm(len(X_batch))
            X_batch = lam * X_batch + (1 - lam) * X_batch[perm]
            y_batch = lam * y_batch + (1 - lam) * y_batch[perm]

        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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

def train_kfold(df: pd.DataFrame, config: dict, save_dir: str = 'data/saved_models', verbose: bool = True):
    os.makedirs(save_dir, exist_ok=True)
    seed = config.get('seed', 42)
    set_seed(seed)

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

    # Identify extreme leverage points to exclude from training folds only
    outlier_mask = (df['GrLivArea'] > 4000) & (df['SalePrice'] < 300000)
    outlier_indices = set(df[outlier_mask].index.tolist())

    n_splits = config.get('n_splits', 10)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    oof_preds_log = np.zeros(len(df))
    fold_rmse_list = []
    history = {'train_losses': [], 'val_losses': [], 'val_rmses': []}

    arch = config.get('architecture', 'resnet')

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all_log)):
        if verbose:
            print(f"\n========== Fold {fold + 1}/{n_splits} ==========")

        clean_train_idx = np.array([i for i in train_idx if i not in outlier_indices])

        X_train_df, y_train_log = X_all.iloc[clean_train_idx], y_all_log[clean_train_idx]
        X_val_df, y_val_log = X_all.iloc[val_idx], y_all_log[val_idx]

        fold_preprocessor = TabularPreprocessor(categories_dict=categories_dict)
        X_train_proc = fold_preprocessor.fit_transform(X_train_df)
        X_val_proc = fold_preprocessor.transform(X_val_df)

        train_loader, val_loader = create_dataloaders(
            X_train_proc, y_train_log, X_val_proc, y_val_log,
            batch_size=config.get('batch_size', 16)
        )

        model = get_model(
            architecture=arch,
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 512),
            num_blocks=config.get('num_blocks', 3),
            dropout=config.get('dropout', 0.15),
            activation=config.get('activation', 'gelu')
        ).to(device)

        criterion = nn.SmoothL1Loss(beta=0.02)
        optimizer = AdamW(
            model.parameters(),
            lr=config.get('lr', 8e-4),
            weight_decay=config.get('weight_decay', 1e-4)
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-6)

        best_val_loss = float('inf')
        best_val_rmse = float('inf')
        best_preds = None
        patience = config.get('patience', 35)
        patience_counter = 0

        fold_train_losses, fold_val_losses = [], []

        for epoch in range(1, config.get('epochs', 160) + 1):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device, use_mixup=True)
            val_loss, val_rmse, val_preds = validate_epoch(model, val_loader, criterion, device)
            scheduler.step()

            fold_train_losses.append(train_loss)
            fold_val_losses.append(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_rmse = val_rmse
                best_preds = val_preds
                patience_counter = 0
                checkpoint_filename = f'model_{arch}_seed_{seed}_fold_{fold}.pt'
                torch.save(model.state_dict(), os.path.join(save_dir, checkpoint_filename))
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

    return overall_oof_rmse, oof_preds_dollars, history


def train_multi_ensemble(df: pd.DataFrame, seeds=(42, 123, 2024), architectures=('resnet', 'swiglu'), save_dir='data/saved_models'):
    os.makedirs(save_dir, exist_ok=True)
    all_oof_predictions = []
    checkpoint_manifest = []

    true_dollars = df['SalePrice'].values

    # Preprocessor initialization and save
    full_prep = TabularPreprocessor()
    X_full = full_prep.fit_transform(df.drop(columns=['SalePrice']))
    full_prep.save(os.path.join(save_dir, 'pipeline.joblib'))
    input_dim = X_full.shape[1]

    print(f"\nTraining Multi-Architecture & Multi-Seed Ensemble ({len(architectures)} Architectures x {len(seeds)} Seeds x 10 Folds)...")

    for arch in architectures:
        for seed in seeds:
            print(f"\n>>> Running 10-Fold CV: Architecture={arch.upper()}, Seed={seed}")
            config = {
                'architecture': arch,
                'hidden_dim': 512,
                'num_blocks': 3,
                'dropout': 0.15,
                'activation': 'gelu',
                'lr': 8e-4,
                'weight_decay': 1e-4,
                'batch_size': 16,
                'epochs': 160,
                'patience': 35,
                'n_splits': 10,
                'seed': seed,
                'input_dim': input_dim
            }
            oof_rmse, oof_preds_dollars, _ = train_kfold(df, config, save_dir=save_dir, verbose=False)
            print(f"    Finished -> OOF RMSE: ${oof_rmse:,.2f}")
            all_oof_predictions.append(oof_preds_dollars)

            for fold in range(10):
                checkpoint_manifest.append({
                    'filename': f'model_{arch}_seed_{seed}_fold_{fold}.pt',
                    'architecture': arch,
                    'seed': seed,
                    'fold': fold,
                    'hidden_dim': 512,
                    'num_blocks': 3,
                    'dropout': 0.15,
                    'activation': 'gelu'
                })

    final_ensemble_oof_dollars = np.mean(all_oof_predictions, axis=0)
    final_rmse = calculate_rmse(true_dollars, final_ensemble_oof_dollars)
    final_mape = np.mean(np.abs((true_dollars - final_ensemble_oof_dollars) / true_dollars)) * 100
    final_r2 = 1.0 - np.sum((true_dollars - final_ensemble_oof_dollars)**2) / np.sum((true_dollars - np.mean(true_dollars))**2)

    print(f"\n==========================================")
    print(f"FINAL BLENDED MULTI-SEED ENSEMBLE RESULTS:")
    print(f"  Ensemble Models Count : {len(checkpoint_manifest)}")
    print(f"  Overall OOF RMSE      : ${final_rmse:,.2f}")
    print(f"  Overall OOF R2        : {final_r2:.4f}")
    print(f"  Overall OOF MAPE      : {final_mape:.2f}%")
    print(f"==========================================")

    config_to_save = {
        'input_dim': input_dim,
        'overall_oof_rmse': float(final_rmse),
        'overall_oof_r2': float(final_r2),
        'overall_oof_mape': float(final_mape),
        'ensemble_manifest': checkpoint_manifest
    }

    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(config_to_save, f, indent=4)

    return final_rmse, final_ensemble_oof_dollars, config_to_save


def tune_hyperparameters(df: pd.DataFrame, n_trials: int = 8, save_dir: str = 'data/saved_models'):
    print(f"Starting Optuna Hyperparameter Optimization ({n_trials} trials)...")

    def objective(trial):
        config = {
            'architecture': trial.suggest_categorical('architecture', ['resnet', 'swiglu']),
            'hidden_dim': trial.suggest_categorical('hidden_dim', [256, 512]),
            'num_blocks': trial.suggest_int('num_blocks', 2, 4),
            'dropout': trial.suggest_float('dropout', 0.10, 0.20, step=0.05),
            'activation': trial.suggest_categorical('activation', ['gelu', 'silu']),
            'lr': trial.suggest_float('lr', 5e-4, 1.5e-3, log=True),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 5e-4, log=True),
            'batch_size': 16,
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
