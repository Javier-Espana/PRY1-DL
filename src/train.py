import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import KFold

from src.data_processing import TabularPreprocessor
from src.dataset import create_dataloaders
from src.models import ResNetMLP
from src.utils import set_seed, calculate_rmse, calculate_metrics

def train_single_model(df: pd.DataFrame, save_dir: str = 'data/saved_models', verbose: bool = True):
    """
    Train and save the single, best-balanced Tabular ResNet-MLP model.
    Evaluates with 10-Fold CV for unbiased metrics, then saves final model.pt.
    """
    os.makedirs(save_dir, exist_ok=True)
    seed = 42
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if verbose:
        print(f"Training single champion model on device: {device}")

    y_all_log = np.log1p(df['SalePrice'].values)
    X_all = df.drop(columns=['SalePrice'])
    true_dollars = df['SalePrice'].values

    # Fit preprocessor on full data
    full_preprocessor = TabularPreprocessor()
    X_full_proc = full_preprocessor.fit_transform(X_all)
    full_preprocessor.save(os.path.join(save_dir, 'pipeline.joblib'))
    input_dim = X_full_proc.shape[1]

    if verbose:
        print(f"Preprocessed input dimension: {input_dim}")

    categories_dict = {}
    if full_preprocessor.nominal_cols_ and full_preprocessor.encoder_ is not None:
        for col, cats in zip(full_preprocessor.nominal_cols_, full_preprocessor.encoder_.categories_):
            categories_dict[col] = cats

    # 10-Fold CV for unbiased out-of-fold generalization evaluation
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    oof_preds_log = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all_log)):
        X_train_df, y_train_log = X_all.iloc[train_idx], y_all_log[train_idx]
        X_val_df, y_val_log = X_all.iloc[val_idx], y_all_log[val_idx]

        fold_preprocessor = TabularPreprocessor(categories_dict=categories_dict)
        X_train_proc = fold_preprocessor.fit_transform(X_train_df)
        X_val_proc = fold_preprocessor.transform(X_val_df)

        train_loader, val_loader = create_dataloaders(
            X_train_proc, y_train_log, X_val_proc, y_val_log,
            batch_size=16
        )

        model = ResNetMLP(
            input_dim=input_dim,
            hidden_dim=512,
            num_blocks=2,
            dropout=0.20,
            activation='gelu'
        ).to(device)

        criterion = nn.SmoothL1Loss()
        optimizer = AdamW(
            model.parameters(),
            lr=0.0007427678230287661,
            weight_decay=0.00023908329860076593
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

        best_val_loss = float('inf')
        best_preds = None
        patience, patience_counter = 30, 0

        for epoch in range(1, 151):
            model.train()
            for bx, by in train_loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                p = model(bx)
                loss = criterion(p, by)
                loss.backward()
                optimizer.step()
            scheduler.step()

            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                fold_preds = []
                for bx, by in val_loader:
                    bx, by = bx.to(device), by.to(device)
                    p = model(bx)
                    val_loss += criterion(p, by).item() * len(bx)
                    fold_preds.append(p.cpu().numpy())
                val_loss /= len(val_loader.dataset)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_preds = np.vstack(fold_preds).ravel()
                    patience_counter = 0
                else:
                    patience_counter += 1

            if patience_counter >= patience:
                break

        oof_preds_log[val_idx] = best_preds

    oof_preds_dollars = np.expm1(oof_preds_log)
    metrics = calculate_metrics(true_dollars, oof_preds_dollars)

    if verbose:
        print("\n==========================================")
        print("TABULAR RESNET-MLP 10-FOLD CV RESULTS:")
        print(f"  OOF RMSE : ${metrics['RMSE']:,.2f}")
        print(f"  OOF MAE  : ${metrics['MAE']:,.2f}")
        print(f"  OOF R2   : {metrics['R2']:.4f}")
        print(f"  OOF MAPE : {metrics['MAPE']:.2f}%")
        print("==========================================")

    # Train final single model on full training set
    if verbose:
        print("\nFitting final single model weights on full dataset...")

    X_full_tensor = torch.tensor(X_full_proc, dtype=torch.float32).to(device)
    y_full_tensor = torch.tensor(y_all_log, dtype=torch.float32).unsqueeze(1).to(device)
    full_dataset = torch.utils.data.TensorDataset(X_full_tensor, y_full_tensor)
    full_loader = torch.utils.data.DataLoader(full_dataset, batch_size=16, shuffle=True)

    final_model = ResNetMLP(
        input_dim=input_dim,
        hidden_dim=512,
        num_blocks=2,
        dropout=0.20,
        activation='gelu'
    ).to(device)

    criterion = nn.SmoothL1Loss()
    optimizer = AdamW(
        final_model.parameters(),
        lr=0.0007427678230287661,
        weight_decay=0.00023908329860076593
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=120, eta_min=1e-6)

    final_model.train()
    for epoch in range(1, 121):
        for bx, by in full_loader:
            optimizer.zero_grad()
            p = final_model(bx)
            loss = criterion(p, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    # Save single model file
    model_file_path = os.path.join(save_dir, 'model.pt')
    torch.save(final_model.state_dict(), model_file_path)

    config_data = {
        'architecture': 'resnet',
        'hidden_dim': 512,
        'num_blocks': 2,
        'dropout': 0.20,
        'activation': 'gelu',
        'lr': 0.0007427678230287661,
        'weight_decay': 0.00023908329860076593,
        'batch_size': 16,
        'epochs': 120,
        'input_dim': input_dim,
        'validation_strategy': '10-Fold Cross Validation',
        'oof_rmse': float(metrics['RMSE']),
        'oof_mae': float(metrics['MAE']),
        'oof_r2': float(metrics['R2']),
        'oof_mape': float(metrics['MAPE']),
        'model_file': 'model.pt',
        'pipeline_file': 'pipeline.joblib'
    }

    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)

    return metrics, oof_preds_dollars, config_data
