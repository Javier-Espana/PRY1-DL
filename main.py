import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

from src.data_processing import TabularPreprocessor
from src.train import train_kfold
from src.evaluate import evaluate_predictions
from src.utils import set_seed, calculate_metrics, plot_actual_vs_predicted, plot_residuals
from src.models import get_model, ResNetMLP

def generate_eda_plots(df: pd.DataFrame, output_dir: str = 'docs/plots'):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1. Target distribution (Raw vs Log Transformed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['SalePrice'], kde=True, ax=axes[0], color='#2563EB', bins=40)
    axes[0].set_title('Distribucion de SalePrice (Sesgada)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('SalePrice ($)')
    axes[0].set_ylabel('Count')

    sns.histplot(np.log1p(df['SalePrice']), kde=True, ax=axes[1], color='#10B981', bins=40)
    axes[1].set_title('Distribucion de log1p(SalePrice) (Normalizada)', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('log1p(SalePrice)')
    axes[1].set_ylabel('Count')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'target_distribution.png'), dpi=300)
    plt.close()

    # 2. Correlation heatmap with SalePrice for top numeric features
    num_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[num_cols].corr()
    top_corr_cols = corr['SalePrice'].abs().sort_values(ascending=False).head(12).index

    plt.figure(figsize=(10, 8))
    sns.heatmap(df[top_corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm', vmin=-1, vmax=1)
    plt.title('Top 12 Variables Numericas Correlacionadas con SalePrice', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'), dpi=300)
    plt.close()

    # 3. OverallQual vs SalePrice Boxplot
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='OverallQual', y='SalePrice', data=df, palette='Blues')
    plt.title('SalePrice vs Overall Quality Rating', fontsize=14, fontweight='bold')
    plt.xlabel('Overall Quality (1-10)')
    plt.ylabel('SalePrice ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'quality_vs_saleprice.png'), dpi=300)
    plt.close()

    # 4. GrLivArea vs SalePrice Scatter
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x='GrLivArea', y='SalePrice', hue='OverallQual', palette='viridis', data=df, alpha=0.8)
    plt.title('Above Grade Living Area (GrLivArea) vs SalePrice', fontsize=14, fontweight='bold')
    plt.xlabel('Above Grade Living Area (sq ft)')
    plt.ylabel('SalePrice ($)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'grlivarea_vs_saleprice.png'), dpi=300)
    plt.close()

    print(f"EDA plots saved to '{output_dir}'.")

def run_training():
    set_seed(42)
    data_path = 'data/train.csv'
    save_dir = 'data/saved_models'
    plots_dir = 'docs/plots'
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    print("Loading training dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    print("Generating EDA plots...")
    generate_eda_plots(df)

    true_dollars = df['SalePrice'].values
    y_all_log = np.log1p(true_dollars)
    X_all = df.drop(columns=['SalePrice'])

    full_preprocessor = TabularPreprocessor()
    X_full = full_preprocessor.fit(X_all).transform(X_all)
    full_preprocessor.save(os.path.join(save_dir, 'pipeline.joblib'))
    input_dim = X_full.shape[1]
    print(f"Processed input dimension: {input_dim}")

    categories_dict = {}
    if full_preprocessor.nominal_cols_ and full_preprocessor.encoder_ is not None:
        for col, cats in zip(full_preprocessor.nominal_cols_, full_preprocessor.encoder_.categories_):
            categories_dict[col] = cats

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    seeds = [42, 123, 2024]
    manifest = []
    seed_oof_dollars = []

    for seed in seeds:
        print(f"\n==========================================")
        print(f"  Training Seed {seed} (10-Fold CV)")
        print(f"==========================================")
        from sklearn.model_selection import KFold
        from torch.optim import AdamW
        from torch.optim.lr_scheduler import CosineAnnealingLR

        set_seed(seed)
        kf = KFold(n_splits=10, shuffle=True, random_state=seed)
        oof_preds_log = np.zeros(len(df))

        for fold, (train_idx, val_idx) in enumerate(kf.split(X_all, y_all_log)):
            X_tr_df, y_tr_log = X_all.iloc[train_idx], y_all_log[train_idx]
            X_va_df, y_va_log = X_all.iloc[val_idx], y_all_log[val_idx]

            fold_prep = TabularPreprocessor(categories_dict=categories_dict)
            X_tr = fold_prep.fit(X_tr_df).transform(X_tr_df)
            X_va = fold_prep.transform(X_va_df)

            X_tr_t = torch.tensor(X_tr, dtype=torch.float32).to(device)
            y_tr_t = torch.tensor(y_tr_log, dtype=torch.float32).unsqueeze(1).to(device)
            X_va_t = torch.tensor(X_va, dtype=torch.float32).to(device)
            y_va_t = torch.tensor(y_va_log, dtype=torch.float32).unsqueeze(1).to(device)

            train_ds = torch.utils.data.TensorDataset(X_tr_t, y_tr_t)
            train_loader = torch.utils.data.DataLoader(train_ds, batch_size=16, shuffle=True)

            model = ResNetMLP(
                input_dim=input_dim,
                hidden_dim=512,
                num_blocks=2,
                dropout=0.20,
                activation='gelu'
            ).to(device)

            criterion = torch.nn.SmoothL1Loss()
            optimizer = AdamW(
                model.parameters(),
                lr=0.0007427678230287661,
                weight_decay=0.00023908329860076593
            )
            scheduler = CosineAnnealingLR(optimizer, T_max=150, eta_min=1e-6)

            best_vl = float('inf')
            best_preds = None
            pat, pat_c = 30, 0

            filename = f"model_resnet_seed_{seed}_fold_{fold}.pt"
            ckpt_path = os.path.join(save_dir, filename)

            for epoch in range(1, 151):
                model.train()
                for bx, by in train_loader:
                    optimizer.zero_grad()
                    p = model(bx)
                    loss = criterion(p, by)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                with torch.no_grad():
                    vp = model(X_va_t)
                    vl = criterion(vp, y_va_t).item()
                    if vl < best_vl:
                        best_vl = vl
                        best_preds = vp.cpu().numpy().ravel()
                        pat_c = 0
                        torch.save(model.state_dict(), ckpt_path)
                    else:
                        pat_c += 1

                if pat_c >= pat:
                    break

            oof_preds_log[val_idx] = best_preds
            manifest.append({
                'filename': filename,
                'architecture': 'resnet',
                'seed': seed,
                'fold': fold,
                'hidden_dim': 512,
                'num_blocks': 2,
                'dropout': 0.20,
                'activation': 'gelu'
            })

        oof_dollars = np.expm1(oof_preds_log)
        seed_rmse = np.sqrt(np.mean((true_dollars - oof_dollars)**2))
        print(f"Seed {seed} OOF RMSE: ${seed_rmse:,.2f}")
        seed_oof_dollars.append(oof_dollars)

    # 3-Seed Blended Ensemble
    champion_oof_dollars = np.mean(seed_oof_dollars, axis=0)
    champion_metrics = calculate_metrics(true_dollars, champion_oof_dollars)

    print("\n==========================================")
    print("CHAMPION MULTI-SEED ENSEMBLE RESULTS:")
    print("==========================================")
    print(f"  OOF RMSE : ${champion_metrics['RMSE']:,.2f}")
    print(f"  OOF MAE  : ${champion_metrics['MAE']:,.2f}")
    print(f"  OOF R2   : {champion_metrics['R2']:.4f}")
    print(f"  OOF MAPE : {champion_metrics['MAPE']:.2f}%")
    print("==========================================")

    # Generate diagnostic plots
    plot_actual_vs_predicted(
        true_dollars, champion_oof_dollars,
        title=f"Out-of-Fold Actual vs Predicted (RMSE: ${champion_metrics['RMSE']:,.2f}, R2: {champion_metrics['R2']:.4f})",
        save_path=os.path.join(plots_dir, "actual_vs_predicted.png")
    )
    plot_residuals(
        true_dollars, champion_oof_dollars,
        title="Out-of-Fold Residual Error Analysis",
        save_path=os.path.join(plots_dir, "residual_analysis.png")
    )

    with open(os.path.join(plots_dir, 'metrics.json'), 'w') as f:
        json.dump(champion_metrics, f, indent=4)

    # Save model config
    config_data = {
        'architecture': 'resnet',
        'hidden_dim': 512,
        'num_blocks': 2,
        'dropout': 0.20,
        'activation': 'gelu',
        'lr': 0.0007427678230287661,
        'weight_decay': 0.00023908329860076593,
        'batch_size': 16,
        'epochs': 150,
        'patience': 30,
        'n_splits': 10,
        'seeds': seeds,
        'input_dim': input_dim,
        'overall_oof_rmse': float(champion_metrics['RMSE']),
        'overall_oof_mae': float(champion_metrics['MAE']),
        'overall_oof_r2': float(champion_metrics['R2']),
        'overall_oof_mape': float(champion_metrics['MAPE']),
        'ensemble_manifest': manifest
    }

    with open(os.path.join(save_dir, 'model_config.json'), 'w') as f:
        json.dump(config_data, f, indent=4)

    print("\nTraining and ensemble saved to 'data/saved_models/'!")

def run_prediction(test_path: str, output_path: str, saved_models_dir: str = 'data/saved_models'):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset file not found at: {test_path}")

    pipeline_path = os.path.join(saved_models_dir, 'pipeline.joblib')
    config_path = os.path.join(saved_models_dir, 'model_config.json')

    if not os.path.exists(pipeline_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Saved models or pipeline not found in '{saved_models_dir}'. Please run training first!"
        )

    with open(config_path, 'r') as f:
        config = json.load(f)

    preprocessor = TabularPreprocessor.load(pipeline_path)

    print(f"Reading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    if 'Id' not in test_df.columns:
        raise ValueError("Test dataframe must contain an 'Id' column.")

    ids = test_df['Id'].values

    X_test_proc = preprocessor.transform(test_df)
    X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    input_dim = X_test_proc.shape[1]

    manifest = config.get('ensemble_manifest', [])
    all_preds_log = []

    if manifest and len(manifest) > 0:
        for item in manifest:
            filename = item['filename']
            checkpoint_path = os.path.join(saved_models_dir, filename)
            if not os.path.exists(checkpoint_path):
                continue

            model = ResNetMLP(
                input_dim=input_dim,
                hidden_dim=item.get('hidden_dim', 512),
                num_blocks=item.get('num_blocks', 2),
                dropout=item.get('dropout', 0.20),
                activation=item.get('activation', 'gelu')
            ).to(device)

            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()

            with torch.no_grad():
                preds = model(X_test_tensor.to(device)).cpu().numpy().ravel()
                all_preds_log.append(preds)
    else:
        # Fallback to default fold checkpoints
        for fold in range(10):
            checkpoint_path = os.path.join(saved_models_dir, f'model_fold_{fold}.pt')
            if not os.path.exists(checkpoint_path):
                continue

            model = ResNetMLP(
                input_dim=input_dim,
                hidden_dim=512,
                num_blocks=2,
                dropout=0.20,
                activation='gelu'
            ).to(device)

            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            model.eval()

            with torch.no_grad():
                preds = model(X_test_tensor.to(device)).cpu().numpy().ravel()
                all_preds_log.append(preds)

    if not all_preds_log:
        raise RuntimeError("No valid model checkpoints were loaded.")

    avg_preds_log = np.mean(all_preds_log, axis=0)
    predictions_dollars = np.expm1(avg_preds_log)
    predictions_dollars = np.maximum(10000.0, predictions_dollars)

    output_df = pd.DataFrame({
        'Id': ids,
        'Prediction': np.round(predictions_dollars, 2)
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully generated predictions for {len(output_df)} samples using ensemble of {len(all_preds_log)} models.")
    print(f"Output saved to: {output_path}\n")
    print("Sample output:")
    print(output_df.head())
    return output_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified ML Pipeline CLI")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Execution mode')

    # Train subparser
    train_parser = subparsers.add_parser('train', help='Run data analysis, tuning and model training')

    # Predict subparser
    predict_parser = subparsers.add_parser('predict', help='Generate test submission predictions')
    predict_parser.add_argument('--test_path', type=str, default='data/pruebas/pipeline_test.csv', help='Path to test CSV')
    predict_parser.add_argument('--output_path', type=str, default='data/pruebas/expected_output.csv', help='Path to output CSV')
    predict_parser.add_argument('--saved_models_dir', type=str, default='data/saved_models', help='Directory of saved models')

    args = parser.parse_args()

    if args.mode == 'train':
        run_training()
    elif args.mode == 'predict':
        run_prediction(args.test_path, args.output_path, args.saved_models_dir)
