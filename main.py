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
from src.train import train_kfold, tune_hyperparameters
from src.evaluate import evaluate_predictions
from src.utils import set_seed
from src.models import get_model

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
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    print("Loading training dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    print("Generating EDA plots...")
    generate_eda_plots(df)

    print("\n==========================================")
    print("  Phase 1: Architecture Comparison Benchmark")
    print("==========================================")

    architectures = ['standard', 'wide_deep', 'resnet']
    benchmark_results = {}

    for arch in architectures:
        print(f"\nBenchmarking architecture: {arch.upper()}")
        config = {
            'architecture': arch,
            'hidden_dim': 256,
            'num_blocks': 3,
            'dropout': 0.15,
            'activation': 'silu',
            'lr': 1e-3,
            'weight_decay': 1e-4,
            'batch_size': 32,
            'epochs': 100,
            'patience': 25,
            'n_splits': 5,
            'seed': 42
        }
        oof_rmse, _, _ = train_kfold(df, config, save_dir=f'data/saved_models/benchmark_{arch}', verbose=False)
        benchmark_results[arch] = oof_rmse
        print(f"  Architecture [{arch.upper()}] 5-Fold OOF RMSE: ${oof_rmse:,.2f}")

    best_arch = min(benchmark_results, key=benchmark_results.get)
    print(f"\nWinner Architecture: {best_arch.upper()} with OOF RMSE ${benchmark_results[best_arch]:,.2f}")

    print("\n==========================================")
    print("  Phase 2: Optuna Hyperparameter Search")
    print("==========================================")
    best_params = tune_hyperparameters(df, n_trials=12)

    print("\n==========================================")
    print("  Phase 3: Final 10-Fold Ensemble Training")
    print("==========================================")

    final_config = {
        'architecture': best_params.get('architecture', best_arch),
        'hidden_dim': best_params.get('hidden_dim', 256),
        'num_blocks': best_params.get('num_blocks', 3),
        'dropout': best_params.get('dropout', 0.15),
        'activation': best_params.get('activation', 'silu'),
        'lr': best_params.get('lr', 1e-3),
        'weight_decay': best_params.get('weight_decay', 1e-4),
        'batch_size': best_params.get('batch_size', 32),
        'epochs': 150,
        'patience': 30,
        'n_splits': 10,
        'seed': 42
    }

    final_oof_rmse, oof_preds_dollars, history = train_kfold(
        df, final_config, save_dir='data/saved_models', verbose=True
    )

    print("\n==========================================")
    print("  Phase 4: Residual Evaluation & Plots")
    print("==========================================")
    metrics = evaluate_predictions(df['SalePrice'].values, oof_preds_dollars, output_dir='docs/plots')

    summary = {
        'benchmark_results': benchmark_results,
        'best_architecture': best_arch,
        'best_hyperparameters': best_params,
        'final_metrics': metrics
    }
    with open('data/saved_models/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nAll experiments successfully completed!")
    print(f"Final Model Saved to 'data/saved_models/' with OOF RMSE: ${metrics['RMSE']:,.2f}")

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
    n_splits = config.get('n_splits', 10)
    input_dim = X_test_proc.shape[1]

    fold_preds_log = []

    for fold in range(n_splits):
        checkpoint_path = os.path.join(saved_models_dir, f'model_fold_{fold}.pt')
        if not os.path.exists(checkpoint_path):
            print(f"Warning: Checkpoint for fold {fold} not found. Skipping.")
            continue

        model = get_model(
            architecture=config.get('architecture', 'resnet'),
            input_dim=input_dim,
            hidden_dim=config.get('hidden_dim', 256),
            num_blocks=config.get('num_blocks', 3),
            dropout=config.get('dropout', 0.15),
            activation=config.get('activation', 'silu')
        ).to(device)

        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.eval()

        with torch.no_grad():
            preds = model(X_test_tensor.to(device)).cpu().numpy().ravel()
            fold_preds_log.append(preds)

    if not fold_preds_log:
        raise RuntimeError("No valid fold model checkpoints were loaded.")

    avg_preds_log = np.mean(fold_preds_log, axis=0)
    predictions_dollars = np.expm1(avg_preds_log)
    predictions_dollars = np.maximum(10000.0, predictions_dollars)

    output_df = pd.DataFrame({
        'Id': ids,
        'Prediction': np.round(predictions_dollars, 2)
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully generated predictions for {len(output_df)} samples.")
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
