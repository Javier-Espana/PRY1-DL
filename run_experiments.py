import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_processing import TabularPreprocessor
from src.train import train_kfold, tune_hyperparameters
from src.evaluate import evaluate_predictions
from src.utils import set_seed

def generate_eda_plots(df: pd.DataFrame, output_dir: str = 'docs/plots'):
    os.makedirs(output_dir, exist_ok=True)
    sns.set_theme(style='whitegrid')

    # 1. Target distribution (Raw vs Log Transformed)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    sns.histplot(df['SalePrice'], kde=True, ax=axes[0], color='#2563EB', bins=40)
    axes[0].set_title('Raw SalePrice Distribution (Skewed)', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('SalePrice ($)')
    axes[0].set_ylabel('Count')

    sns.histplot(np.log1p(df['SalePrice']), kde=True, ax=axes[1], color='#10B981', bins=40)
    axes[1].set_title('Log Transformed log1p(SalePrice) (Normal)', fontsize=13, fontweight='bold')
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
    plt.title('Top 12 Correlated Numerical Features with SalePrice', fontsize=14, fontweight='bold')
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

def main():
    set_seed(42)
    data_path = 'data/train.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Training dataset not found at {data_path}")

    print("Loading training dataset...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")

    # Generate EDA plots
    print("Generating EDA plots...")
    generate_eda_plots(df)

    # Compare Baseline Architectures (5-Fold for fast benchmarking)
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
        oof_rmse, _, _ = train_kfold(df, config, save_dir=f'saved_models/benchmark_{arch}', verbose=False)
        benchmark_results[arch] = oof_rmse
        print(f"  Architecture [{arch.upper()}] 5-Fold OOF RMSE: ${oof_rmse:,.2f}")

    best_arch = min(benchmark_results, key=benchmark_results.get)
    print(f"\nWinner Architecture: {best_arch.upper()} with OOF RMSE ${benchmark_results[best_arch]:,.2f}")

    # Phase 2: Hyperparameter Tuning
    print("\n==========================================")
    print("  Phase 2: Optuna Hyperparameter Search")
    print("==========================================")
    best_params = tune_hyperparameters(df, n_trials=12)

    # Phase 3: Final 10-Fold CV Training with Best Config
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
        df, final_config, save_dir='saved_models', verbose=True
    )

    # Phase 4: Full Evaluation & Plotting
    print("\n==========================================")
    print("  Phase 4: Residual Evaluation & Plots")
    print("==========================================")
    metrics = evaluate_predictions(df['SalePrice'].values, oof_preds_dollars, output_dir='docs/plots')

    # Save benchmark & tuning summaries
    summary = {
        'benchmark_results': benchmark_results,
        'best_architecture': best_arch,
        'best_hyperparameters': best_params,
        'final_metrics': metrics
    }
    with open('saved_models/experiment_summary.json', 'w') as f:
        json.dump(summary, f, indent=4)

    print("\nAll experiments successfully completed!")
    print(f"Final Model Saved to 'saved_models/' with OOF RMSE: ${metrics['RMSE']:,.2f}")

if __name__ == '__main__':
    main()
