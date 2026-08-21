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
from src.train import train_single_model
from src.evaluate import evaluate_predictions
from src.utils import set_seed
from src.models import ResNetMLP

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
    generate_eda_plots(df, output_dir=plots_dir)

    print("\nTraining and evaluating single balanced Tabular ResNet-MLP model...")
    metrics, oof_preds_dollars, config = train_single_model(df, save_dir=save_dir, verbose=True)

    print("\nGenerating evaluation and residual plots...")
    evaluate_predictions(df['SalePrice'].values, oof_preds_dollars, output_dir=plots_dir)

    print(f"\nTraining completed successfully! Model saved to '{save_dir}/model.pt'.")
    print(f"Final OOF RMSE: ${metrics['RMSE']:,.2f} | R2: {metrics['R2']:.4f} | MAPE: {metrics['MAPE']:.2f}%")

def run_prediction(test_path: str, output_path: str, saved_models_dir: str = 'data/saved_models'):
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset file not found at: {test_path}")

    pipeline_path = os.path.join(saved_models_dir, 'pipeline.joblib')
    model_path = os.path.join(saved_models_dir, 'model.pt')
    config_path = os.path.join(saved_models_dir, 'model_config.json')

    if not os.path.exists(pipeline_path) or not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Saved model or pipeline not found in '{saved_models_dir}'. Please run training first!"
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

    model = ResNetMLP(
        input_dim=input_dim,
        hidden_dim=config.get('hidden_dim', 512),
        num_blocks=config.get('num_blocks', 2),
        dropout=config.get('dropout', 0.20),
        activation=config.get('activation', 'gelu')
    ).to(device)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        preds_log = model(X_test_tensor.to(device)).cpu().numpy().ravel()

    predictions_dollars = np.expm1(preds_log)
    predictions_dollars = np.maximum(10000.0, predictions_dollars)

    output_df = pd.DataFrame({
        'Id': ids,
        'Prediction': np.round(predictions_dollars, 2)
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully generated predictions for {len(output_df)} samples using Tabular ResNet model.")
    print(f"Output saved to: {output_path}\n")
    print("Sample output:")
    print(output_df.head())
    return output_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Unified ML Pipeline CLI")
    subparsers = parser.add_subparsers(dest='mode', required=True, help='Execution mode')

    # Train subparser
    train_parser = subparsers.add_parser('train', help='Run data analysis and single model training')

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
