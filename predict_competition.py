import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import torch

from src.data_processing import TabularPreprocessor
from src.models import get_model

def predict(test_path: str, output_path: str, saved_models_dir: str = 'saved_models'):
    """
    Consumes test CSV dataset, applies pre-fitted preprocessor, loads trained fold models,
    generates ensembled predictions, and writes output matching expected_output.csv format.
    """
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset file not found at: {test_path}")

    pipeline_path = os.path.join(saved_models_dir, 'pipeline.joblib')
    config_path = os.path.join(saved_models_dir, 'model_config.json')

    if not os.path.exists(pipeline_path) or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Saved models or pipeline not found in '{saved_models_dir}'. Please run model training first!"
        )

    # 1. Load config and preprocessor
    with open(config_path, 'r') as f:
        config = json.load(f)

    preprocessor = TabularPreprocessor.load(pipeline_path)

    # 2. Read test dataset
    print(f"Reading test data from: {test_path}")
    test_df = pd.read_csv(test_path)
    
    if 'Id' not in test_df.columns:
        raise ValueError("Test dataframe must contain an 'Id' column.")

    ids = test_df['Id'].values

    # 3. Transform test features
    X_test_proc = preprocessor.transform(test_df)
    X_test_tensor = torch.tensor(X_test_proc, dtype=torch.float32)

    # 4. Load all trained fold models and compute ensembled predictions
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

    # 5. Average fold log predictions and invert log1p transformation
    avg_preds_log = np.mean(fold_preds_log, axis=0)
    predictions_dollars = np.expm1(avg_preds_log)

    # Ensure non-negative realistic predictions
    predictions_dollars = np.maximum(10000.0, predictions_dollars)

    # 6. Format output DataFrame matching expected_output.csv
    output_df = pd.DataFrame({
        'Id': ids,
        'Prediction': np.round(predictions_dollars, 2)
    })

    # Ensure target output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    output_df.to_csv(output_path, index=False)

    print(f"\nSuccessfully generated predictions for {len(output_df)} samples.")
    print(f"Output saved to: {output_path}\n")
    print("Sample output:")
    print(output_df.head())
    return output_df

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate competition test predictions.")
    parser.add_argument('--test_path', type=str, default='data/pruebas/pipeline_test.csv', help='Path to test dataset CSV')
    parser.add_argument('--output_path', type=str, default='data/pruebas/output.csv', help='Path to output predictions CSV')
    parser.add_argument('--saved_models_dir', type=str, default='saved_models', help='Directory containing trained models & pipeline')

    args = parser.parse_args()
    predict(args.test_path, args.output_path, args.saved_models_dir)
