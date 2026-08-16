import os
import json
import numpy as np
import pandas as pd
from src.utils import calculate_metrics, plot_actual_vs_predicted, plot_residuals

def evaluate_predictions(y_true, y_pred, output_dir="docs/plots"):
    """
    Computes regression performance metrics and saves diagnostic plots.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = calculate_metrics(y_true, y_pred)
    print("\n========== Model Performance Metrics ==========")
    for k, v in metrics.items():
        if k in ['RMSE', 'MAE']:
            print(f"  {k:6s}: ${v:,.2f}")
        elif k == 'R2':
            print(f"  {k:6s}: {v:.4f}")
        elif k == 'MAPE':
            print(f"  {k:6s}: {v:.2f}%")
    print("===============================================")

    # Plot actual vs predicted
    plot_actual_vs_predicted(
        y_true, y_pred,
        title=f"Out-of-Fold Actual vs Predicted (RMSE: ${metrics['RMSE']:,.2f}, R2: {metrics['R2']:.4f})",
        save_path=os.path.join(output_dir, "actual_vs_predicted.png")
    )

    # Plot residual distribution
    plot_residuals(
        y_true, y_pred,
        title="Out-of-Fold Residual Error Analysis",
        save_path=os.path.join(output_dir, "residual_analysis.png")
    )

    # Save metrics JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics
