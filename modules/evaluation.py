# modules/evaluation.py
import numpy as np
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def categorize_loss(value):
    """
    Categorize continuous soil loss into discrete classes.

    Args:
        value (float): soil loss value

    Returns:
        int: category label
    """
    if value == 0:
        return 0
    if value < 0.25:
        return 1
    if value < 1.0:
        return 2
    if value < 2.0:
        return 3
    if value < 5.0:
        return 4
    return 5


def calculate_metrics(y_true, y_pred, threshold=0.25):
    """
    Compute regression and classification metrics.

    Args:
        y_true (array-like): true target values
        y_pred (array-like): predicted values
        threshold (float): cutoff for binary classification

    Returns:
        dict: metrics {mse, rmse, mae, accuracy, precision, recall, f1}
    """
    # Regression metrics
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)

    # Binary classification
    bin_true = (y_true >= threshold).astype(int)
    bin_pred = (y_pred >= threshold).astype(int)

    # Multi-class classification
    true_cat = [categorize_loss(v) for v in y_true]
    pred_cat = [categorize_loss(v) for v in y_pred]

    accuracy = accuracy_score(bin_true, bin_pred)
    precision = precision_score(true_cat, pred_cat, average='weighted', zero_division=0)
    recall = recall_score(true_cat, pred_cat, average='weighted', zero_division=0)
    f1 = f1_score(true_cat, pred_cat, average='weighted', zero_division=0)

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }



