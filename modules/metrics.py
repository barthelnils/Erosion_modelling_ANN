import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    f1_score, precision_score, recall_score, accuracy_score
)


def categorize(v):
    if v == 0: return 0
    if v < 0.25: return 1
    if v < 1.0: return 2
    if v < 2.0: return 3
    if v < 5.0: return 4
    return 5


def regression_metrics(y, yp):
    mse = mean_squared_error(y, yp)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y, yp)
    return {"mse": mse, "rmse": rmse, "mae": mae}


def classification_metrics(y, yp):
    yt = np.vectorize(categorize)(y)
    yp = np.vectorize(categorize)(yp)
    return {
        "f1": f1_score(yt, yp, average="weighted", zero_division=0),
        "precision": precision_score(yt, yp, average="weighted", zero_division=0),
        "recall": recall_score(yt, yp, average="weighted", zero_division=0),
        "accuracy": accuracy_score(yt, yp)
    }


def spatial_corr(y_full, y_pred_full, mask):
    valid = mask & np.isfinite(y_full) & np.isfinite(y_pred_full)
    if valid.sum() < 2:
        return {"spatial_corr": np.nan}
    corr = np.corrcoef(y_full[valid], y_pred_full[valid])[0,1]
    return {"spatial_corr": float(corr)}
