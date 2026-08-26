import numpy as np
from sklearn.metrics import mean_absolute_error, root_mean_squared_log_error, root_mean_squared_error

def evaluate_model(y_true, y_pred):
    y_pred_clipped = np.clip(y_pred, 0, None)

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "RMSLE": root_mean_squared_log_error(y_true, y_pred_clipped),
    }