import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    """
    Implements binary classification prediction using Logistic Regression.

    Args:
        X: Input feature matrix (shape: N x D)
        weights: Model weights (shape: D)
        bias: Model bias

    Returns:
        Binary predictions (0 or 1)
    """
    # Your code here

    Xbar = np.column_stack((X, np.ones(X.shape[0])))
    w = np.append(weights, bias)

    Xbar = Xbar.T

    # tong tuyen tinh
    z = w @ Xbar

    # ham kich hoat
    a = sigmoid(z)

    y_pred = (a >= 0.5).astype(int)

    return y_pred
    pass