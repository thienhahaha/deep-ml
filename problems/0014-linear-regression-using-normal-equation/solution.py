import numpy as np
def linear_regression_normal_equation(X: list[list[float]], y: list[float]) -> list[float]:
    # Your code here, make sure to round

    X = np.array(X)
    Y = np.array(y)
    Xbar = X.T
    Y = Y.T

    theta = theta = np.linalg.inv(X.T @ X) @ X.T @ Y

    return np.round(theta.flatten(), 4).tolist()
