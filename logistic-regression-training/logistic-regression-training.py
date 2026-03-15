import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.01, steps=1000):
    N, D = X.shape
    X=np.array(X,dtype=float)
    y=np.array(y,dtype=float)
    # Initialize parameters
    w = np.zeros(D)
    b = 0.0

    for _ in range(steps):
        # Forward pass
        z = X @ w + b
        p = 1 / (1 + np.exp(-z))  # sigmoid

        # Compute gradients
        error = p - y
        dw = (X.T @ error) / N
        db = np.sum(error) / N

        # Update parameters
        w -= lr * dw
        b -= lr * db

    return w, b
        