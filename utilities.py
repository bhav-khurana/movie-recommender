import numpy as np

def mean_squared_error(predictions, targets):
    return np.mean((predictions - targets) ** 2)

def mean_absolute_error(predictions, targets):
    return np.mean(np.abs(predictions - targets))

def pearsonr(x, y):
    # Avoid division by zero
    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0
    
    return np.cov(x, y)[0, 1] / (np.std(x) * np.std(y))
