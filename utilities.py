import numpy as np

def mean_squared_error(y_true, y_pred):
    """
    Compute the mean squared error between the true and predicted values

    Parameters:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values

    Returns:
        float: Mean squared error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = ((y_true - y_pred) ** 2).mean()
    return mse

def mean_absolute_error(y_true, y_pred):
    """
    Compute the mean absolute error between the true and predicted values

    Parameters:
        y_true (array-like): True target values
        y_pred (array-like): Predicted target values

    Returns:
        float: Mean absolute error
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mae = np.abs(y_true - y_pred).mean()

    return mae

def pearson_correlation_coefficient(x, y):
    """
    Compute the Pearson correlation coefficient between two arrays

    Parameters:
        x (list): Input array
        y (list): Input array

    Returns:
        tuple: A tuple containing Pearson's correlation coefficient and its associated p-value

    Notes:
        The function calculates the Pearson product-moment correlation coefficient and the p-value
    """
    # Throw an error if the length of the datasets is not the same
    if len(x) != len(y):
        raise ValueError("Input arrays must have the same length")

    N = len(x)

    # Calculate means of x and y.
    mean_x = sum(x) / N
    mean_y = sum(y) / N

    # Calculate the numerator and denominators for Pearson's correlation coefficient.
    numerator = sum((x[i] - mean_x) * (y[i] - mean_y) for i in range(N))
    denominator_x = (sum((x[i] - mean_x) ** 2 for i in range(N))) ** 0.5
    denominator_y = (sum((y[i] - mean_y) ** 2 for i in range(N))) ** 0.5

    # Calculate Pearson's correlation coefficient.
    r = numerator / (denominator_x * denominator_y)

    # Calculate the t-statistic and p-value.
    t_statistic = r * ((N - 2) ** 0.5) / ((1 - r**2) ** 0.5)
    p_value = 2 * (1 - abs(t_statistic))

    return (r, p_value)
