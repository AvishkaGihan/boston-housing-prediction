from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib
import os

def train_linear_regression(X_train, y_train):
    """
    Train a linear regression model

    Parameters:
    X_train: Scaled training features
    y_train: Training target values

    Returns:
    model: Trained linear regression model
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model performance

    Parameters:
    model: Trained model
    X_test: Scaled test features
    y_test: Test target values

    Returns:
    metrics: Dictionary containing MAE, RMSE, and R2 scores
    """
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = model.score(X_test, y_test)

    metrics = {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2': r2
    }

    return metrics, y_pred

def save_model(model, filename):
    """
    Save the trained model to disk

    Parameters:
    model: Trained model to save
    filename: Path to save the model
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)

def load_model(filename):
    """
    Load a trained model from disk

    Parameters:
    filename: Path to the saved model

    Returns:
    model: Loaded model
    """
    return joblib.load(filename)