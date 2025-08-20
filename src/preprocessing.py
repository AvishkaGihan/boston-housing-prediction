import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, test_size=0.2, random_state=42):
    """
    Preprocess the housing data for modeling

    Parameters:
    df (DataFrame): The raw housing data
    test_size (float): Proportion of data to use for testing
    random_state (int): Random seed for reproducibility

    Returns:
    X_train, X_test, y_train, y_test, scaler: Processed data and scaler object
    """
    # Separate features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler