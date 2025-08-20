import pandas as pd
from sklearn.datasets import fetch_california_housing
from src.preprocessing import preprocess_data
from src.model import train_linear_regression, evaluate_model, save_model
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("Boston House Price Prediction Project")
    print("=====================================")

    # Load data
    print("1. Loading data...")
    housing = fetch_california_housing()
    df = pd.DataFrame(housing.data, columns=housing.feature_names)
    df['PRICE'] = housing.target

    # Preprocess data
    print("2. Preprocessing data...")
    X_train, X_test, y_train, y_test, scaler = preprocess_data(df)

    # Train model
    print("3. Training model...")
    model = train_linear_regression(X_train, y_train)

    # Evaluate model
    print("4. Evaluating model...")
    metrics, y_pred = evaluate_model(model, X_test, y_test)

    # Print results
    print("\nModel Performance Metrics:")
    print(f"Mean Absolute Error (MAE): {metrics['MAE']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['RMSE']:.4f}")
    print(f"R-squared (R2): {metrics['R2']:.4f}")

    # Save model
    print("5. Saving model...")
    save_model(model, 'models/linear_regression_model.pkl')
    print("Model saved to models/linear_regression_model.pkl")

    # Create visualization of predictions vs actual
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Actual vs Predicted House Prices')
    plt.savefig('images/actual_vs_predicted.png')
    plt.show()

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': housing.feature_names,
        'importance': model.coef_
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=feature_importance)
    plt.title('Feature Importance in Linear Regression Model')
    plt.tight_layout()
    plt.savefig('images/feature_importance.png')
    plt.show()

if __name__ == "__main__":
    main()