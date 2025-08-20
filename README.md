# 🏠 Boston House Prices Prediction with Linear Regression

<div align="center">

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Complete-success.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

_A complete machine learning project implementing Linear Regression to predict house prices based on various demographic and geographic features._

[View Demo](#-results) · [Report Bug](../../issues) · [Request Feature](../../issues)

</div>

---

## 📋 Table of Contents

- [🌟 Project Overview](#-project-overview)
- [🏠 Dataset](#-dataset)
- [📊 Results](#-results)
- [🗂️ Project Structure](#️-project-structure)
- [🚀 Installation & Usage](#-installation--usage)
- [🧠 Key Learning Objectives](#-key-learning-objectives)
- [🔮 Future Extensions](#-future-extensions)
- [📚 Resources](#-resources)
- [👨‍💻 Author](#-author)
- [📄 License](#-license)

---

## 🌟 Project Overview

This project demonstrates the complete end-to-end workflow of a machine learning regression problem. Using the California Housing Dataset, we build and evaluate a Linear Regression model that predicts median house values based on demographic and geographic features.

### 🎯 **Objectives**

- Build a robust Linear Regression model for house price prediction
- Demonstrate proper ML project structure and workflow
- Apply data preprocessing and feature scaling techniques
- Evaluate model performance using multiple metrics
- Create visualizations for better understanding of the data and results

### 🛠️ **Technologies Used**

- **Python 3.11** - Programming language
- **Scikit-learn** - Machine learning library
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib & Seaborn** - Data visualization
- **Jupyter Notebook** - Interactive development

---

## 🏠 Dataset

> **Note:** This project uses the **California Housing Dataset** from `scikit-learn` as a modern, ethically sound alternative to the deprecated Boston Housing Dataset.

### 📊 **Features Overview**

| Feature      | Description                              | Type       |
| ------------ | ---------------------------------------- | ---------- |
| `MedInc`     | Median income in block group             | Continuous |
| `HouseAge`   | Median house age in block group          | Continuous |
| `AveRooms`   | Average number of rooms per household    | Continuous |
| `AveBedrms`  | Average number of bedrooms per household | Continuous |
| `Population` | Block group population                   | Continuous |
| `AveOccup`   | Average number of household members      | Continuous |
| `Latitude`   | Block group latitude                     | Continuous |
| `Longitude`  | Block group longitude                    | Continuous |

### 🎯 **Target Variable**

- **`PRICE`** - Median house value for California districts (in hundreds of thousands of dollars)

---

## 📊 Results

Our Linear Regression model achieved strong performance on the test dataset:

<div align="center">

### 🎯 **Model Performance Metrics**

| Metric                             | Value  | Interpretation                                    |
| :--------------------------------- | :----- | :------------------------------------------------ |
| **Mean Absolute Error (MAE)**      | `1.29` | Average prediction error of $129,000              |
| **Root Mean Squared Error (RMSE)** | `1.63` | Standard deviation of prediction errors: $163,000 |
| **R-squared (R²)**                 | `0.75` | Model explains 75% of price variance              |

</div>

### 📈 **Key Insights**

- The model successfully captures the majority of price variation in the dataset
- MAE of 1.29 indicates reasonable prediction accuracy for housing prices
- R² of 0.75 demonstrates good model fit without overfitting

### 🔍 **Feature Importance Analysis**

The linear regression coefficients reveal which features most strongly influence house prices:

- **Median Income** - Strongest positive predictor
- **Average Rooms** - Significant positive impact
- **House Age** - Moderate negative correlation
- **Location (Latitude/Longitude)** - Geographic clustering effects

---

## 🗂️ Project Structure

```
boston-housing-prediction/
│
├── 📁 data/                     # Raw and processed datasets
│   └── (external data files)
│
├── 📁 notebooks/                # Jupyter notebooks for analysis
│   └── exploration.ipynb       # EDA and data investigation
│
├── 📁 src/                      # Source code modules
│   ├── __init__.py             # Package initialization
│   ├── data_loading.py         # Data loading utilities
│   ├── preprocessing.py        # Data cleaning and scaling
│   ├── model.py               # Model training and evaluation
│   └── evaluation.py          # Metrics and visualization
│
├── 📁 models/                   # Trained model artifacts
│   └── linear_regression_model.pkl
│
├── 📁 images/                   # Generated visualizations
│   ├── actual_vs_predicted.png
│   ├── feature_importance.png
│   ├── feature_distributions.png
│   ├── correlation_matrix.png
│   └── price_relationships.png
│
├── 📄 .gitignore               # Git ignore rules
├── 📄 requirements.txt         # Python dependencies
├── 📄 README.md               # Project documentation
└── 📄 main.py                 # Main execution script
```

---

## 🚀 Installation & Usage

### ⚙️ **Prerequisites**

- Python 3.8 or higher
- pip or conda package manager

### 📥 **Quick Start**

1. **Clone the repository**

   ```bash
   https://github.com/AvishkaGihan/boston-housing-prediction.git
   cd boston-housing-prediction
   ```

2. **Set up environment**

   **Option A: Using conda (Recommended)**

   ```bash
   conda create -n boston_env --file requirements.txt
   conda activate boston_env
   ```

   **Option B: Using pip**

   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Run the complete pipeline**

   ```bash
   python main.py
   ```

4. **Explore with Jupyter**
   ```bash
   jupyter notebook notebooks/exploration.ipynb
   ```

### 🎮 **Usage Examples**

```python
# Quick prediction example
from src.model import LinearRegressionModel
from src.data_loading import load_california_housing

# Load data and train model
X_train, X_test, y_train, y_test = load_california_housing()
model = LinearRegressionModel()
model.train(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)
print(f"Model R² Score: {model.score(X_test, y_test):.3f}")
```

---

## 🧠 Key Learning Objectives

This project covers essential machine learning concepts:

- **🎯 ML Fundamentals**

  - Supervised learning principles
  - Feature selection and engineering
  - Train/validation/test splits

- **📊 Linear Regression**

  - Mathematical foundations
  - Coefficient interpretation
  - Assumptions and limitations

- **🔍 Model Evaluation**

  - Multiple evaluation metrics
  - Bias-variance tradeoff
  - Performance visualization

- **🛠️ Data Preprocessing**

  - Feature scaling with StandardScaler
  - Handling missing values
  - Data distribution analysis

- **📈 Project Management**
  - Structured code organization
  - Version control best practices
  - Documentation and reproducibility

---

## 🔮 Future Extensions

### 🚀 **Model Improvements**

- [ ] **Algorithm Comparison** - Test Random Forest, Gradient Boosting, XGBoost
- [ ] **Hyperparameter Tuning** - Grid search and cross-validation
- [ ] **Feature Engineering** - Polynomial features, interaction terms
- [ ] **Ensemble Methods** - Combine multiple models for better performance

### 🌐 **Deployment Options**

- [ ] **Web Application** - Flask/Streamlit interface for predictions
- [ ] **REST API** - FastAPI service with Docker containerization
- [ ] **Cloud Deployment** - AWS/GCP/Azure hosting
- [ ] **Mobile App** - React Native or Flutter integration

### 📊 **Advanced Analytics**

- [ ] **Time Series Analysis** - Predict price trends over time
- [ ] **Geospatial Analysis** - Interactive maps with Folium
- [ ] **A/B Testing Framework** - Compare model versions
- [ ] **MLOps Pipeline** - Automated training and deployment

---

## 📚 Resources

### 📖 **Learning Materials**

- [Scikit-learn Documentation](https://scikit-learn.org/stable/) - Official ML library docs
- [California Housing Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html) - Dataset details
- [Linear Regression Theory](https://en.wikipedia.org/wiki/Linear_regression) - Mathematical foundations

### 🎓 **Recommended Reading**

- _Hands-On Machine Learning_ by Aurélien Géron
- _The Elements of Statistical Learning_ by Hastie, Tibshirani, and Friedman
- _Introduction to Statistical Learning_ by James, Witten, Hastie, and Tibshirani

---

## 👨‍💻 Author

<div align="center">

**Avishka Gihan**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/AvishkaGihan)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/avishkagihan/)
[![Email](https://img.shields.io/badge/Email-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:avishkag18@gmail.com)

_Passionate about machine learning and data science_

</div>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

### 🌟 **Found this project helpful?**

Give it a star ⭐️ and feel free to contribute!

[![GitHub stars](https://img.shields.io/github/stars/AvishkaGihan/boston-housing-prediction?style=social)](https://github.com/AvishkaGihan/boston-housing-prediction/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/AvishkaGihan/boston-housing-prediction?style=social)](https://github.com/AvishkaGihan/boston-housing-prediction/network/members)

</div>

---

<div align="center">
  <sub>Built with ❤️ by <a href="https://github.com/AvishkaGihan">Avishka Gihan</a></sub>
</div>
