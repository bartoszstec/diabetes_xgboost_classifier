# Project Overview
A machine learning model for predicting diabetes based on biometric blood-test data from Kaggle.

# Main assumption:
- Algorithm: XGBoost (gradient boosting decision trees)
- Loss function: Binary cross-entropy
- Environment: Research/development only (not planning web interface)
- Dataset: [health test by blood dataset - Kaggle](https://www.kaggle.com/datasets/simaanjali/diabetes-classification-dataset)

# Project Structure (planned):
- data/ - dataset (not included in repo)
- src/ - model training scripts
- models/ - saved models
## Goals
- Build a clean, reproducible ML workflow
- Evaluate baseline and optimized XGBoost model
- Analyze which biometric factors contribute most to prediction
- REALLY NOT IMPLEMENTING WEB INTERFACE (optional)