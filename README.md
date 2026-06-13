# California Housing Price Prediction

## Project Overview

This project predicts California house prices using Machine Learning.

The model is trained using the California Housing Dataset and uses a Random Forest Regressor to estimate house prices based on housing-related features.

---

## Features Used

- Longitude
- Latitude
- Housing Median Age
- Total Rooms
- Total Bedrooms
- Population
- Households
- Median Income
- Ocean Proximity

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Joblib

---

## Machine Learning Pipeline

1. Stratified Train-Test Split
2. Missing Value Imputation
3. Feature Scaling
4. One-Hot Encoding
5. Random Forest Regression

---

## Evaluation Metrics

- RMSE
- R² Score

---

## Files

- housing.csv → Dataset
- train.py → Main project code
- model.pkl → Trained model
- pipeline.pkl → Preprocessing pipeline
- input.csv → Input data for prediction
- output.csv → Predicted results

---

## How to Run

Install dependencies:

```bash
pip install pandas numpy scikit-learn joblib
```

Run:

```bash
python train.py
```

---

## Author

Faizan Shaikh
=======
# California-Housing-Price-Prediction
Machine Learning project that predicts California house prices using Random Forest Regression, data preprocessing pipelines, feature scaling, one-hot encoding, and stratified sampling.
