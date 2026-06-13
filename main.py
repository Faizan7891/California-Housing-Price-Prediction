import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"


def build_pipeline(num_attribs, cat_attribs):
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_attribs)
    ])

    return full_pipeline


if not os.path.exists(MODEL_FILE):

    print("Training model...\n")

    # Load dataset
    housing = pd.read_csv("housing.csv")

    # Create income categories for stratified sampling
    housing["income_cat"] = pd.cut(
        housing["median_income"],
        bins=[0.0, 1.5, 3.0, 4.5, 6.0, np.inf],
        labels=[1, 2, 3, 4, 5]
    )

    # Stratified split
    split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=42
    )

    for train_index, test_index in split.split(
        housing,
        housing["income_cat"]
    ):
        train_set = housing.iloc[train_index].drop(
            "income_cat",
            axis=1
        )

        test_set = housing.iloc[test_index].drop(
            "income_cat",
            axis=1
        )

    # Save test set for inference
    test_set.drop(
        "median_house_value",
        axis=1
    ).to_csv(
        "input.csv",
        index=False
    )

    # Separate labels and features
    housing_labels = train_set["median_house_value"].copy()

    housing_features = train_set.drop(
        "median_house_value",
        axis=1
    )

    # Numerical and categorical columns
    num_attribs = housing_features.drop(
        "ocean_proximity",
        axis=1
    ).columns.tolist()

    cat_attribs = ["ocean_proximity"]

    # Build pipeline
    pipeline = build_pipeline(
        num_attribs,
        cat_attribs
    )

    # Prepare data
    housing_prepared = pipeline.fit_transform(
        housing_features
    )

    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(
        housing_prepared,
        housing_labels
    )

    # Evaluate on training data
    predictions = model.predict(
        housing_prepared
    )

    rmse = np.sqrt(
        mean_squared_error(
            housing_labels,
            predictions
        )
    )

    r2 = r2_score(
        housing_labels,
        predictions
    )

    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")

    # Feature Importance
    feature_names = pipeline.get_feature_names_out()

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    })

    importance_df = importance_df.sort_values(
        by="Importance",
        ascending=False
    )

    print("\nTop 10 Important Features:\n")
    print(importance_df.head(10))

    # Save model and pipeline
    joblib.dump(
        model,
        MODEL_FILE
    )

    joblib.dump(
        pipeline,
        PIPELINE_FILE
    )

    print("\nModel and pipeline saved.")

else:

    print("Running inference...\n")

    # Load model and pipeline
    model = joblib.load(
        MODEL_FILE
    )

    pipeline = joblib.load(
        PIPELINE_FILE
    )

    # Read input
    input_data = pd.read_csv(
        "input.csv"
    )

    # Transform
    transformed_input = pipeline.transform(
        input_data
    )

    # Predict
    predictions = model.predict(
        transformed_input
    )

    # Save output
    input_data[
        "median_house_value"
    ] = predictions

    input_data.to_csv(
        "output.csv",
        index=False
    )

    print("Predictions saved to output.csv")