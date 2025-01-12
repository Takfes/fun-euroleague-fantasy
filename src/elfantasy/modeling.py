import pandas as pd
from category_encoders import TargetEncoder
from lightgbm import LGBMRegressor
from sklearn.cluster import FeatureAgglomeration
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, PowerTransformer, StandardScaler
from xgboost import XGBRegressor

from elfantasy.utils import timeit


def build_pred_model(model_string, nums, cats, transform_target=False):
    # Preprocessing for numerical data
    numerical_transformer = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("polynomial_features", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("feature_agglomeration", FeatureAgglomeration(n_clusters=10)),
            # ("polynomial_features", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ]
    )

    # Preprocessing for categorical data
    categorical_transformer = TargetEncoder()

    # Bundle preprocessing for numerical and categorical data
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, nums),
            ("cat", categorical_transformer, cats),
        ]
    )

    # Define the models
    model_lgbm = LGBMRegressor(
        n_estimators=1000,
        learning_rate=0.001,
        num_leaves=71,
        max_depth=-1,
        min_child_samples=10,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        random_state=1990,
    )

    model_xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.001,
        max_depth=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=0.01,
        min_child_weight=1,
        random_state=1990,
    )

    model_knn = KNeighborsRegressor(
        n_neighbors=10,
        weights="uniform",
        algorithm="auto",
        leaf_size=30,
        p=2,
        metric="minkowski",
    )

    model_lr = LinearRegression()

    # Add the linear regression model to the list of models
    models = {
        "LGBM": model_lgbm,
        "XGB": model_xgb,
        "KNN": model_knn,
        "LR": model_lr,
    }

    # Choose the model for hyperparameter optimization
    model = models[model_string]  # or any other model from the models dictionary

    # Create and evaluate the pipeline
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])

    # Create a power transformer
    if transform_target:
        return TransformedTargetRegressor(regressor=pipeline, transformer=PowerTransformer(method="yeo-johnson"))
    else:
        return pipeline
