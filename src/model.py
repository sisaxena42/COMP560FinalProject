# src/model.py

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score


DATA_PATH_DEFAULT = "data/merged.csv"

# All numeric feature columns we want to use (drop IDs / strings / messy columns)
FEATURE_COLUMNS: List[str] = [
    "Unsafe water source",
    "Unsafe sanitation",
    "No access to handwashing facility",
    "Household air pollution from solid fuels",
    "Non-exclusive breastfeeding",
    "Discontinued breastfeeding",
    "Child wasting",
    "Child stunting",
    "Low birth weight for gestation",
    "Secondhand smoke",
    "Alcohol use",
    "Drug use",
    "Diet low in fruits",
    "Diet low in vegetables",
    "Unsafe sex",
    "Low physical activity",
    "High fasting plasma glucose",
    # "High total cholesterol",  # has many missing values – easiest is to drop for now
    "High body-mass index",
    "High systolic blood pressure",
    "Smoking",
    "Iron deficiency",
    "Vitamin A deficiency",
    "Low bone mineral density",
    "Air pollution",
    "Outdoor air pollution",
    "Diet high in sodium",
    "Diet low in whole grains",
    "Diet low in nuts and seeds",
    "Population",
]


def load_raw_data(path: str = DATA_PATH_DEFAULT) -> pd.DataFrame:
    """Load the raw merged CSV and do minimal cleaning."""
    df = pd.read_csv(path)

    # Clean Calories (string with commas) if you decide to use it later
    if "Calories" in df.columns:
        df["Calories"] = (
            df["Calories"]
            .astype(str)
            .str.replace(",", "", regex=False)
        )
        df["Calories"] = pd.to_numeric(df["Calories"], errors="coerce")

    # Drop rows with missing GDP (we can't train on them)
    df = df.dropna(subset=["GDP"]).reset_index(drop=True)

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived targets: GDP_per_capita and a high/low GDP label."""
    df = df.copy()
    df["GDP_per_capita"] = df["GDP"] / df["Population"]

    median_gdppc = df["GDP_per_capita"].median()
    df["high_gdp"] = (df["GDP_per_capita"] >= median_gdppc).astype(int)

    return df


def make_feature_matrix(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
) -> np.ndarray:
    """Extract X from dataframe."""
    return df[feature_cols].to_numpy(dtype=float)


def get_targets(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Return regression and classification targets."""
    y_reg = df["GDP"].to_numpy(dtype=float)
    y_clf = df["high_gdp"].to_numpy(dtype=int)
    return y_reg, y_clf


@dataclass
class RegressionResult:
    model_name: str
    mse: float
    r2: float


@dataclass
class ClassificationResult:
    model_name: str
    accuracy: float
    f1: float


def make_regression_pipeline(model: str = "linear"):
    """
    model: "linear" (closed-form linear regression) or "sgd" (gradient descent).
    """
    if model == "linear":
        est = LinearRegression()
    elif model == "sgd":
        # SGDRegressor uses (stochastic) gradient descent under the hood
        est = SGDRegressor(
            max_iter=2000,
            tol=1e-3,
            penalty="l2",
            learning_rate="invscaling",
        )
    else:
        raise ValueError(f"Unknown regression model: {model}")

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", est),
        ]
    )
    return pipe


def make_classification_pipeline(model: str = "logreg_gd"):
    """
    model:
      - "logreg_liblinear": LogisticRegression with liblinear solver
      - "logreg_gd": SGDClassifier as logistic regression via GD
    """
    if model == "logreg_liblinear":
        est = LogisticRegression(max_iter=1000, solver="liblinear")
    elif model == "logreg_gd":
        est = SGDClassifier(
            loss="log_loss",  # logistic regression
            max_iter=2000,
            tol=1e-3,
        )
    else:
        raise ValueError(f"Unknown classification model: {model}")

    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("model", est),
        ]
    )
    return pipe


def train_test_split_all(
    df: pd.DataFrame,
    feature_cols: List[str] = FEATURE_COLUMNS,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Produce train/test splits for both regression and classification."""
    X = make_feature_matrix(df, feature_cols)
    y_reg, y_clf = get_targets(df)

    X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
        X,
        y_reg,
        y_clf,
        test_size=test_size,
        random_state=random_state,
        stratify=y_clf,
    )

    return X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test


def run_regression_models(
    df: pd.DataFrame,
    models: List[str] = ("linear", "sgd"),
) -> Dict[str, RegressionResult]:
    """Fit regression models, return metrics on test set."""
    (
        X_train,
        X_test,
        y_train,
        y_test,
        _,
        _,
    ) = train_test_split_all(df)

    results: Dict[str, RegressionResult] = {}

    for name in models:
        pipe = make_regression_pipeline(name)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[name] = RegressionResult(model_name=name, mse=mse, r2=r2)

    return results


def run_classification_models(
    df: pd.DataFrame,
    models: List[str] = ("logreg_gd", "logreg_liblinear"),
) -> Dict[str, ClassificationResult]:
    """Fit classification models, return metrics on test set."""
    (
        X_train,
        X_test,
        _,
        _,
        y_train,
        y_test,
    ) = train_test_split_all(df)

    results: Dict[str, ClassificationResult] = {}

    for name in models:
        pipe = make_classification_pipeline(name)
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = ClassificationResult(model_name=name, accuracy=acc, f1=f1)

    return results
