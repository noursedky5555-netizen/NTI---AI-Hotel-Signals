"""Small, reproducible modeling helpers used by the dashboard."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

try:
    from .data import prepare_features
except ImportError:
    from data import prepare_features


CATEGORICAL_COLUMNS = [
    "deposit_type",
    "customer_type",
    "market_segment",
    "hotel",
    "city",
]
NUMERIC_COLUMNS = [
    "lead_time",
    "adults",
    "children",
    "babies",
    "stays_in_weekend_nights",
    "stays_in_week_nights",
    "adr",
    "total_of_special_requests",
]


def _build_cancellation_model(feature_columns: list[str] | None = None, trees: int = 120) -> Pipeline:
    feature_columns = feature_columns or (NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    numeric_columns = [col for col in feature_columns if col in NUMERIC_COLUMNS]
    categorical_columns = [col for col in feature_columns if col in CATEGORICAL_COLUMNS]

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=trees, max_depth=10, random_state=42, n_jobs=1)),
    ])
    return model


def _build_rate_model(feature_columns: list[str] | None = None, trees: int = 120) -> Pipeline:
    feature_columns = feature_columns or (NUMERIC_COLUMNS + CATEGORICAL_COLUMNS)
    numeric_columns = [col for col in feature_columns if col in NUMERIC_COLUMNS]
    categorical_columns = [col for col in feature_columns if col in CATEGORICAL_COLUMNS]

    numeric_transformer = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_columns),
            ("cat", categorical_transformer, categorical_columns),
        ]
    )
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=trees, random_state=42, n_jobs=1)),
    ])
    return model


def _prepare_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if pd.api.types.is_numeric_dtype(cleaned[column]):
            cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").fillna(cleaned[column].median())
        else:
            cleaned[column] = cleaned[column].fillna("missing")
    return cleaned


def prepare_prediction_input(raw: dict[str, Any]) -> pd.DataFrame:
    defaults = {
        "lead_time": 0,
        "adults": 1,
        "children": 0,
        "babies": 0,
        "stays_in_weekend_nights": 0,
        "stays_in_week_nights": 0,
        "adr": 0.0,
        "total_of_special_requests": 0,
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
        "market_segment": "Online TA",
        "hotel": "Resort Hotel - Delhi",
        "city": "Delhi",
    }
    merged = {**defaults, **raw}
    row = {key: merged.get(key, defaults.get(key)) for key in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS}
    frame = pd.DataFrame([row])
    for col in NUMERIC_COLUMNS:
        frame[col] = pd.to_numeric(frame[col], errors="coerce").fillna(0)
    for col in CATEGORICAL_COLUMNS:
        frame[col] = frame[col].fillna(defaults[col]).astype(str)
    return frame


@lru_cache(maxsize=4)
def _trained_cancellation_model(csv_path: str, modified_ns: int) -> Pipeline:
    training_data = pd.read_csv(csv_path)
    training_data = training_data[training_data["is_canceled"].notna()].copy()
    for col in NUMERIC_COLUMNS:
        training_data[col] = pd.to_numeric(training_data[col], errors="coerce")
    for col in NUMERIC_COLUMNS:
        training_data[col] = training_data[col].fillna(training_data[col].median())
    for col in CATEGORICAL_COLUMNS:
        training_data[col] = training_data[col].fillna("missing").astype(str)
    training_data["total_guests"] = training_data["adults"] + training_data["children"] + training_data["babies"]
    training_data["total_stay_nights"] = training_data["stays_in_weekend_nights"] + training_data["stays_in_week_nights"]
    training_data = training_data[(training_data["total_guests"] > 0) & (training_data["total_stay_nights"] > 0)].copy()
    training_data["is_canceled"] = training_data["is_canceled"].astype(int)

    features = training_data[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS].copy()
    target = training_data["is_canceled"]
    return _build_cancellation_model().fit(features, target)


def predict_cancellation_risk(raw: dict[str, Any]) -> float:
    csv_path = Path(__file__).resolve().parents[1] / "data" / "hotel_bookings_updated_2024.csv"
    model = _trained_cancellation_model(str(csv_path), csv_path.stat().st_mtime_ns)
    single = prepare_prediction_input(raw)
    probability = model.predict_proba(single[NUMERIC_COLUMNS + CATEGORICAL_COLUMNS])[0][1]
    return float(probability)


def classification_report(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features, target = prepare_features(data)
    features = _prepare_numeric_frame(features)
    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.25, random_state=42, stratify=target)
    model = _build_cancellation_model(list(train_x.columns), trees=40)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    rows = [{"Model": "Random forest", "Accuracy": accuracy_score(test_y, prediction), "F1 score": f1_score(test_y, prediction)}]
    return pd.DataFrame(rows), pd.Series(prediction, index=test_y.index)


def regression_report(data: pd.DataFrame) -> pd.DataFrame:
    feature_columns = [col for col in NUMERIC_COLUMNS + CATEGORICAL_COLUMNS if col != "adr"]
    features = data[feature_columns].copy()
    for column in features.columns:
        if column in NUMERIC_COLUMNS:
            features[column] = pd.to_numeric(features[column], errors="coerce").fillna(features[column].median())
        else:
            features[column] = features[column].fillna("missing").astype(str)
    target = pd.to_numeric(data["adr"], errors="coerce").fillna(data["adr"].median())

    train_x, test_x, train_y, test_y = train_test_split(features, target, test_size=0.25, random_state=42)
    model = _build_rate_model(feature_columns, trees=40)
    model.fit(train_x, train_y)
    prediction = model.predict(test_x)
    return pd.DataFrame([{"Model": "Random forest regressor", "MAE": mean_absolute_error(test_y, prediction), "R2": r2_score(test_y, prediction)}])


def model_comparison(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare multiple classification and regression models using shared preprocessing.
    
    This function applies the project's configured preprocessing and model training
    to ensure consistent results.
    """
    # Classification: prepare the configured feature set.
    classification_frame = data.copy()
    
    # Handle missing values for specific columns.
    for column, value in [("agent", 0), ("company", 0), ("country", "Unknown")]:
        if column in classification_frame:
            classification_frame[column] = classification_frame[column].fillna(value)
    
    # Drop data leakage columns
    classification_frame = classification_frame.drop(
        columns=["reservation_status", "reservation_status_date"],
        errors="ignore",
    )
    
    # Convert categorical columns to numeric using get_dummies.
    classification_frame = pd.get_dummies(
        classification_frame,
        columns=classification_frame.select_dtypes(include="object").columns,
        drop_first=True,
    )
    
    # Prepare features and target
    classification_features = classification_frame.drop(columns=["is_canceled"])
    classification_target = classification_frame["is_canceled"]
    
    # Train/test split with stratification.
    train_x, test_x, train_y, test_y = train_test_split(
        classification_features,
        classification_target,
        test_size=0.2,
        random_state=42,
        stratify=classification_target,
    )
    
    # Scale features for models that need it
    scaler = StandardScaler()
    train_x_scaled = scaler.fit_transform(train_x)
    test_x_scaled = scaler.transform(test_x)
    
    # Classification models.
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=5),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
    }
    
    classification_rows = []
    for name, model in classifiers.items():
        # Use scaled data for Logistic Regression and KNN.
        use_scaled = name in {"Logistic Regression", "KNN"}
        fit_x = train_x_scaled if use_scaled else train_x
        pred_x = test_x_scaled if use_scaled else test_x
        
        model.fit(fit_x, train_y)
        prediction = model.predict(pred_x)
        
        classification_rows.append({
            "Model": name,
            "Accuracy": accuracy_score(test_y, prediction),
            "Precision": precision_score(test_y, prediction, zero_division=0),
            "Recall": recall_score(test_y, prediction, zero_division=0),
            "F1 score": f1_score(test_y, prediction),
        })
    
    # Regression: prepare the configured feature set.
    regression_frame = classification_frame.copy()
    regression_features = regression_frame.drop(columns=["adr"])
    regression_target = regression_frame["adr"]
    
    # Remove negative ADR values.
    valid = regression_target >= 0
    regression_features = regression_features[valid]
    regression_target = regression_target[valid]
    
    # Train/test split
    train_x, test_x, train_y, test_y = train_test_split(
        regression_features,
        regression_target,
        test_size=0.2,
        random_state=42,
    )
    
    # Scale features for linear regression
    regression_scaler = StandardScaler()
    train_x_scaled = regression_scaler.fit_transform(train_x)
    test_x_scaled = regression_scaler.transform(test_x)
    
    # Regression models.
    regressors = {
        "Linear Regression": (LinearRegression(), True),  # Uses scaling
        "Decision Tree": (DecisionTreeRegressor(random_state=42), False),
        "Random Forest": (RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1), False),
        "Gradient Boosting": (GradientBoostingRegressor(n_estimators=100, random_state=42), False),
    }
    
    regression_rows = []
    for name, (model, use_scaling) in regressors.items():
        fit_x = train_x_scaled if use_scaling else train_x
        pred_x = test_x_scaled if use_scaling else test_x
        
        model.fit(fit_x, train_y)
        prediction = model.predict(pred_x)
        
        regression_rows.append({
            "Model": name,
            "MAE": mean_absolute_error(test_y, prediction),
            "RMSE": mean_squared_error(test_y, prediction) ** 0.5,
            "R2": r2_score(test_y, prediction),
        })
    
    return pd.DataFrame(classification_rows), pd.DataFrame(regression_rows)


def cluster_data(data: pd.DataFrame, clusters: int = 4) -> tuple[pd.DataFrame, float]:
    """Cluster hotel booking data using K-Means with configured preprocessing."""
    # Select clustering features.
    columns = ["lead_time", "total_stay_nights", "adr", "total_guests", "total_of_special_requests"]
    source = data[columns].copy()
    
    # Convert to numeric and handle missing values using median
    for col in source.columns:
        source[col] = pd.to_numeric(source[col], errors="coerce")
        source[col] = source[col].fillna(source[col].median())
    
    # Scale features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(source)
    
    # Fit K-Means with configured settings.
    model = KMeans(n_clusters=clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(scaled)
    
    # Add cluster assignments to result
    result = data.copy()
    result["cluster"] = cluster_labels
    
    return result, float(model.inertia_)
