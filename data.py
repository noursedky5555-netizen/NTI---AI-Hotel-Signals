"""Data loading, cleaning, and feature preparation."""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from .config import REQUIRED_COLUMNS
except ImportError:
    from config import REQUIRED_COLUMNS


def demo_data(rows: int = 1200, seed: int = 42) -> pd.DataFrame:
    """Create a realistic demo dataset when no booking CSV is supplied."""
    rng = np.random.default_rng(seed)
    lead_time = np.clip(rng.gamma(2.2, 65, rows).astype(int), 0, 650)
    adults = rng.choice([1, 2, 3], rows, p=[0.12, 0.68, 0.20])
    children = rng.choice([0, 1, 2], rows, p=[0.72, 0.20, 0.08])
    babies = rng.choice([0, 1], rows, p=[0.96, 0.04])
    weekend = rng.poisson(1.1, rows)
    week = rng.poisson(2.8, rows)
    adr = np.clip(rng.normal(112, 38, rows) + weekend * 8, 35, 380).round(2)
    requests = np.clip(rng.poisson(0.8, rows), 0, 5)
    risk = -1.0 + lead_time / 180 - requests * 0.32 + (adr > 180) * 0.25
    cancellation = (rng.random(rows) < (1 / (1 + np.exp(-risk)))).astype(int)

    return pd.DataFrame({
        "hotel": rng.choice(["Harbor House", "Cedar Court"], rows),
        "city": rng.choice(["Lisbon", "Porto", "Faro", "Braga"], rows, p=[0.44, 0.30, 0.16, 0.10]),
        "customer_type": rng.choice(["Transient", "Contract", "Group", "Transient-party"], rows, p=[0.62, 0.12, 0.10, 0.16]),
        "market_segment": rng.choice(["Online TA", "Offline TA", "Direct", "Corporate", "Groups"], rows, p=[0.46, 0.18, 0.18, 0.10, 0.08]),
        "deposit_type": rng.choice(["No Deposit", "Refundable", "Non Refund"], rows, p=[0.72, 0.08, 0.20]),
        "lead_time": lead_time,
        "adults": adults,
        "children": children,
        "babies": babies,
        "stays_in_weekend_nights": weekend,
        "stays_in_week_nights": week,
        "adr": adr,
        "total_of_special_requests": requests,
        "is_repeated_guest": rng.choice([0, 1], rows, p=[0.86, 0.14]),
        "is_canceled": cancellation,
    })


def clean_data(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize a booking frame and add the features used by the dashboard."""
    data = frame.copy()
    for column in REQUIRED_COLUMNS:
        if column not in data.columns:
            raise ValueError(f"Missing required column: {column}")
    for column in ["adults", "children", "babies", "lead_time", "adr", "total_of_special_requests"]:
        data[column] = pd.to_numeric(data[column], errors="coerce").fillna(0)
    data["children"] = data["children"].clip(lower=0)
    data["total_guests"] = data["adults"] + data["children"] + data["babies"]
    data["total_stay_nights"] = data["stays_in_weekend_nights"] + data["stays_in_week_nights"]
    data = data[(data["total_guests"] > 0) & (data["total_stay_nights"] > 0)].copy()
    data["is_canceled"] = data["is_canceled"].astype(int)
    return data.reset_index(drop=True)


def prepare_features(data: pd.DataFrame, target: str = "is_canceled") -> tuple[pd.DataFrame, pd.Series]:
    frame = data.copy()
    for column in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(frame[column].median())
        else:
            frame[column] = frame[column].fillna("missing")

    features = frame.drop(columns=[target], errors="ignore").copy()
    features = features.select_dtypes(include=["number", "bool"])
    for column in features.columns:
        features[column] = pd.to_numeric(features[column], errors="coerce").fillna(features[column].median())

    target_series = frame[target].copy()
    if pd.api.types.is_numeric_dtype(target_series):
        target_series = pd.to_numeric(target_series, errors="coerce").fillna(target_series.median())
    return features, target_series
