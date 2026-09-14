import pandas as pd

from data import clean_data
from models import prepare_prediction_input, predict_cancellation_risk


def test_prepare_features_removes_nan():
    df = pd.DataFrame(
        {
            "is_canceled": [0, 1, 0],
            "lead_time": [10, None, 5],
            "adr": [100.0, 80.0, None],
            "adults": [1, 2, 1],
            "children": [0, 1, None],
            "babies": [0, 0, 0],
            "stays_in_weekend_nights": [1, 0, 1],
            "stays_in_week_nights": [2, 3, 2],
            "total_of_special_requests": [1, None, 0],
        }
    )
    features, target = clean_data(df).pipe(lambda x: x)
    assert features.isna().sum().sum() == 0
    assert len(target) == len(features)


def test_prediction_accepts_user_input():
    sample = {
        "lead_time": 15,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adr": 120,
        "total_of_special_requests": 1,
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
        "market_segment": "Online TA",
        "hotel": "Resort Hotel - Delhi",
        "city": "Delhi",
    }
    risk = predict_cancellation_risk(sample)
    assert 0 <= risk <= 1
    assert isinstance(risk, float)


def test_prepare_prediction_input_has_expected_columns():
    sample = {
        "hotel": "Resort Hotel - Delhi",
        "city": "Delhi",
        "lead_time": 10,
        "adults": 2,
        "children": 0,
        "babies": 0,
        "stays_in_weekend_nights": 1,
        "stays_in_week_nights": 2,
        "adr": 100,
        "total_of_special_requests": 1,
        "deposit_type": "No Deposit",
        "customer_type": "Transient",
        "market_segment": "Online TA",
    }
    frame = prepare_prediction_input(sample)
    assert list(frame.columns) == [
        "lead_time",
        "adults",
        "children",
        "babies",
        "stays_in_weekend_nights",
        "stays_in_week_nights",
        "adr",
        "total_of_special_requests",
        "deposit_type_No Deposit",
        "deposit_type_Refundable",
        "deposit_type_Non Refund",
        "customer_type_Contract",
        "customer_type_Group",
        "customer_type_Transient",
        "customer_type_Transient-party",
        "market_segment_Corporate",
        "market_segment_Direct",
        "market_segment_Groups",
        "market_segment_Offline TA",
        "market_segment_Online TA",
        "hotel_Resort Hotel - Delhi",
        "city_Ahmedabad",
        "city_Bangalore",
        "city_Chandigarh",
        "city_Chennai",
        "city_Delhi",
        "city_Goa",
        "city_Hyderabad",
        "city_Indore",
        "city_Jaipur",
        "city_Kochi",
        "city_Kolkata",
        "city_Lucknow",
        "city_Mumbai",
        "city_Pune",
        "city_Bhopal",
    ]
