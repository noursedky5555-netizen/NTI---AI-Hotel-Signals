"""Plotly figures for the dashboard."""

import pandas as pd
import plotly.express as px

try:
    from .config import COLORS
except ImportError:
    from config import COLORS


def cancellation_by_city(data: pd.DataFrame):
    summary = data.groupby("city", as_index=False)["is_canceled"].mean().sort_values("is_canceled")
    summary["cancellation_rate"] = summary["is_canceled"] * 100
    return px.bar(summary, x="cancellation_rate", y="city", orientation="h", text_auto=".1f", labels={"cancellation_rate": "Cancellation rate (%)", "city": ""}, color_discrete_sequence=[COLORS["coral"]])


def lead_time_distribution(data: pd.DataFrame):
    return px.histogram(data, x="lead_time", color="is_canceled", nbins=28, barmode="overlay", opacity=0.78, labels={"lead_time": "Lead time (days)", "is_canceled": "Canceled"}, color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]})


def booking_map(data: pd.DataFrame):
    return px.scatter(data, x="lead_time", y="adr", size="total_guests", color="is_canceled", hover_data=["city", "customer_type"], labels={"lead_time": "Lead time (days)", "adr": "Average daily rate", "is_canceled": "Canceled"}, color_discrete_map={0: COLORS["teal"], 1: COLORS["coral"]})


def cluster_view(data: pd.DataFrame):
    return px.scatter(data, x="lead_time", y="adr", color="cluster", size="total_stay_nights", hover_data=["city", "total_guests"], labels={"lead_time": "Lead time (days)", "adr": "Average daily rate"}, color_continuous_scale=[COLORS["teal"], COLORS["gold"], COLORS["coral"]])


def insight_visualizations(data: pd.DataFrame):
    request_data = data.copy()
    request_data["request_group"] = request_data["total_of_special_requests"].gt(0).map({True: "With special request", False: "No special request"})
    request_summary = request_data.groupby("request_group", as_index=False)["is_canceled"].mean()
    request_summary["cancellation_rate"] = request_summary["is_canceled"] * 100

    rate_data = data.copy()
    rate_data["booking_status"] = rate_data["is_canceled"].map({0: "Completed", 1: "Canceled"})
    rate_data = rate_data.sample(n=min(len(rate_data), 5000), random_state=42)

    segment_summary = data.groupby("market_segment", as_index=False)["is_canceled"].mean()
    segment_summary["cancellation_rate"] = segment_summary["is_canceled"] * 100
    return [
        cancellation_by_city(data),
        lead_time_distribution(data),
        px.bar(request_summary, x="request_group", y="cancellation_rate", color="request_group", text_auto=".1f", labels={"request_group": "", "cancellation_rate": "Cancellation rate (%)"}, color_discrete_sequence=[COLORS["teal"], COLORS["coral"]]),
        px.box(rate_data, x="booking_status", y="adr", color="booking_status", points=False, labels={"booking_status": "", "adr": "Average daily rate"}, color_discrete_map={"Completed": COLORS["teal"], "Canceled": COLORS["coral"]}),
        px.bar(segment_summary.sort_values("cancellation_rate"), x="cancellation_rate", y="market_segment", orientation="h", text_auto=".1f", labels={"market_segment": "", "cancellation_rate": "Cancellation rate (%)"}, color_discrete_sequence=[COLORS["gold"]]),
    ]


def model_metric_chart(data: pd.DataFrame, metric: str, title: str):
    return px.bar(data, x="Model", y=metric, color="Model", text_auto=".2f", title=title, labels={metric: metric}, color_discrete_sequence=[COLORS["teal"], COLORS["coral"], COLORS["gold"]])


def classification_comparison_chart(data: pd.DataFrame):
    chart_data = data.melt(
        id_vars="Model",
        value_vars=["Accuracy", "Precision", "Recall", "F1 score"],
        var_name="Metric",
        value_name="Score",
    )
    return px.bar(
        chart_data,
        x="Model",
        y="Score",
        color="Metric",
        barmode="group",
        text_auto=".2f",
        title="Classification metrics by model",
        labels={"Score": "Score", "Model": ""},
        color_discrete_sequence=[COLORS["teal"], COLORS["coral"], COLORS["gold"], COLORS["blue"]],
    )
