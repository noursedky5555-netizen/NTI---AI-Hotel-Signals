"""Plotly figures for the dashboard."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio

try:
    from .config import COLORS
except ImportError:
    from config import COLORS


pio.templates["hotel_signals"] = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor=COLORS["paper"],
        plot_bgcolor=COLORS["panel"],
        font={"color": COLORS["ink"], "family": "DM Sans, sans-serif"},
        title={"font": {"color": COLORS["ink"]}},
        xaxis={"gridcolor": COLORS["line"], "zerolinecolor": COLORS["line"]},
        yaxis={"gridcolor": COLORS["line"], "zerolinecolor": COLORS["line"]},
        legend={"bgcolor": COLORS["panel"], "bordercolor": COLORS["line"], "borderwidth": 1},
    )
)
px.defaults.template = "hotel_signals"


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


def _category(data: pd.DataFrame, column: str) -> pd.Series:
    if column in data:
        return data[column].fillna("Unknown").astype(str)
    return pd.Series(["Unavailable"] * len(data), index=data.index)


def _category_vs_cancellation(data: pd.DataFrame, column: str, title: str):
    frame = pd.DataFrame({column: _category(data, column), "is_canceled": data["is_canceled"]})
    summary = frame.groupby(column, as_index=False)["is_canceled"].mean()
    summary["cancellation_rate"] = summary["is_canceled"] * 100
    return px.bar(summary.sort_values("cancellation_rate"), x="cancellation_rate", y=column, orientation="h", text_auto=".1f", title=title, labels={column: "", "cancellation_rate": "Cancellation rate (%)"}, color_discrete_sequence=[COLORS["coral"]])


def _target_box(data: pd.DataFrame, column: str, title: str, label: str):
    frame = data[["is_canceled", column]].copy()
    frame["booking_status"] = frame["is_canceled"].map({0: "Completed", 1: "Canceled"})
    return px.box(frame, x="booking_status", y=column, color="booking_status", points=False, title=title, labels={"booking_status": "", column: label}, color_discrete_map={"Completed": COLORS["teal"], "Canceled": COLORS["coral"]})


def _distribution(data: pd.DataFrame, column: str, title: str, label: str, categorical: bool = False):
    if categorical:
        return px.histogram(data, x=_category(data, column), color_discrete_sequence=[COLORS["teal"]], title=title, labels={"count": "Bookings", "x": label})
    return px.histogram(data, x=column, nbins=40, title=title, labels={column: label, "count": "Bookings"}, color_discrete_sequence=[COLORS["teal"]])


def eda_visualizations(data: pd.DataFrame):
    """Return every EDA visualization in display order."""
    status = data["is_canceled"].map({0: "Completed", 1: "Canceled"})
    sample = data.sample(n=min(len(data), 5000), random_state=42).copy()
    sample["booking_status"] = sample["is_canceled"].map({0: "Completed", 1: "Canceled"})
    request_data = data.copy()
    request_data["request_group"] = request_data["total_of_special_requests"].gt(0).map({True: "With special request", False: "No special request"})
    request_summary = request_data.groupby("request_group", as_index=False)["is_canceled"].mean()
    request_summary["cancellation_rate"] = request_summary["is_canceled"] * 100
    figures = [
        ("Target balance", "How the booking population splits between completed and canceled reservations.", px.histogram(x=status, title="Canceled vs non-canceled bookings", labels={"x": "Booking status", "y": "Bookings"}, color_discrete_sequence=[COLORS["teal"]])),
    ]
    for column, title in [
        ("city", "Cancellation vs city"),
        ("customer_type", "Cancellation vs customer type"),
        ("hotel", "Cancellation vs hotel type"),
        ("market_segment", "Cancellation vs market segment"),
        ("deposit_type", "Cancellation vs deposit type"),
        ("is_repeated_guest", "Cancellation vs repeated guest"),
        ("distribution_channel", "Cancellation vs distribution channel"),
    ]:
        figures.append((title, f"Cancellation exposure across {column.replace('_', ' ')} groups.", _category_vs_cancellation(data, column, title)))
    figures.append(("Cancellation vs special requests", "Whether a booking contains at least one special request and how that relates to cancellation.", px.bar(request_summary, x="request_group", y="cancellation_rate", color="request_group", text_auto=".1f", title="Cancellation vs special requests", labels={"request_group": "", "cancellation_rate": "Cancellation rate (%)"}, color_discrete_sequence=[COLORS["teal"], COLORS["coral"]])))
    for column, title, label in [
        ("total_stay_nights", "Total nights vs cancellation", "Total stay nights"),
        ("adr", "ADR vs cancellation", "Average daily rate"),
        ("lead_time", "Lead time vs cancellation", "Lead time (days)"),
    ]:
        figures.append((title, f"The distribution of {label.lower()} for completed and canceled bookings.", _target_box(data, column, title, label)))
    figures.extend([
        ("Lead time distribution", "Most bookings are concentrated in shorter lead-time ranges, with cancellations overlaid.", px.histogram(data, x="lead_time", color=status, nbins=40, barmode="overlay", opacity=0.78, title="Distribution of lead time", labels={"lead_time": "Lead time (days)", "is_canceled": "Booking status"}, color_discrete_map={"Completed": COLORS["teal"], "Canceled": COLORS["coral"]})),
        ("ADR distribution", "The spread of average daily rates across the booking population.", _distribution(data, "adr", "Distribution of average daily rate", "Average daily rate")),
        ("Total nights distribution", "The most common stay lengths in the cleaned booking population.", _distribution(data, "total_stay_nights", "Distribution of total nights", "Total stay nights")),
        ("Total guests distribution", "The number of guests represented by each booking.", _distribution(data, "total_guests", "Distribution of total guests", "Total guests", categorical=True)),
        ("Lead time vs ADR", "Whether advance purchase timing is associated with different daily rates.", px.scatter(sample, x="lead_time", y="adr", color="booking_status", opacity=0.5, title="Lead time vs average daily rate", labels={"lead_time": "Lead time (days)", "adr": "Average daily rate", "booking_status": "Booking status"}, color_discrete_map={"Completed": COLORS["teal"], "Canceled": COLORS["coral"]})),
        ("ADR vs total nights", "Whether longer stays are associated with different average daily rates.", px.scatter(sample, x="total_stay_nights", y="adr", color="booking_status", opacity=0.5, title="Average daily rate vs total nights", labels={"total_stay_nights": "Total stay nights", "adr": "Average daily rate", "booking_status": "Booking status"}, color_discrete_map={"Completed": COLORS["teal"], "Canceled": COLORS["coral"]})),
    ])
    return figures


def insight_visualizations(data: pd.DataFrame):
    """Backward-compatible five-chart subset used by older callers."""
    return [figure for _, _, figure in eda_visualizations(data)[:5]]


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
