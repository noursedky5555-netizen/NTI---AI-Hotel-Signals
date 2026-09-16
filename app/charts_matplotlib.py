"""Matplotlib/seaborn charts for the dashboard."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from pathlib import Path

try:
    from .config import COLORS
except ImportError:
    from config import COLORS

# Set matplotlib style for consistent appearance
plt.style.use('seaborn-v0_8-darkgrid')


def _get_figure_bytes(fig):
    """Convert matplotlib figure to bytes for Streamlit display."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf


def cancellation_by_city(data: pd.DataFrame):
    """Horizontal bar chart of cancellation rates by city."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    summary = data.groupby("city", as_index=False)["is_canceled"].mean().sort_values("is_canceled")
    summary["cancellation_rate"] = summary["is_canceled"] * 100
    
    bars = ax.barh(summary["city"], summary["cancellation_rate"], color=COLORS["coral"], alpha=0.85)
    ax.set_xlabel("Cancellation rate (%)", fontsize=10)
    ax.set_ylabel("")
    ax.set_title("Cancellation rate by city", fontsize=13, fontweight='bold', pad=14)
    
    # Add value labels on bars
    for i, (idx, row) in enumerate(summary.iterrows()):
        ax.text(row["cancellation_rate"] + 0.4, i, f'{row["cancellation_rate"]:.1f}%',
            va='center', fontsize=9)
    
    plt.tight_layout()
    return fig


def lead_time_distribution(data: pd.DataFrame):
    """Histogram of lead time with cancellation status overlay."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    lead_time_completed = data[data["is_canceled"] == 0]["lead_time"]
    lead_time_canceled = data[data["is_canceled"] == 1]["lead_time"]
    
    ax.hist(lead_time_completed, bins=24, alpha=0.62, label="Completed", color=COLORS["teal"])
    ax.hist(lead_time_canceled, bins=24, alpha=0.62, label="Canceled", color=COLORS["coral"])
    
    ax.set_xlabel("Lead time (days)", fontsize=10)
    ax.set_ylabel("Bookings", fontsize=10)
    ax.set_title("Lead time by booking status", fontsize=13, fontweight='bold', pad=14)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def booking_map(data: pd.DataFrame):
    """Scatter plot of lead time vs ADR with cancellation coloring."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    # Sample data for better visualization (if dataset is too large)
    plot_data = data.sample(n=min(len(data), 2500), random_state=42)
    
    completed = plot_data[plot_data["is_canceled"] == 0]
    canceled = plot_data[plot_data["is_canceled"] == 1]
    
    ax.scatter(completed["lead_time"], completed["adr"],
              s=completed["total_guests"].clip(upper=6)*12, alpha=0.28, color=COLORS["teal"],
              label="Completed", edgecolors='none')
    ax.scatter(canceled["lead_time"], canceled["adr"],
              s=canceled["total_guests"].clip(upper=6)*12, alpha=0.38, color=COLORS["coral"],
              label="Canceled", edgecolors='none')
    
    ax.set_xlabel("Lead time (days)", fontsize=10)
    ax.set_ylabel("Average daily rate ($)", fontsize=10)
    ax.set_title("Lead time and rate by booking status", fontsize=13, fontweight='bold', pad=14)
    ax.legend(fontsize=9, frameon=False)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def cluster_view(data: pd.DataFrame):
    """Scatter plot of clusters with lead time vs ADR."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    plot_data = data.sample(n=min(len(data), 3000), random_state=42)
    
    clusters = plot_data["cluster"].unique()
    colors = plt.cm.Set3(np.linspace(0, 1, len(clusters)))
    
    for i, cluster in enumerate(sorted(clusters)):
        cluster_data = plot_data[plot_data["cluster"] == cluster]
        ax.scatter(cluster_data["lead_time"], cluster_data["adr"],
                  s=cluster_data["total_stay_nights"].clip(upper=10)*8, alpha=0.48,
                  color=colors[i], label=f"Cluster {cluster}",
                  edgecolors='none')
    
    ax.set_xlabel("Lead time (days)", fontsize=10)
    ax.set_ylabel("Average daily rate ($)", fontsize=10)
    ax.set_title("Booking segments by lead time and rate", fontsize=13, fontweight='bold', pad=14)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False, title="Segment")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def classification_comparison_chart(classification: pd.DataFrame):
    """Bar chart comparing classification model F1 scores."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = classification["Model"].values
    f1_scores = classification["F1 score"].values * 100
    
    bars = ax.barh(models, f1_scores, color=COLORS["teal"], alpha=0.85)
    
    # Highlight best model
    best_idx = int(np.argmax(f1_scores))
    bars[best_idx].set_color(COLORS["coral"])
    
    ax.set_xlabel("F1 score (%)", fontsize=10)
    ax.set_ylabel("")
    ax.set_title("Classification model comparison", fontsize=13, fontweight='bold', pad=14)
    ax.set_xlim([0, 100])
    
    # Add value labels on bars
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score + 1, bar.get_y() + bar.get_height() / 2,
            f'{score:.1f}%', ha='left', va='center', fontsize=8)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def model_metric_chart(regression: pd.DataFrame, metric: str, title: str):
    """Bar chart for regression model metrics."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    models = regression["Model"].values
    
    if metric == "R2":
        values = regression["R2"].values * 100
        ylabel = "R² Score (%)"
    elif metric == "MAE":
        values = regression["MAE"].values
        ylabel = "Mean Absolute Error ($)"
    else:
        values = regression[metric].values
        ylabel = metric
    
    bars = ax.barh(models, values, color=COLORS["teal"], alpha=0.85)
    
    # Highlight best model (highest R2, lowest MAE/RMSE)
    if metric == "R2":
        best_idx = int(np.argmax(values))
    else:
        best_idx = int(np.argmin(values))
    bars[best_idx].set_color(COLORS["coral"])
    
    ax.set_xlabel(ylabel, fontsize=10)
    ax.set_ylabel("")
    ax.set_title(f"Regression models: {title}", fontsize=13, fontweight='bold', pad=14)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        if metric == "R2":
            label = f'{value:.1f}%'
        elif metric in ["MAE", "RMSE"]:
            label = f'${value:.2f}'
        else:
            label = f'{value:.3f}'
        ax.text(value + max(abs(values)) * 0.01, bar.get_y() + bar.get_height() / 2,
            label, ha='left', va='center', fontsize=8)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def _category_vs_cancellation(data: pd.DataFrame, column: str, title: str):
    """Plot cancellation rates by categorical feature."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    if column not in data:
        ax.text(0.5, 0.5, "Feature not available", ha='center', va='center')
        plt.tight_layout()
        return fig
    
    summary = data.groupby(column, as_index=False)["is_canceled"].agg(
        cancellation_rate="mean", bookings="size"
    )
    omitted = summary.loc[summary["bookings"] < 25, "bookings"].sum()
    summary = summary.loc[summary["bookings"] >= 25].sort_values("cancellation_rate")
    summary["cancellation_rate"] *= 100
    
    bars = ax.barh(summary[column].astype(str), summary["cancellation_rate"],
                   color=COLORS["coral"], alpha=0.85)

    ax.set_xlabel("Cancellation rate (%)", fontsize=10)
    ax.set_ylabel("")
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    if omitted:
        ax.text(
            0,
            1.01,
            f"Low-volume categories omitted: {omitted:,} bookings (<25 each)",
            transform=ax.transAxes,
            fontsize=8,
            color=COLORS["muted"],
            va="bottom",
        )
    
    # Add value labels
    for bar, rate in zip(bars, summary["cancellation_rate"]):
        ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2,
                f'{rate:.1f}%', ha='left', va='center', fontsize=8)
    
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def _target_box(data: pd.DataFrame, column: str, title: str, label: str):
    """Box plot of numeric feature by cancellation status."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    plot_data = data[[column, "is_canceled"]].copy()
    plot_data["Status"] = plot_data["is_canceled"].map({0: "Completed", 1: "Canceled"})
    
    statuses = ["Completed", "Canceled"]
    values = [plot_data.loc[plot_data["Status"] == status, column].dropna() for status in statuses]
    boxplot = ax.boxplot(
        values,
        tick_labels=statuses,
        patch_artist=True,
        medianprops=dict(color="black", linewidth=2),
    )
    for patch, color in zip(boxplot["boxes"], [COLORS["teal"], COLORS["coral"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_xlabel("Booking Status", fontsize=11, fontweight='bold')
    ax.set_ylabel(label, fontsize=11, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold', pad=20)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig


def _distribution(data: pd.DataFrame, column: str, title: str, label: str, categorical: bool = False):
    """Histogram or count plot of feature distribution."""
    fig, ax = plt.subplots(figsize=(9, 5.5))
    
    if categorical:
        value_counts = data[column].value_counts().head(20)
        if column == "is_canceled":
            value_counts.index = value_counts.index.map({0: "Completed", 1: "Canceled"})
        ax.barh(value_counts.index.astype(str), value_counts.values,
                color=COLORS["teal"], alpha=0.85)
        ax.set_xlabel("Bookings", fontsize=10)
        ax.set_ylabel("")
        for index, value in enumerate(value_counts.values):
            ax.text(value + max(value_counts.values) * 0.01, index, f"{value:,}", va="center", fontsize=8)
    else:
        ax.hist(data[column].dropna(), bins=32, color=COLORS["teal"], alpha=0.82)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Bookings", fontsize=10)
    
    ax.set_title(title, fontsize=13, fontweight='bold', pad=14)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    return fig


def eda_visualizations(data: pd.DataFrame):
    """Generate all EDA visualizations for the dashboard.
    
    Returns list of tuples: (title, description, figure)
    """
    edas = [
        ("Cancellation distribution", "Overall breakdown of canceled vs completed bookings.", 
         _distribution(data, "is_canceled", "Cancellation Distribution", "", categorical=True)),
        
        ("Hotel type vs cancellation", "Cancellation rates differ significantly by hotel type.",
         _category_vs_cancellation(data, "hotel", "Cancellation by Hotel Type")),
        
        ("Market segment vs cancellation", "Customer acquisition channel affects cancellation risk.",
         _category_vs_cancellation(data, "market_segment", "Cancellation by Market Segment")),
        
        ("Lead time vs cancellation", "Longer lead times correlate with higher cancellation rates.",
         _target_box(data, "lead_time", "Lead Time Distribution by Cancellation Status", "Lead Time (days)")),
        
        ("Stay length vs cancellation", "Shorter stays have higher cancellation rates.",
         _target_box(data, "total_stay_nights", "Stay Length by Cancellation Status", "Total Stay Nights")),
        
        ("ADR vs cancellation", "Daily rate distribution differs between canceled and completed.",
         _target_box(data, "adr", "Average Daily Rate by Cancellation Status", "ADR ($)")),
        
        ("Guest count distribution", "Most bookings are small groups.",
         _distribution(data, "total_guests", "Guest Count Distribution", "Total Guests")),
        
        ("Special requests impact", "Bookings with special requests cancel less.",
         _category_vs_cancellation(data, "total_of_special_requests", "Cancellation by Special Requests")),
        
        ("Deposit type vs cancellation", "Non-refundable deposits reduce cancellations.",
         _category_vs_cancellation(data, "deposit_type", "Cancellation by Deposit Type")),
        
        ("Customer type vs cancellation", "Customer type is a strong cancellation predictor.",
         _category_vs_cancellation(data, "customer_type", "Cancellation by Customer Type")),
        
        ("Repeated guests", "Repeat customers have different cancellation patterns.",
         _category_vs_cancellation(data, "is_repeated_guest", "Cancellation by Repeat Guest Status")),
        
        ("City distribution", "Cancellations vary significantly by city.",
         _category_vs_cancellation(data, "city", "Cancellation by City")),
        
        ("Distribution channel", "Booking channel affects cancellation likelihood.",
         _category_vs_cancellation(data, "distribution_channel", "Cancellation by Distribution Channel")),
        
        ("Lead time distribution", "Lead time histogram with cancellation overlay.",
         lead_time_distribution(data)),
        
        ("Booking texture", "Scatter plot of lead time vs ADR colored by cancellation.",
         booking_map(data)),
    ]
    
    return edas
