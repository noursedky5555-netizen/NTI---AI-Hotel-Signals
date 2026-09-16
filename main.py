"""Hotel Signals: single Streamlit entry point."""

from pathlib import Path
import re

import pandas as pd
import streamlit as st

from app.data import clean_data, demo_data
from app.gemini_service import ask_gemini
from app.models import model_comparison
from app.pages import about, ai_assistant, dashboard, eda_insights, model_performance
from app.config import COLORS, PAGE_TITLE

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = PROJECT_ROOT / "data" / "hotel_bookings_updated_2024.csv"
LOGO_PATH = PROJECT_ROOT / "assets" / "logo.png"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=str(LOGO_PATH),
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
:root {{
    --ink:{COLORS['ink']}; --muted:{COLORS['muted']}; --paper:{COLORS['paper']};
    --panel:{COLORS['panel']}; --line:{COLORS['line']}; --coral:{COLORS['coral']};
    --teal:{COLORS['teal']}; --accent:{COLORS['gold']};
    --input:#FFFFFF; --control:#E3ECE7; --soft:#F4F8F5;
}}
html[data-theme='dark'], [data-theme='dark'] {{
    --ink:{COLORS['ink']}; --muted:{COLORS['muted']}; --paper:{COLORS['paper']}; --panel:{COLORS['panel']};
    --line:{COLORS['line']}; --input:#FFFFFF; --control:#E3ECE7; --soft:#F4F8F5;
}}
html, body, [data-testid='stAppViewContainer'], [data-testid='stHeader'] {{ background:var(--paper); color:var(--ink); }}
.block-container {{ max-width:1320px; padding-top:3rem; }}
h1, h2, h3, body, p, label, [data-testid='stMetricLabel'], [data-testid='stMarkdownContainer'] {{ color:var(--ink); }}
h1, h2, h3 {{ font-family:'Fraunces', Georgia, serif !important; letter-spacing:0 !important; }}
body, p, label, [data-testid='stMetricValue'], [data-testid='stMetricLabel'] {{ font-family:'DM Sans', sans-serif; }}
[data-testid='stMetric'] {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1rem; }}
[data-testid='stMetricValue'] {{ color:var(--teal); }}
[data-testid='stSidebar'] {{ background:var(--control); border-right:1px solid var(--line); }}
button[kind='primary'], .stButton > button, .stDownloadButton > button {{ border-radius:6px; border:1px solid var(--line); background:var(--panel); color:var(--ink); }}
button[kind='primary']:hover, .stButton > button:hover, .stDownloadButton > button:hover {{ border-color:var(--teal); color:var(--ink); }}
div[data-baseweb='select'] > div, input, textarea, .stTextInput > div > div, .stNumberInput > div > div, .stTextArea > div > div {{ background:var(--input); color:var(--ink); border-color:var(--line); }}
[data-baseweb='popover'], [data-baseweb='menu'] {{ background:var(--panel); color:var(--ink); }}
[data-testid='stFileUploaderDropzone'] {{ background:var(--soft); border:1px dashed var(--line); color:var(--ink); }}
[data-testid='stFileUploaderDropzone'] * {{ color:var(--ink) !important; }}
[data-testid='stCaptionContainer'], [data-testid='stCaptionContainer'] p {{ color:var(--muted); }}
[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] p, [data-testid='stSidebar'] [data-testid='stMarkdownContainer'] strong {{ color:var(--ink); }}
.data-source-card {{ margin-top:.75rem; padding:.8rem .9rem; border-radius:8px; border:1px solid var(--line); background:var(--panel); color:var(--ink); font-size:.92rem; line-height:1.5; font-weight:500; }}
.data-source-card strong {{ color:var(--teal); }}
hr {{ border-color:var(--line); }}
@media (max-width:768px) {{
    .block-container {{ max-width:100%; padding:1.25rem .85rem 2rem; }}
    h1 {{ font-size:2rem !important; line-height:1.12 !important; }}
    h2 {{ font-size:1.55rem !important; }}
    h3 {{ font-size:1.25rem !important; }}
    p, li, [data-testid='stCaptionContainer'] {{ font-size:.95rem; line-height:1.5; }}
    [data-testid='stMetric'] {{ padding:.75rem; min-height:92px; }}
    [data-testid='stSidebar'] {{ min-width:280px; }}
    button, [role='button'], .stButton > button, .stDownloadButton > button {{ min-height:44px; }}
    input, textarea, [data-baseweb='select'] {{ font-size:16px !important; }}
}}
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner=False)
def load_frame(uploaded_file):
    if uploaded_file is not None:
        return clean_data(pd.read_csv(uploaded_file)), uploaded_file.name
    if DEFAULT_CSV_PATH.exists():
        return clean_data(pd.read_csv(DEFAULT_CSV_PATH)), DEFAULT_CSV_PATH.name
    return clean_data(demo_data()), "Demo bookings"


@st.cache_data(show_spinner=False)
def cached_model_reports(frame, comparison_version=2):
    return model_comparison(frame)


def top_insights(frame):
    overall = frame["is_canceled"].mean()
    candidates = [(1.0, "Overall cancellation baseline", f"{overall:.1%} of cleaned bookings are canceled, establishing the baseline for every comparison.", "Cancellation distribution")]
    chart_titles = {
        "hotel": "Hotel type vs cancellation",
        "market_segment": "Market segment vs cancellation",
        "customer_type": "Customer type vs cancellation",
        "deposit_type": "Deposit type vs cancellation",
        "is_repeated_guest": "Repeated guests",
        "distribution_channel": "Distribution channel",
        "city": "City distribution",
    }
    for column in ["city", "customer_type", "hotel", "market_segment", "deposit_type", "is_repeated_guest", "distribution_channel"]:
        if column not in frame:
            continue
        grouped = frame.groupby(column)["is_canceled"].agg(rate="mean", bookings="size")
        rates = grouped.loc[grouped["bookings"] >= 25, "rate"].dropna()
        if len(rates) < 2:
            continue
        high, low = rates.idxmax(), rates.idxmin()
        gap = float(rates.max() - rates.min())
        candidates.append((gap, f"{column.replace('_', ' ').title()} creates a {gap:.1%} gap", f"{high} has the highest cancellation rate at {rates.max():.1%}, while {low} is lowest at {rates.min():.1%}.", chart_titles[column]))

    request_rates = frame.groupby(frame["total_of_special_requests"].gt(0))["is_canceled"].mean()
    if len(request_rates) == 2:
        gap = float(request_rates.max() - request_rates.min())
        candidates.append((gap, "Special requests signal commitment", f"Bookings with a special request cancel at {request_rates[True]:.1%}, versus {request_rates[False]:.1%} without requests.", "Special requests impact"))

    for column, label in [("lead_time", "Lead time"), ("total_stay_nights", "Stay length"), ("adr", "Daily rate")]:
        cutoff = frame[column].median()
        lower = frame.loc[frame[column] < cutoff, "is_canceled"].mean()
        upper = frame.loc[frame[column] >= cutoff, "is_canceled"].mean()
        gap = abs(float(upper - lower))
        chart_title = {"lead_time": "Lead time vs cancellation", "total_stay_nights": "Stay length vs cancellation", "adr": "ADR vs cancellation"}[column]
        candidates.append((gap, f"{label} separates booking risk", f"Bookings at or above the median ({cutoff:.0f}) cancel at {upper:.1%}, compared with {lower:.1%} below the median.", chart_title))

    canceled_adr = frame.loc[frame["is_canceled"] == 1, "adr"].mean()
    kept_adr = frame.loc[frame["is_canceled"] == 0, "adr"].mean()
    candidates.append((abs(float(canceled_adr - kept_adr)) / max(float(frame["adr"].mean()), 1), "Canceled stays have a different rate profile", f"Canceled bookings average ${canceled_adr:,.0f} per night, compared with ${kept_adr:,.0f} for completed bookings.", "ADR vs cancellation"))
    return [(title, description, chart_title) for _, title, description, chart_title in sorted(candidates, reverse=True)[:10]]


def anchor_id(title: str) -> str:
    return "eda-" + re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")


def _cancellation_rates(frame, column: str) -> list[dict]:
    if column not in frame:
        return []
    summary = frame.groupby(column)["is_canceled"].agg(rate="mean", bookings="count").sort_values("rate", ascending=False)
    return [{"group": str(index), "cancellation_rate": round(float(row.rate), 4), "bookings": int(row.bookings)} for index, row in summary.head(20).iterrows()]


def build_ai_context(frame, classification=None, regression=None) -> dict:
    lead_bins = pd.cut(frame["lead_time"], bins=[-1, 7, 30, 90, 180, 365, float("inf")], labels=["0-7 days", "8-30 days", "31-90 days", "91-180 days", "181-365 days", "366+ days"])
    lead_rates = frame.groupby(lead_bins, observed=False)["is_canceled"].agg(rate="mean", bookings="count")
    context = {
        "general_statistics": {
            "total_bookings": int(len(frame)),
            "cancellation_rate": round(float(frame["is_canceled"].mean()), 4),
            "average_adr": round(float(frame["adr"].mean()), 2),
            "special_request_rate": round(float(frame["total_of_special_requests"].gt(0).mean()), 4),
            "average_lead_time_days": round(float(frame["lead_time"].mean()), 2),
            "average_stay_nights": round(float(frame["total_stay_nights"].mean()), 2),
            "average_guests": round(float(frame["total_guests"].mean()), 2),
        },
        "cancellation_analysis": {
            "by_hotel": _cancellation_rates(frame, "hotel"),
            "by_city": _cancellation_rates(frame, "city"),
            "by_customer_type": _cancellation_rates(frame, "customer_type"),
            "by_market_segment": _cancellation_rates(frame, "market_segment"),
            "by_deposit_type": _cancellation_rates(frame, "deposit_type"),
            "by_distribution_channel": _cancellation_rates(frame, "distribution_channel"),
            "by_lead_time_group": [{"group": str(index), "cancellation_rate": round(float(row.rate), 4), "bookings": int(row.bookings)} for index, row in lead_rates.iterrows() if int(row.bookings) > 0],
        },
        "eda_insights": [{"title": title, "description": description} for title, description, _ in top_insights(frame)],
    }
    if classification is not None:
        context["classification_models"] = classification.to_dict(orient="records")
    if regression is not None:
        context["regression_models"] = regression.to_dict(orient="records")
    return context


def render_ai_chat(context: dict) -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Ask me about cancellations, pricing, customers, predictions, or model performance."}]
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    question = st.chat_input("Ask Hotel Signals AI Assistant")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing the dashboard context..."):
                answer = ask_gemini(question, context)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


with st.sidebar:
    logo_col, title_col = st.columns([0.6, 2.4], gap="small", vertical_alignment="center")
    with logo_col:
        if LOGO_PATH.exists():
            st.image(str(LOGO_PATH), width=48)
    with title_col:
        st.markdown("<h1 style='font-size:24px; margin:0;'>Hotel Signals</h1>", unsafe_allow_html=True)
    st.caption("A working view of booking behavior, cancellation exposure, and rate patterns.")
    page = st.radio("Navigation", ["Dashboard", "EDA Insights", "Model Performance", "AI Assistant", "About"], index=0)
    uploaded = st.file_uploader("Load a bookings CSV", type="csv")
    st.divider()
    st.markdown("<div class='data-source-card'><strong>Data source</strong><br><a href='https://www.kaggle.com/datasets/kundanbedmutha/hotel-booking-reservation' target='_blank' style='color:var(--ink); text-decoration:underline;'>Hotel Booking Reservation</a><br><span style='opacity:.82;'>Kaggle</span></div>", unsafe_allow_html=True)
    st.caption("Required fields are checked when a file is loaded.")

try:
    data, source_name = load_frame(uploaded)
except Exception as error:
    st.error(str(error))
    st.stop()

if page == "Dashboard":
    dashboard.render(data, cached_model_reports, build_ai_context, ask_gemini)
elif page == "EDA Insights":
    eda_insights.render(data, top_insights, anchor_id)
elif page == "Model Performance":
    model_performance.render(data, cached_model_reports, build_ai_context, ask_gemini)
elif page == "AI Assistant":
    ai_assistant.render(data, cached_model_reports, build_ai_context, ask_gemini, render_ai_chat)
else:
    about.render()
