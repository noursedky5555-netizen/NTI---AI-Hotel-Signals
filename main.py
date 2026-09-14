"""Hotel Signals: Streamlit entry point."""

from pathlib import Path

import pandas as pd
import streamlit as st

try:
    from hotel.charts import booking_map, cancellation_by_city, classification_comparison_chart, cluster_view, insight_visualizations, lead_time_distribution, model_metric_chart
    from hotel.config import COLORS, PAGE_ICON, PAGE_TITLE
    from hotel.data import clean_data, demo_data
    from hotel.models import classification_report, cluster_data, model_comparison, predict_cancellation_risk, regression_report
except ModuleNotFoundError:
    from charts import booking_map, cancellation_by_city, classification_comparison_chart, cluster_view, insight_visualizations, lead_time_distribution, model_metric_chart
    from config import COLORS, PAGE_ICON, PAGE_TITLE
    from data import clean_data, demo_data
    from models import classification_report, cluster_data, model_comparison, predict_cancellation_risk, regression_report

DEFAULT_CSV_PATH = Path(__file__).with_name("hotel_bookings_updated_2024.csv")
FAVICON_PATH = Path(__file__).with_name("hotel_favicon.svg")

st.set_page_config(page_title=PAGE_TITLE, page_icon=str(FAVICON_PATH), layout="wide", initial_sidebar_state="expanded")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Fraunces:opsz,wght@9..144,600&display=swap');
:root {{
    --ink:{COLORS['ink']}; --muted:{COLORS['muted']}; --paper:{COLORS['paper']};
    --panel:{COLORS['panel']}; --line:{COLORS['line']}; --coral:{COLORS['coral']};
    --teal:{COLORS['teal']}; --input:#FFFFFF; --control:#E9EEE8; --soft:{COLORS['panel']};
}}
html[data-theme='dark'], [data-theme='dark'] {{
    --ink:#F1F5F0; --muted:#B7C4BD; --paper:#111916; --panel:#1B2622;
    --line:#34443D; --input:#23312C; --control:#2B3A34; --soft:#17211D;
}}
html, body, [data-testid='stAppViewContainer'] {{ background:var(--paper); color:var(--ink); }}
.block-container {{ max-width: 1320px; padding-top: 3rem; }}
h1, h2, h3, body, p, label, [data-testid='stMetricLabel'], [data-testid='stMarkdownContainer'] {{ color:var(--ink); }}
h1, h2, h3 {{ font-family:'Fraunces', Georgia, serif !important; letter-spacing:0 !important; }}
body, p, label, [data-testid='stMetricValue'], [data-testid='stMetricLabel'] {{ font-family:'DM Sans', sans-serif; }}
[data-testid='stMetric'] {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:1rem; box-shadow:none; }}
[data-testid='stMetricValue'] {{ color:var(--teal); }}
[data-testid='stSidebar'] {{ background:var(--control); border-right:1px solid var(--line); }}
[data-testid='stSidebar'] h1 {{ font-size:1.7rem !important; color:var(--ink); }}
button[kind='primary'], .stButton > button, .stDownloadButton > button {{ border-radius:6px; border:1px solid var(--line); background:var(--panel); color:var(--ink); }}
button[kind='primary']:hover, .stButton > button:hover {{ border-color:var(--teal); color:var(--ink); }}
div[data-baseweb='select'] > div, input, textarea, div[data-testid='stCodeBlock'], .stTextInput > div > div, .stNumberInput > div > div, .stTextArea > div > div {{ background:var(--input); color:var(--ink); border-color:var(--line); }}
[data-testid='stFileUploaderDropzone'] {{ background:var(--soft); border:1px dashed var(--line); color:var(--ink); }}
[data-testid='stFileUploaderDropzone'] * {{ color:var(--ink) !important; }}
[data-testid='stCaptionContainer'], [data-testid='stCaptionContainer'] p {{ color:var(--muted); }}
[data-testid='stSidebar'] [data-testid='stMarkdownContainer'] p, [data-testid='stSidebar'] [data-testid='stMarkdownContainer'] strong {{ color:var(--ink); }}
.data-source-card {{
    margin-top: 0.75rem; padding: 0.8rem 0.9rem; border-radius: 12px;
    border: 1px solid var(--line); background: linear-gradient(180deg, var(--panel), var(--soft));
    color: var(--ink); box-shadow: inset 0 1px 0 rgba(255,255,255,0.14);
    font-size: 0.92rem; line-height: 1.5; font-weight: 500;
}}
.data-source-card strong {{ color: var(--teal); }}
.data-source-card .source-name {{ display:block; margin-top:0.25rem; color: var(--ink); font-weight:700; }}
hr {{ border-color:var(--line); }}
@media (max-width: 768px) {{
    .block-container {{ max-width: 100%; padding: 1.25rem 0.85rem 2rem; }}
    h1 {{ font-size: 2rem !important; line-height: 1.12 !important; }}
    h2 {{ font-size: 1.55rem !important; }}
    h3 {{ font-size: 1.25rem !important; }}
    p, li, [data-testid='stCaptionContainer'] {{ font-size: 0.95rem; line-height: 1.5; }}
    [data-testid='stMetric'] {{ padding: 0.75rem; min-height: 92px; }}
    [data-testid='stMetricValue'] {{ font-size: 1.35rem; }}
    [data-testid='stSidebar'] {{ min-width: 280px; }}
    [data-testid='stSidebar'] h1 {{ font-size: 1.35rem !important; white-space: nowrap; }}
    button, [role='button'], .stButton > button, .stDownloadButton > button {{ min-height: 44px; }}
    input, textarea, [data-baseweb='select'] {{ font-size: 16px !important; }}
    [data-testid='stPlotlyChart'] {{ width: 100% !important; overflow: hidden; }}
    [data-testid='stDataFrame'] {{ width: 100%; overflow-x: auto; }}
    .data-source-card {{ font-size: 0.86rem; padding: 0.7rem; }}
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
    report_frame = frame.sample(n=min(len(frame), 5000), random_state=42)
    return model_comparison(report_frame)


def top_insights(frame):
    city_rates = frame.groupby("city")["is_canceled"].mean().sort_values(ascending=False)
    lead_cutoff = frame["lead_time"].median()
    long_lead_rate = frame.loc[frame["lead_time"] >= lead_cutoff, "is_canceled"].mean()
    short_lead_rate = frame.loc[frame["lead_time"] < lead_cutoff, "is_canceled"].mean()
    request_rate = frame.loc[frame["total_of_special_requests"] > 0, "is_canceled"].mean()
    no_request_rate = frame.loc[frame["total_of_special_requests"] == 0, "is_canceled"].mean()
    canceled_adr = frame.loc[frame["is_canceled"] == 1, "adr"].mean()
    kept_adr = frame.loc[frame["is_canceled"] == 0, "adr"].mean()
    segment_rates = frame.groupby("market_segment")["is_canceled"].mean().sort_values(ascending=False)
    return [
        ("Highest cancellation exposure", f"{city_rates.index[0]} has the highest cancellation rate at {city_rates.iloc[0]:.1%}."),
        ("Longer lead times carry more risk", f"Bookings made at least {lead_cutoff:.0f} days ahead cancel at {long_lead_rate:.1%}, compared with {short_lead_rate:.1%} for shorter lead times."),
        ("Special requests signal commitment", f"Bookings with a special request cancel at {request_rate:.1%}, versus {no_request_rate:.1%} without requests."),
        ("Canceled stays have a different rate profile", f"Canceled bookings average ${canceled_adr:,.0f} per night, compared with ${kept_adr:,.0f} for completed bookings."),
        ("Market segment matters", f"{segment_rates.index[0]} has the highest segment cancellation rate at {segment_rates.iloc[0]:.1%}."),
    ]

with st.sidebar:
    st.markdown("# Hotel Signals")
    st.caption("A working view of booking behavior, cancellation exposure, and rate patterns.")
    page = st.radio("Navigation", ["Dashboard", "EDA Insights", "Model Performance", "About"], index=0)
    uploaded = st.file_uploader("Load a bookings CSV", type="csv")
    st.divider()
    st.markdown(
        "<div class='data-source-card'><strong>Data source</strong><br><a href='https://www.kaggle.com/datasets/kundanbedmutha/hotel-booking-reservation' target='_blank' style='color: var(--ink); text-decoration: underline; text-underline-offset: 2px;'>Hotel Booking Reservation</a><br><span style='opacity:0.82;'>Kaggle</span></div>",
        unsafe_allow_html=True,
    )
    st.caption("Required fields are checked when a file is loaded.")

try:
    data, source_name = load_frame(uploaded)
except Exception as error:
    st.error(str(error))
    st.stop()

if page == "Dashboard":
    st.title("The booking picture, before arrival day.")
    st.markdown("A compact operating view for spotting cancellation exposure and the booking patterns behind it.")

    with st.container():
        st.subheader("Predict cancellation risk")
        prediction_form, _ = st.columns([1.6, 0.4])
        with prediction_form:
            with st.form("booking_prediction"):
                fields = [
                    ("Lead time (days)", "lead_time", 15),
                    ("Adults", "adults", 2),
                    ("Children", "children", 0),
                    ("Babies", "babies", 0),
                    ("Weekend nights", "stays_in_weekend_nights", 1),
                    ("Week nights", "stays_in_week_nights", 2),
                    ("ADR", "adr", 120.0),
                    ("Special requests", "total_of_special_requests", 1),
                ]
                row1 = st.columns(2)
                form_values = {}
                for i, (label, key, default) in enumerate(fields):
                    with row1[i % 2]:
                        if isinstance(default, float):
                            form_values[key] = st.number_input(label, min_value=0.0, value=float(default), step=1.0)
                        else:
                            form_values[key] = st.number_input(label, min_value=0, value=int(default))
                rate_one = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
                customer_type = st.selectbox("Customer type", ["Transient", "Contract", "Group", "Transient-party"])
                market_segment = st.selectbox("Market segment", ["Online TA", "Offline TA", "Direct", "Corporate", "Groups"])
                hotel = st.selectbox("Hotel", ["Resort Hotel - Delhi", "Resort Hotel - Mumbai", "Resort Hotel - Bangalore", "City Hotel - Delhi"])
                city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chandigarh", "Jaipur", "Kolkata", "Lucknow", "Pune", "Hyderabad", "Chennai", "Indore", "Ahmedabad", "Bhopal", "Goa", "Kochi"])
                submitted = st.form_submit_button("Predict cancellation risk")
                if submitted:
                    user_input = {
                        **form_values,
                        "deposit_type": rate_one,
                        "customer_type": customer_type,
                        "market_segment": market_segment,
                        "hotel": hotel,
                        "city": city,
                    }
                    risk_value = predict_cancellation_risk(user_input)
                    st.metric("Predicted cancellation risk", f"{risk_value:.1%}")
                    if risk_value >= 0.5:
                        st.info("This booking looks higher risk. Consider flexible cancellation policies or proactive follow-up.")
                    else:
                        st.success("This booking looks relatively stable. Revenue risk appears moderate.")

    cancel_rate = data["is_canceled"].mean() * 100
    avg_rate = data["adr"].mean()
    avg_lead = data["lead_time"].mean()
    request_rate = (data["total_of_special_requests"] > 0).mean() * 100

    metric_cols = st.columns(4)
    metric_cols[0].metric("Bookings", f"{len(data):,}")
    metric_cols[1].metric("Cancellation rate", f"{cancel_rate:.1f}%")
    metric_cols[2].metric("Average daily rate", f"${avg_rate:,.0f}")
    metric_cols[3].metric("Bookings with requests", f"{request_rate:.1f}%")

    st.caption(f"Dataset reference: Hotel Booking Reservation  ·  {avg_lead:.0f} day average lead time")

    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.subheader("Where exposure concentrates")
        st.plotly_chart(cancellation_by_city(data), use_container_width=True, config={"displayModeBar": False}, key="dashboard_cancellation_by_city")
    with right:
        st.subheader("Lead time changes the shape")
        st.plotly_chart(lead_time_distribution(data), use_container_width=True, config={"displayModeBar": False}, key="dashboard_lead_time")

    st.subheader("Booking texture")
    st.plotly_chart(booking_map(data), use_container_width=True, config={"displayModeBar": False}, key="dashboard_booking_texture")

    with st.expander("Model notes", expanded=True):
        classification, regression = cached_model_reports(data, comparison_version=2)
        st.caption("The first run trains the models; later reruns reuse cached results until the dataset changes.")
        model_left, model_right = st.columns(2)
        with model_left:
            st.markdown("**Cancellation classification**  \nRandom forest with missing-value imputation and categorical encoding.")
            st.dataframe(classification.style.format({"Accuracy": "{:.1%}", "F1 score": "{:.1%}"}), hide_index=True, use_container_width=True)
        with model_right:
            st.markdown("**Rate estimation**  \nRandom forest regression for average daily rate with the same safeguards.")
            st.dataframe(regression.style.format({"MAE": "${:.2f}", "R2": "{:.1%}"}), hide_index=True, use_container_width=True)

    st.subheader("Behavior groups")
    cluster_count = st.slider("Number of groups", min_value=2, max_value=6, value=4, label_visibility="collapsed")
    clustered, inertia = cluster_data(data, cluster_count)
    st.plotly_chart(cluster_view(clustered), use_container_width=True, config={"displayModeBar": False}, key="dashboard_behavior_groups")
    st.caption(f"K-means inertia: {inertia:,.0f}. Groups are descriptive patterns, not customer labels.")

    st.download_button("Download cleaned data", data.to_csv(index=False), file_name="hotel_bookings_cleaned.csv", mime="text/csv")

elif page == "EDA Insights":
    st.title("EDA Insights")
    st.markdown("Key patterns discovered in the booking data.")
    st.subheader("Top Insights:")
    for (title, description), figure in zip(top_insights(data), insight_visualizations(data)):
        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.caption(description)
            st.plotly_chart(figure, use_container_width=True, config={"displayModeBar": False}, key=f"eda_insight_{title.lower().replace(' ', '_')}")
    st.write("These findings summarize what the charts show and what each pattern means for booking decisions.")

elif page == "Model Performance":
    st.title("Model Performance")
    st.markdown("Model quality checks and fit summary for the predictive sections of the app.")
    st.caption("Metrics use a representative sample of up to 5,000 cleaned bookings for a responsive page.")
    with st.spinner("Calculating model performance..."):
        classification, regression = cached_model_reports(data, comparison_version=2)
    classification_left, classification_right = st.columns(2)
    with classification_left:
        st.subheader("Cancellation model comparison")
        st.dataframe(classification.style.format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1 score": "{:.1%}"}), hide_index=True, use_container_width=True)
    with classification_right:
        st.plotly_chart(classification_comparison_chart(classification), use_container_width=True, config={"displayModeBar": False}, key="performance_classification_metrics")
    regression_left, regression_right = st.columns(2)
    with regression_left:
        st.subheader("Rate model comparison")
        st.dataframe(regression.style.format({"MAE": "${:.2f}", "R2": "{:.1%}"}), hide_index=True, use_container_width=True)
    with regression_right:
        st.plotly_chart(model_metric_chart(regression, "R2", "R2 by model"), use_container_width=True, config={"displayModeBar": False}, key="performance_regression_r2")

else:
    st.title("About")
    st.markdown("### An AI-assisted view of hotel booking behavior")
    st.write(
        "Hotel Signals is an NTI - AI Project designed to turn hotel reservation data into practical, understandable booking intelligence. "
        "The application helps users explore cancellation patterns, compare booking behavior across segments and cities, identify meaningful trends, "
        "and estimate cancellation risk for a new booking before arrival."
    )

    about_left, about_right = st.columns([1.15, 0.85], gap="large")
    with about_left:
        st.subheader("Project purpose")
        st.write(
            "Hotel cancellations affect room availability, revenue planning, staffing, and the ability to serve future guests. "
            "This project provides a focused workspace for understanding where cancellation exposure is concentrated and which booking details "
            "are associated with more stable or more uncertain reservations."
        )
        st.subheader("What the application does")
        st.markdown("""
        - **Dashboard:** summarizes the booking population and provides an interactive cancellation-risk prediction form.
        - **EDA Insights:** explains five important patterns with a short interpretation and a visualization for each one.
        - **Model Performance:** compares baseline, linear or logistic, and random-forest approaches using accuracy, F1 score, MAE, and R².
        - **Clustering:** groups bookings by descriptive characteristics such as lead time, stay length, guest count, and daily rate.
        """)
    with about_right:
        st.subheader("Project details")
        st.markdown("""
        **Project:** NTI - AI Project  
        **Supervisor:** Eng. Fady Attia  
        **Data source:** Hotel Booking Reservation dataset from Kaggle  
        **Primary objective:** Support earlier, evidence-based booking decisions  
        **Technology:** Python, Streamlit, pandas, Plotly, and scikit-learn
        """)

    st.subheader("How the analysis works")
    st.write(
        "The dataset is cleaned before analysis so missing numeric values, incomplete categorical values, invalid guest counts, and unusable stay records "
        "do not distort the results. The dashboard then combines descriptive statistics, interactive visualizations, clustering, and supervised machine-learning "
        "models. Models use imputation and categorical encoding, while trained artifacts and report calculations are cached to keep the interface responsive."
    )

    st.subheader("Team Members")
    team_columns = st.columns(3)
    team_members = [
        "Rawan Ibrahim Ahmed Zaki",
        "Nada Mostafa",
        "Salwa Hisham",
        "Nafe Emad",
        "Nour Eldin Yasser Mahmoud",
    ]
    for index, member in enumerate(team_members):
        with team_columns[index % 3]:
            st.container(border=True).markdown(f"**{member}**")

    st.caption("This project was developed under the supervision of Eng. Fady Attia as part of the NTI AI learning experience.")
