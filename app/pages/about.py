"""About page."""

import streamlit as st


def render():
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
        - **EDA Insights:** covers every unique exploratory analysis, with jump navigation and a prioritized Top 10.
        - **Model Performance:** compares classification and regression models using accuracy, F1 score, MAE, RMSE, and R².
        - **AI Assistant:** answers questions using aggregated dashboard context and can explain predictions and model results.
        - **Clustering:** groups bookings by descriptive characteristics such as lead time, stay length, guest count, and daily rate.
        """)
    with about_right:
        st.subheader("Project details")
        st.markdown("""
        **Project:** NTI - AI Project  
        **Supervisor:** Eng. Fady Attia  
        **Data source:** Hotel Booking Reservation dataset from Kaggle  
        **Primary objective:** Support earlier, evidence-based booking decisions  
        **Technology:** Python, Streamlit, pandas, Matplotlib, seaborn, scikit-learn, XGBoost, and Gemini
        """)

    st.subheader("How the analysis works")
    st.write(
        "The dataset is cleaned before analysis by normalizing required numeric fields, filling supported source fields, removing bookings with no guests, "
        "and dropping reservation outcome fields that could leak the target. The dashboard then combines descriptive statistics, Matplotlib visualizations, "
        "clustering, and supervised machine-learning models. Model inputs are prepared consistently, and report calculations are cached to keep the interface responsive."
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
