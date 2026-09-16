"""Dashboard page."""

import streamlit as st

from app.charts_matplotlib import booking_map, cancellation_by_city, cluster_view, lead_time_distribution
from app.models import cluster_data, predict_cancellation_risk


def render(data, cached_model_reports, build_ai_context, ask_gemini):
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
                for index, (label, key, default) in enumerate(fields):
                    with row1[index % 2]:
                        if isinstance(default, float):
                            form_values[key] = st.number_input(label, min_value=0.0, value=float(default), step=1.0)
                        else:
                            form_values[key] = st.number_input(label, min_value=0, value=int(default))
                deposit_type = st.selectbox("Deposit type", ["No Deposit", "Refundable", "Non Refund"])
                customer_type = st.selectbox("Customer type", ["Transient", "Contract", "Group", "Transient-party"])
                market_segment = st.selectbox("Market segment", ["Online TA", "Offline TA", "Direct", "Corporate", "Groups"])
                hotel = st.selectbox("Hotel", ["Resort Hotel - Delhi", "Resort Hotel - Mumbai", "Resort Hotel - Bangalore", "City Hotel - Delhi"])
                city = st.selectbox("City", ["Delhi", "Mumbai", "Bangalore", "Chandigarh", "Jaipur", "Kolkata", "Lucknow", "Pune", "Hyderabad", "Chennai", "Indore", "Ahmedabad", "Bhopal", "Goa", "Kochi"])
                submitted = st.form_submit_button("Predict cancellation risk")
                if submitted:
                    user_input = {
                        **form_values,
                        "deposit_type": deposit_type,
                        "customer_type": customer_type,
                        "market_segment": market_segment,
                        "hotel": hotel,
                        "city": city,
                    }
                    risk_value = predict_cancellation_risk(user_input)
                    prediction_features = {
                        **user_input,
                        "total_guests": form_values["adults"] + form_values["children"] + form_values["babies"],
                        "total_stay_nights": form_values["stays_in_weekend_nights"] + form_values["stays_in_week_nights"],
                    }
                    st.session_state["last_prediction"] = {"features": prediction_features, "risk": risk_value}
                    st.metric("Predicted cancellation risk", f"{risk_value:.1%}")
                    if risk_value >= 0.5:
                        st.info("This booking looks higher risk. Consider flexible cancellation policies or proactive follow-up.")
                    else:
                        st.success("This booking looks relatively stable. Revenue risk appears moderate.")

    last_prediction = st.session_state.get("last_prediction")
    if last_prediction:
        with st.expander("AI explanation for this prediction"):
            st.caption("The existing ML model produced the probability. Gemini only explains the supplied result and does not override it.")
            if st.button("Explain this prediction with AI", key="explain_prediction"):
                prediction_context = build_ai_context(data)
                prediction_context["current_prediction"] = {
                    "booking_features": last_prediction["features"],
                    "cancellation_probability": round(float(last_prediction["risk"]), 4),
                    "predicted_class": int(last_prediction["risk"] >= 0.5),
                }
                with st.spinner("Preparing the explanation..."):
                    explanation = ask_gemini(
                        "Explain why this booking received this cancellation risk based on the supplied booking information and model result. Give 3 practical actions a hotel manager could take. Clearly state that this is a model prediction and not a certainty.",
                        prediction_context,
                    )
                st.markdown(explanation)

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
        st.pyplot(cancellation_by_city(data), width="stretch")
    with right:
        st.subheader("Lead time changes the shape")
        st.pyplot(lead_time_distribution(data), width="stretch")

    st.subheader("Booking texture")
    st.pyplot(booking_map(data), width="stretch")

    with st.expander("Model notes", expanded=True):
        classification, regression = cached_model_reports(data, comparison_version=5)
        st.caption("The first run trains the models on the full cleaned dataset; later reruns reuse cached results until the dataset changes.")
        model_left, model_right = st.columns(2)
        with model_left:
            st.markdown("**Cancellation classification**  \nRandom forest with missing-value imputation and categorical encoding.")
            st.dataframe(classification.style.format({"Accuracy": "{:.1%}", "F1 score": "{:.1%}"}), hide_index=True, width="stretch")
        with model_right:
            st.markdown("**Rate estimation**  \nRandom forest regression for average daily rate with the same safeguards.")
            st.dataframe(regression.style.format({"MAE": "${:.2f}", "RMSE": "${:.2f}", "R2": "{:.1%}"}), hide_index=True, width="stretch")

    st.subheader("Behavior groups")
    cluster_count = st.slider("Number of groups", min_value=2, max_value=6, value=4, label_visibility="collapsed")
    clustered, inertia = cluster_data(data, cluster_count)
    st.pyplot(cluster_view(clustered), width="stretch")
    st.caption(f"K-means inertia: {inertia:,.0f}. Groups are descriptive patterns, not customer labels.")
    st.download_button("Download cleaned data", data.to_csv(index=False), file_name="hotel_bookings_cleaned.csv", mime="text/csv")
