"""Model performance page."""

import streamlit as st

from app.charts_matplotlib import classification_comparison_chart, cluster_view, model_metric_chart
from app.models import cluster_data


def render(data, cached_model_reports, build_ai_context, ask_gemini):
    st.title("Model Performance")
    st.markdown("Model quality checks and fit summary for the predictive sections of the app.")
    st.caption("Metrics use the full cleaned dataset and the configured preprocessing and model settings.")
    with st.spinner("Calculating model performance..."):
        classification, regression = cached_model_reports(data, comparison_version=5)
        clustered_data, _ = cluster_data(data, clusters=6)

    st.subheader("1. Cancellation Classification Models")
    classification_left, classification_right = st.columns(2)
    with classification_left:
        st.markdown("**Model Comparison**")
        st.dataframe(classification.style.format({"Accuracy": "{:.1%}", "Precision": "{:.1%}", "Recall": "{:.1%}", "F1 score": "{:.1%}"}), hide_index=True, width="stretch")
    with classification_right:
        st.pyplot(classification_comparison_chart(classification), width="stretch")

    best_classification_idx = classification["F1 score"].idxmax()
    best_clf_model = classification.iloc[best_classification_idx]
    st.info(f"**Best Classification Model:** {best_clf_model['Model']} with F1 Score of {best_clf_model['F1 score']:.4f} and Accuracy of {best_clf_model['Accuracy']:.4f}")
    st.divider()

    st.subheader("2. Rate Prediction Regression Models")
    regression_left, regression_right = st.columns(2)
    with regression_left:
        st.markdown("**Model Comparison**")
        st.dataframe(regression.style.format({"MAE": "${:.2f}", "RMSE": "${:.2f}", "R2": "{:.1%}"}), hide_index=True, width="stretch")
    with regression_right:
        st.pyplot(model_metric_chart(regression, "R2", "R2 by model"), width="stretch")

    best_regression_idx = regression["R2"].idxmax()
    best_reg_model = regression.iloc[best_regression_idx]
    st.info(f"**Best Regression Model:** {best_reg_model['Model']} with R2 Score of {best_reg_model['R2']:.4f} and MAE of ${best_reg_model['MAE']:.2f}")
    st.divider()

    st.subheader("3. Customer Segmentation Clustering")
    st.markdown("K-Means clustering groups bookings into 6 customer segments based on lead time, stay length, ADR, guest count, and special requests.")
    cluster_columns = st.columns([1, 1.2])
    with cluster_columns[0]:
        st.markdown("**Cluster Distribution**")
        cluster_dist = clustered_data["cluster"].value_counts().sort_index()
        st.bar_chart(cluster_dist, width="stretch")
    with cluster_columns[1]:
        st.markdown("**Cluster Statistics**")
        cluster_summary = clustered_data.groupby("cluster").agg({
            "lead_time": "mean",
            "adr": "mean",
            "total_stay_nights": "mean",
            "total_guests": "mean",
            "total_of_special_requests": "mean",
        }).round(2)
        st.dataframe(cluster_summary.style.format({"lead_time": "{:.0f}", "adr": "${:.2f}", "total_stay_nights": "{:.1f}", "total_guests": "{:.1f}", "total_of_special_requests": "{:.2f}"}), width="stretch")
    st.pyplot(cluster_view(clustered_data), width="stretch")
    st.divider()

    if st.button("Explain model performance with AI", key="explain_model_performance"):
        model_context = build_ai_context(data, classification, regression)
        with st.spinner("Preparing the model comparison..."):
            explanation = ask_gemini(
                "Explain the classification and regression model results using the actual metrics. Identify the strongest models, discuss precision versus recall, explain MAE, RMSE, and R2, and describe which model may be preferable for different hotel business goals. Do not claim that the highest accuracy is automatically the best choice.",
                model_context,
            )
        st.markdown(explanation)
