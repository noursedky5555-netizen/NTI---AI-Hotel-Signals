"""AI assistant page."""

import streamlit as st


def render(data, cached_model_reports, build_ai_context, ask_gemini, render_ai_chat):
    st.title("Hotel Signals AI Assistant")
    st.markdown("Ask questions about the actual booking data, EDA findings, predictions, and model performance.")
    with st.spinner("Preparing the current dashboard context..."):
        classification, regression = cached_model_reports(data, comparison_version=5)
    ai_context = build_ai_context(data, classification, regression)

    summary_left, summary_right = st.columns([1, 1.6])
    with summary_left:
        if st.button("Generate AI Summary", key="generate_ai_summary"):
            with st.spinner("Writing the management summary..."):
                summary = ask_gemini(
                    "Generate a concise management summary with: 1. overall situation, 2. main cancellation patterns, 3. customer behavior, 4. pricing observations, 5. important ML findings, and 6. recommended actions. Use only the supplied dashboard context.",
                    ai_context,
                )
            st.session_state["ai_summary"] = summary
    with summary_right:
        st.caption("Suggested questions: Why are cancellations high? Which city is most risky? Explain my prediction. Which model performs best?")
    if st.session_state.get("ai_summary"):
        with st.container(border=True):
            st.subheader("AI management summary")
            st.markdown(st.session_state["ai_summary"])

    st.subheader("Ask the assistant")
    render_ai_chat(ai_context)
