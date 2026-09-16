"""Exploratory data analysis page."""

import streamlit as st

from app.charts_matplotlib import eda_visualizations


def render(data, top_insights, anchor_id):
    st.title("EDA Insights")
    st.markdown("Every exploratory view, with the strongest ten patterns prioritized first.")
    edas = eda_visualizations(data)
    figures_by_title = {title: figure for title, _, figure in edas}

    st.subheader("Top 10 insights")
    for rank, (title, description, chart_title) in enumerate(top_insights(data), start=1):
        with st.container(border=True):
            st.markdown(f"### {rank}. {title}")
            st.caption(description)
            st.pyplot(figures_by_title[chart_title], width="stretch")

    st.subheader("Jump to an EDA")
    st.markdown(" | ".join(f"[{index}. {title}](#{anchor_id(title)})" for index, (title, _, _) in enumerate(edas, start=1)))
    st.divider()
    st.subheader("All EDA visualizations")
    for index, (title, description, figure) in enumerate(edas, start=1):
        st.markdown(f"<div id='{anchor_id(title)}'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"### {index}. {title}")
            st.caption(description)
            st.pyplot(figure, width="stretch")
