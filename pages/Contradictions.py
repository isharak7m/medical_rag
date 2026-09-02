from __future__ import annotations

import plotly.express as px
import streamlit as st

from ui import get_response, init_page, render_hero, render_metric_card


init_page("MyoCortex | Contradictions", "MX")

render_hero(
    "Surface conflict instead of hiding it.",
    "The contradiction view breaks the latest result into support, oppose, and neutral evidence so disagreement is explicit.",
    eyebrow="Contradiction Analysis",
)

response = get_response()
if not response:
    st.markdown('<div class="empty-card">Run a query on Home to generate contradiction metrics.</div>', unsafe_allow_html=True)
else:
    contradiction = response.get("contradiction", {})
    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Conflict", "Yes" if contradiction.get("has_conflict") else "No", "Whether opposing evidence was found")
    with cols[1]:
        render_metric_card("Supporting", contradiction.get("support_count", 0), "Claims supporting the answer")
    with cols[2]:
        render_metric_card("Opposing", contradiction.get("oppose_count", 0), "Claims challenging the answer")
    with cols[3]:
        render_metric_card("Neutral", contradiction.get("neutral_count", 0), "Claims without directional impact")

    chart_data = [
        {"stance": "Support", "count": contradiction.get("support_count", 0)},
        {"stance": "Oppose", "count": contradiction.get("oppose_count", 0)},
        {"stance": "Neutral", "count": contradiction.get("neutral_count", 0)},
    ]
    fig = px.bar(
        chart_data,
        x="stance",
        y="count",
        color="stance",
        color_discrete_map={
            "Support": "#62f2bc",
            "Oppose": "#ff8d8d",
            "Neutral": "#ffd66e",
        },
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7fb",
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
    st.info(contradiction.get("explanation", "No contradiction explanation returned."))
    if contradiction.get("scope_note"):
        st.caption(contradiction["scope_note"])
