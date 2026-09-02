from __future__ import annotations

import plotly.express as px
import streamlit as st

from ui import get_response, init_page, load_eval_log, render_confidence_bar, render_hero, render_metric_card


init_page("MyoCortex | Evaluation", "ME")

render_hero(
    "Track retrieval and answer quality over time.",
    "Evaluation scores come from the latest live response and the persisted JSONL log.",
    eyebrow="Evaluation Metrics",
)

response = get_response()
if response:
    cols = st.columns(4)
    with cols[0]:
        render_metric_card("Faithfulness", f"{response.get('faithfulness_score', 0.0):.2f}", "How well the answer matches evidence")
    with cols[1]:
        render_metric_card("Coverage", f"{response.get('coverage_score', 0.0):.2f}", "How much relevant evidence was covered")
    with cols[2]:
        render_metric_card("Retrieval", f"{response.get('retrieval_score', 0.0):.2f}", "Search quality score")
    with cols[3]:
        render_metric_card("Confidence", response.get("confidence", {}).get("score", 0), response.get("confidence", {}).get("label", "Unknown"))
    render_confidence_bar(int(response.get("confidence", {}).get("score", 0)))
else:
    st.markdown('<div class="empty-card">Run a query on Home to populate the live evaluation metrics.</div>', unsafe_allow_html=True)

rows = load_eval_log()
st.markdown('<div class="section-title">Historical runs</div>', unsafe_allow_html=True)
if not rows:
    st.markdown('<div class="empty-card">No evaluation log entries were found.</div>', unsafe_allow_html=True)
else:
    fig = px.line(
        [
            {"run": idx, "score": row.get(score_name, 0.0), "metric": score_name.replace("_score", "").title()}
            for idx, row in enumerate(rows, start=1)
            for score_name in ("faithfulness_score", "coverage_score", "retrieval_score")
        ],
        x="run",
        y="score",
        color="metric",
        markers=True,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#f4f7fb",
        legend_title_text="Score",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)
    table_rows = [
        {
            "query": row.get("query", ""),
            "verdict": row.get("verdict", ""),
            "confidence_score": row.get("confidence_score", 0),
            "faithfulness_score": row.get("faithfulness_score", 0.0),
            "coverage_score": row.get("coverage_score", 0.0),
            "retrieval_score": row.get("retrieval_score", 0.0),
        }
        for row in rows
    ]
    st.dataframe(table_rows, use_container_width=True)
