from __future__ import annotations

import streamlit as st

from ui import call_health, call_query, get_response, init_page, render_confidence_bar, render_evidence_cards, render_hero, render_metric_card, require_login


init_page("MyoCortex | Home", "MC")
require_login()

render_hero(
    "Research synthesis with clearer evidence and a working interface.",
    "Submit a biomedical question, inspect the pipeline verdict, and navigate through papers, claims, contradictions, and evaluation scores.",
)

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.markdown('<div class="section-title">Run a query</div>', unsafe_allow_html=True)
    default_query = st.session_state.get("last_query") or "Does creatine improve muscle strength in adults?"
    with st.form("query-form", clear_on_submit=False):
        query = st.text_area(
            "Biomedical query",
            value=default_query,
            height=120,
            placeholder="Ask a focused question about a supplement, disease, intervention, or outcome.",
        )
        submitted = st.form_submit_button("Analyze evidence")

    if submitted:
        result = call_query(query.strip())
        if result["ok"]:
            st.success("Structured response loaded from `/query/rich`.")
        else:
            st.error(f"Request failed: {result['error']}")

with right:
    st.markdown('<div class="section-title">System status</div>', unsafe_allow_html=True)
    health = call_health()
    status_cols = st.columns(2)
    with status_cols[0]:
        render_metric_card("API", "Online" if health["ok"] else "Offline", "FastAPI health endpoint")
    with status_cols[1]:
        render_metric_card("Mode", "Rich Query", "Dashboard consumes structured evidence")

response = get_response()
if response:
    st.markdown('<div class="section-title">Latest answer</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="panel">
            <div class="eyebrow">Verdict</div>
            <div style="font-size:1.8rem;font-weight:750;">{response.get("final_answer", "No answer generated.")}</div>
            <div class="muted" style="margin-top:0.85rem;line-height:1.7;">{response.get("summary", "")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    confidence = response.get("confidence", {})
    contradiction = response.get("contradiction", {})
    with metric_cols[0]:
        render_metric_card("Verdict", response.get("verdict", "Unknown"), "Decision engine outcome")
    with metric_cols[1]:
        render_metric_card("Confidence", confidence.get("label", "Unknown"), confidence.get("explanation", ""))
    with metric_cols[2]:
        render_metric_card("Evidence", response.get("evidence_strength", "Unknown"), "Pipeline evidence strength")
    with metric_cols[3]:
        render_metric_card("Backend", response.get("llm_backend_used", "unknown"), "LLM actually used")

    render_confidence_bar(int(confidence.get("score", 0)))

    support_cols = st.columns(3)
    with support_cols[0]:
        render_metric_card("Supporting", contradiction.get("support_count", 0), "Claims aligned with answer")
    with support_cols[1]:
        render_metric_card("Opposing", contradiction.get("oppose_count", 0), "Claims against answer")
    with support_cols[2]:
        render_metric_card("Neutral", contradiction.get("neutral_count", 0), contradiction.get("explanation", ""))

    st.markdown('<div class="section-title">Top evidence</div>', unsafe_allow_html=True)
    render_evidence_cards(response.get("evidence_cards", [])[:4])
    diagnostics = response.get("diagnostics", {})
    if diagnostics:
        st.markdown('<div class="section-title">Diagnostics</div>', unsafe_allow_html=True)
        diag_cols = st.columns(4)
        with diag_cols[0]:
            render_metric_card("Unique Papers", diagnostics.get("total_unique_papers", 0), "Before final ranking")
        with diag_cols[1]:
            render_metric_card("After Filter", diagnostics.get("papers_after_filter", 0), "Topic-filtered pool")
        with diag_cols[2]:
            render_metric_card("Ranked Final", diagnostics.get("final_ranked_papers", 0), "Evidence cards shown to the model")
        with diag_cols[3]:
            ratio = float(diagnostics.get("direct_evidence_ratio", 0.0)) * 100
            render_metric_card("Direct Ratio", f"{ratio:.0f}%", "Support or oppose among top papers")
        if diagnostics.get("search_queries"):
            st.caption("Search queries: " + " | ".join(diagnostics["search_queries"]))
        for note in diagnostics.get("notes", []):
            st.caption(note)
else:
    st.markdown(
        """
        <div class="empty-card" style="margin-top:1.2rem;">
            Run a query to populate the dashboard. The other pages will automatically use the latest response stored in this session.
        </div>
        """,
        unsafe_allow_html=True,
    )
