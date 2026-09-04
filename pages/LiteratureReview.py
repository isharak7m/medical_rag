from __future__ import annotations

import streamlit as st
import requests

from ui import init_page, render_hero, render_metric_card, API_BASE, get_auth_headers, require_login

init_page("MyoCortex | Literature Review", "LR")
require_login()

render_hero(
    "Literature Review Generator",
    "Auto-generate synthesized reviews across multiple papers. Covers consensus, disagreements, research gaps, and clinical implications.",
)

with st.form("review-form"):
    query = st.text_input(
        "Research question",
        value="creatine supplementation muscle strength",
        placeholder="Enter a focused research question",
    )
    submitted = st.form_submit_button("Generate Literature Review")

if submitted and query:
    with st.spinner("Generating literature review..."):
        try:
            resp = requests.post(f"{API_BASE}/review", json={"query": query}, headers=get_auth_headers(), timeout=90)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["review_data"] = data
            else:
                st.error(f"API error: {resp.status_code} — {resp.text[:200]}")
        except Exception as e:
            st.error(f"Connection error: {e}")

review_data = st.session_state.get("review_data")

if review_data:
    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("Papers Reviewed", review_data.get("papers_reviewed", 0), "Studies included in review")
    with col2:
        sections = review_data.get("sections", [])
        render_metric_card("Sections", len(sections), "Structured review sections")

    review_text = review_data.get("review", "")
    if review_text:
        st.markdown('<div class="section-title">Literature Review</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="panel" style="line-height:1.8;">{review_text}</div>',
            unsafe_allow_html=True,
        )

        st.download_button(
            label="Download Review",
            data=review_text,
            file_name=f"literature_review_{query.replace(' ', '_')[:50]}.md",
            mime="text/markdown",
        )
    else:
        st.warning("No review content generated.")

    if sections:
        st.markdown('<div class="section-title">Sections</div>', unsafe_allow_html=True)
        for s in sections:
            st.markdown(f"- **{s}**")

else:
    st.markdown(
        '<div class="empty-card" style="margin-top:1.2rem;">'
        "Enter a research question above to generate a comprehensive literature review."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-title">Example Queries</div>', unsafe_allow_html=True)
    examples = [
        "creatine supplementation muscle strength",
        "omega-3 fatty acids cardiovascular disease",
        "metformin anti-aging properties",
        "vitamin D immune function",
        "intermittent fasting weight loss",
    ]
    for ex in examples:
        st.markdown(f"- `{ex}`")
