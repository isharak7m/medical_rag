from __future__ import annotations

import streamlit as st

from ui import get_response, init_page, render_claim_links, render_hero, render_metric_card, require_login


init_page("MyoCortex | Claims", "MC")
require_login()

render_hero(
    "Trace extracted claims back to their paper sources.",
    "This view focuses on stance distribution and claim-level evidence links from the latest query run.",
    eyebrow="Claim Extraction",
)

response = get_response()
if not response:
    st.markdown('<div class="empty-card">Run a query on Home to inspect extracted claims.</div>', unsafe_allow_html=True)
else:
    contradiction = response.get("contradiction", {})
    cols = st.columns(3)
    with cols[0]:
        render_metric_card("Supporting", contradiction.get("support_count", 0), "Positive evidence links")
    with cols[1]:
        render_metric_card("Opposing", contradiction.get("oppose_count", 0), "Contrary evidence links")
    with cols[2]:
        render_metric_card("Neutral", contradiction.get("neutral_count", 0), "Descriptive or mixed findings")

    st.markdown('<div class="section-title">Claim links</div>', unsafe_allow_html=True)
    render_claim_links(response.get("claim_links", []))
