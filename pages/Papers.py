from __future__ import annotations

import streamlit as st

from ui import fetch_paper, get_paper_detail, get_response, init_page, render_evidence_cards, render_hero


init_page("MyoCortex | Papers", "MP")

render_hero(
    "Read source papers without leaving the app.",
    "Pick a PMID from the latest evidence set or load one directly from the paper endpoint.",
    eyebrow="Paper Reader",
)

response = get_response()

left, right = st.columns([1.05, 1.35], gap="large")

with left:
    st.markdown('<div class="section-title">Available evidence</div>', unsafe_allow_html=True)
    cards = response.get("evidence_cards", []) if response else []
    if cards:
        options = {
            f"{card['pmid']} • {card['title'][:75]}": card["pmid"]
            for card in cards
        }
        selected = st.selectbox("Choose a paper from the latest run", list(options.keys()))
        if st.button("Load selected paper"):
            result = fetch_paper(options[selected])
            if not result["ok"]:
                st.error(f"Paper fetch failed: {result['error']}")
        render_evidence_cards(cards)
    else:
        st.info("Run a query on Home first to populate evidence cards.")

with right:
    st.markdown('<div class="section-title">Direct PMID lookup</div>', unsafe_allow_html=True)
    pmid = st.text_input("PMID", placeholder="Enter a PubMed ID")
    if st.button("Fetch by PMID", type="primary") and pmid.strip():
        result = fetch_paper(pmid.strip())
        if not result["ok"]:
            st.error(f"Paper fetch failed: {result['error']}")

    paper = get_paper_detail()
    if paper:
        doi = paper.get("doi")
        doi_line = f"https://doi.org/{doi}" if doi else "No DOI returned"
        authors = ", ".join(paper.get("authors", [])) or "Authors unavailable"
        st.markdown(
            f"""
            <div class="panel">
                <div class="eyebrow">PMID {paper.get("pmid", "")}</div>
                <div style="font-size:1.6rem;font-weight:750;">{paper.get("title", "Untitled paper")}</div>
                <div class="muted" style="margin-top:0.6rem;">{authors}</div>
                <div class="muted small" style="margin-top:0.4rem;">{paper.get("journal", "Unknown journal")} • {paper.get("year", "n/a")}</div>
                <div class="muted small" style="margin-top:0.4rem;">{doi_line}</div>
                <div style="margin-top:1rem;line-height:1.7;">{paper.get("abstract", "No abstract returned.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown('<div class="empty-card">Load a paper to display its abstract and metadata here.</div>', unsafe_allow_html=True)
