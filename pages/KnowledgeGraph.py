from __future__ import annotations

import streamlit as st
import requests
import json

from ui import init_page, render_hero, render_metric_card, API_BASE, get_auth_headers, require_login

init_page("MyoCortex | Knowledge Graph", "KG")
require_login()

render_hero(
    "Biomedical Knowledge Graph",
    "Extract entities (diseases, drugs, genes) and their relationships from research papers. Visualize connections across the biomedical literature.",
)

st.markdown('<div class="section-title">Build a Knowledge Graph</div>', unsafe_allow_html=True)

with st.form("kg-form"):
    query = st.text_input(
        "Research topic",
        value="creatine supplementation muscle strength",
        placeholder="Enter a biomedical topic to extract entities and relationships",
    )
    submitted = st.form_submit_button("Build Knowledge Graph")

if submitted and query:
    with st.spinner("Building knowledge graph..."):
        try:
            resp = requests.post(f"{API_BASE}/kg/build", json={"query": query}, headers=get_auth_headers(), timeout=60)
            if resp.status_code == 200:
                data = resp.json()
                st.session_state["kg_data"] = data
                st.success(f"Knowledge graph built: {data.get('entity_count', 0)} entities, {data.get('relation_count', 0)} relations")
            else:
                st.error(f"API error: {resp.status_code}")
        except Exception as e:
            st.error(f"Connection error: {e}")

kg_data = st.session_state.get("kg_data")

if kg_data:
    col1, col2 = st.columns(2)
    with col1:
        render_metric_card("Entities", kg_data.get("entity_count", 0), "Extracted biomedical entities")
    with col2:
        render_metric_card("Relations", kg_data.get("relation_count", 0), "Relationships between entities")

    entity_types = kg_data.get("entity_types", {})
    if entity_types:
        st.markdown('<div class="section-title">Entity Types</div>', unsafe_allow_html=True)
        type_cols = st.columns(len(entity_types))
        for i, (etype, count) in enumerate(entity_types.items()):
            with type_cols[i]:
                render_metric_card(etype, count, f"{etype} entities found")

    top_entities = kg_data.get("top_entities", [])
    if top_entities:
        st.markdown('<div class="section-title">Top Entities</div>', unsafe_allow_html=True)
        for ent in top_entities[:15]:
            sources = len(ent.get("sources", [])) if isinstance(ent.get("sources"), list) else ent.get("sources", 0)
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="card-topline">
                        <div class="card-title">{ent.get('name', 'Unknown')}</div>
                        <div class="stance-pill stance-support">{ent.get('type', 'Unknown')}</div>
                    </div>
                    <div class="muted small">ID: {ent.get('id', 'N/A')} | Sources: {sources} paper(s)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    nodes = kg_data.get("nodes", [])
    edges = kg_data.get("edges", [])
    if nodes:
        st.markdown('<div class="section-title">Graph Visualization</div>', unsafe_allow_html=True)
        graph_json = json.dumps({"nodes": nodes, "edges": edges}, indent=2)
        st.code(graph_json, language="json")
        st.caption(f"{len(nodes)} nodes and {len(edges)} edges. Full data via API: GET /api/v1/kg/graph/{query}")

    if edges:
        st.markdown('<div class="section-title">Relationships</div>', unsafe_allow_html=True)
        for edge in edges[:20]:
            src = edge.get("source", "")
            tgt = edge.get("target", "")
            rel = edge.get("relation", "")
            evidence = edge.get("evidence", "")
            st.markdown(
                f"""
                <div class="evidence-card">
                    <div class="card-topline">
                        <div class="card-title">{src} → {rel} → {tgt}</div>
                        <div class="stance-pill stance-neutral">{rel}</div>
                    </div>
                    <div class="muted small">{evidence}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
else:
    st.markdown(
        '<div class="empty-card" style="margin-top:1.2rem;">'
        "Enter a topic above to build a knowledge graph from biomedical literature."
        "</div>",
        unsafe_allow_html=True,
    )
