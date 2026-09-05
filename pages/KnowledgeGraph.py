from __future__ import annotations

import streamlit as st
import requests
from streamlit_agraph import agraph, Node, Edge, Config

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
                st.success(f"Built: {data.get('entity_count', 0)} entities, {data.get('relation_count', 0)} relations")
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

    nodes_data = kg_data.get("nodes", [])
    edges_data = kg_data.get("edges", [])

    if nodes_data:
        st.markdown('<div class="section-title">Graph Visualization</div>', unsafe_allow_html=True)

        TYPE_COLORS = {
            "Disease": "#ff8d8d",
            "Drug": "#62f2bc",
            "Gene": "#8cb8ff",
            "Protein": "#ffd66e",
            "Compound": "#c49bff",
            "Phenotype": "#ff9de2",
            "Intervention": "#6ee7d8",
            "Outcome": "#ffd280",
            "Nutrient": "#a8e6cf",
            "Supplement": "#dcedc1",
        }
        DEFAULT_COLOR = "#9fb1c8"

        nodes = []
        for n in nodes_data:
            ntype = n.get("type", "Unknown")
            color = TYPE_COLORS.get(ntype, DEFAULT_COLOR)
            label = n.get("name", n.get("id", "?"))
            title = f"{label}\nType: {ntype}"
            if n.get("sources"):
                src_count = len(n["sources"]) if isinstance(n["sources"], list) else n["sources"]
                title += f"\nSources: {src_count} paper(s)"
            nodes.append(Node(
                id=n.get("id", label),
                label=label,
                title=title,
                color=color,
                size=max(15, min(40, 15 + (len(n.get("sources", [])) if isinstance(n.get("sources"), list) else 0) * 5)),
            ))

        edges = []
        for e in edges_data:
            src = e.get("source", "")
            tgt = e.get("target", "")
            rel = e.get("relation", "")
            evidence = e.get("evidence", "")
            edges.append(Edge(
                source=src,
                target=tgt,
                label=rel,
                title=evidence,
                color="#557799",
                arrows="to",
            ))

        config = Config(
            width="100%",
            height=550,
            directed=True,
            physics={
                "enabled": True,
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 120,
                    "springConstant": 0.06,
                },
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 150},
            },
            interaction={
                "hover": True,
                "zoomView": True,
                "dragView": True,
                "multiselect": True,
            },
            node={
                "font": {"size": 13, "color": "#f4f7fb", "strokeWidth": 3, "strokeColor": "#000000"},
                "borderWidth": 2,
                "borderWidthSelected": 4,
            },
            edge={
                "font": {"size": 10, "color": "#9fb1c8", "align": "middle"},
                "smooth": {"type": "continuous"},
            },
        )

        agraph(nodes=nodes, edges=edges, config=config)

    if edges_data:
        st.markdown('<div class="section-title">Relationships</div>', unsafe_allow_html=True)
        for edge in edges_data[:20]:
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
