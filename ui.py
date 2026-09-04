from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests
import streamlit as st
from streamlit.errors import StreamlitAPIException

API_BASE = "http://localhost:8000/api/v1"
EVAL_LOG_PATH = Path(__file__).parent / "eval_log.jsonl"

STANCE_LABELS = {
    "support": "Support",
    "oppose": "Oppose",
    "neutral": "Neutral",
}

STANCE_CLASS = {
    "support": "stance-support",
    "oppose": "stance-oppose",
    "neutral": "stance-neutral",
}


def get_auth_headers() -> dict[str, str]:
    """Return Authorization header if user is logged in."""
    token = st.session_state.get("auth_token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}


def is_logged_in() -> bool:
    """Check if user is authenticated."""
    return bool(st.session_state.get("auth_token"))


def require_login():
    """Show inline login form if not authenticated, then stop page rendering."""
    if not is_logged_in():
        st.markdown('<div class="section-title">Sign In Required</div>', unsafe_allow_html=True)
        with st.form("inline-login", clear_on_submit=False):
            li_user = st.text_input("Username", key="li_user")
            li_pass = st.text_input("Password", type="password", key="li_pass")
            li_sub = st.form_submit_button("Sign In", use_container_width=True)
        if li_sub and li_user and li_pass:
            try:
                resp = requests.post(f"{API_BASE}/auth/login", json={
                    "username": li_user, "password": li_pass,
                }, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["auth_token"] = data["token"]
                    st.session_state["auth_user_id"] = data["user_id"]
                    st.session_state["auth_username"] = data["username"]
                    st.rerun()
                else:
                    st.error("Invalid username or password.")
            except Exception as e:
                st.error(f"Error: {e}")
        st.markdown(
            '<div class="empty-card" style="margin-top:1rem;">'
            'Or go to the <a href="/" style="color:var(--accent);">home page</a> to register a new account.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.stop()


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #07111f;
            --bg-soft: #0d1a2c;
            --panel: rgba(12, 24, 42, 0.78);
            --panel-strong: rgba(17, 31, 54, 0.96);
            --text: #f4f7fb;
            --muted: #9fb1c8;
            --line: rgba(132, 169, 211, 0.18);
            --accent: #6ee7d8;
            --accent-2: #8cb8ff;
            --accent-warm: #ffd280;
            --good: #62f2bc;
            --bad: #ff8d8d;
            --neutral: #ffd66e;
            --shadow: 0 22px 80px rgba(0, 0, 0, 0.35);
            --radius: 22px;
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(110, 231, 216, 0.16), transparent 28%),
                radial-gradient(circle at top right, rgba(140, 184, 255, 0.18), transparent 26%),
                linear-gradient(180deg, #050d18 0%, #091321 42%, #07111f 100%);
            color: var(--text);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 3rem;
            max-width: 1220px;
        }

        [data-testid="stHeader"] {
            background: rgba(7, 17, 31, 0.72);
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, rgba(4, 12, 22, 0.96), rgba(9, 19, 33, 0.92));
            border-right: 1px solid var(--line);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] > ul {
            display: none;
        }

        section[data-testid="stSidebar"] div[data-testid="stSidebarNav"] {
            display: none;
        }

        h1, h2, h3 {
            letter-spacing: -0.03em;
        }

        .hero {
            position: relative;
            overflow: hidden;
            padding: 1.8rem 1.8rem 1.5rem;
            border: 1px solid rgba(145, 189, 236, 0.18);
            border-radius: 28px;
            background:
                linear-gradient(135deg, rgba(14, 31, 53, 0.94), rgba(7, 17, 31, 0.9)),
                radial-gradient(circle at top right, rgba(110, 231, 216, 0.18), transparent 28%);
            box-shadow: var(--shadow);
            animation: rise-in 0.55s ease-out;
        }

        .hero:before {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(120deg, transparent, rgba(255, 255, 255, 0.06), transparent);
            transform: translateX(-100%);
            animation: shine 7s linear infinite;
        }

        .eyebrow {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            font-size: 0.72rem;
            text-transform: uppercase;
            letter-spacing: 0.16em;
            color: var(--accent);
            margin-bottom: 0.8rem;
        }

        .panel, .metric-card, .evidence-card, .empty-card {
            border: 1px solid var(--line);
            border-radius: var(--radius);
            background: var(--panel);
            backdrop-filter: blur(14px);
            box-shadow: var(--shadow);
            animation: rise-in 0.45s ease-out;
        }

        .panel {
            padding: 1.25rem 1.3rem;
        }

        .metric-card {
            padding: 1rem 1.15rem;
            min-height: 120px;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }

        .metric-card:hover, .evidence-card:hover {
            transform: translateY(-4px);
            border-color: rgba(110, 231, 216, 0.4);
            box-shadow: 0 28px 70px rgba(0, 0, 0, 0.42);
        }

        .metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.1em;
        }

        .metric-value {
            font-size: 2rem;
            font-weight: 700;
            margin-top: 0.35rem;
        }

        .metric-hint {
            color: var(--muted);
            font-size: 0.92rem;
            margin-top: 0.45rem;
        }

        .evidence-card {
            padding: 1.05rem 1.1rem;
            margin-bottom: 1rem;
            transition: transform 0.18s ease, border-color 0.18s ease, box-shadow 0.18s ease;
        }

        .card-topline {
            display: flex;
            justify-content: space-between;
            gap: 0.8rem;
            margin-bottom: 0.7rem;
            align-items: center;
            flex-wrap: wrap;
        }

        .card-title {
            font-size: 1.05rem;
            font-weight: 700;
            color: var(--text);
            margin-bottom: 0.4rem;
        }

        .muted {
            color: var(--muted);
        }

        .stance-pill {
            display: inline-flex;
            align-items: center;
            padding: 0.28rem 0.7rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 700;
            border: 1px solid transparent;
        }

        .stance-support {
            color: #c5ffea;
            background: rgba(98, 242, 188, 0.12);
            border-color: rgba(98, 242, 188, 0.22);
        }

        .stance-oppose {
            color: #ffd2d2;
            background: rgba(255, 141, 141, 0.12);
            border-color: rgba(255, 141, 141, 0.22);
        }

        .stance-neutral {
            color: #ffe9ab;
            background: rgba(255, 214, 110, 0.12);
            border-color: rgba(255, 214, 110, 0.22);
        }

        .progress-shell {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(255, 255, 255, 0.08);
            margin: 0.65rem 0 0.35rem;
        }

        .progress-fill {
            height: 100%;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--accent), var(--accent-2));
        }

        .empty-card {
            padding: 1rem 1.1rem;
            color: var(--muted);
        }

        .section-title {
            margin: 1.2rem 0 0.8rem;
            font-size: 1.2rem;
            font-weight: 700;
        }

        .nav-box {
            padding: 0.95rem 1rem;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: rgba(10, 20, 34, 0.74);
            margin-bottom: 0.8rem;
        }

        .small {
            font-size: 0.88rem;
        }

        .stButton > button, .stDownloadButton > button {
            border-radius: 14px;
            border: 1px solid rgba(110, 231, 216, 0.28);
            background: linear-gradient(135deg, rgba(110, 231, 216, 0.2), rgba(140, 184, 255, 0.18));
            color: var(--text);
            transition: transform 0.16s ease, box-shadow 0.16s ease, border-color 0.16s ease;
        }

        .stButton > button:hover, .stDownloadButton > button:hover {
            transform: translateY(-2px);
            border-color: rgba(110, 231, 216, 0.5);
            box-shadow: 0 16px 32px rgba(0, 0, 0, 0.25);
        }

        .stTextInput input, .stTextArea textarea {
            background: rgba(7, 17, 31, 0.78);
            border-radius: 14px;
            border: 1px solid rgba(132, 169, 211, 0.18);
            color: var(--text);
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem;
        }

        .stTabs [data-baseweb="tab"] {
            background: rgba(255, 255, 255, 0.03);
            border-radius: 14px;
            padding: 0.45rem 0.9rem;
        }

        @keyframes rise-in {
            from {
                opacity: 0;
                transform: translateY(16px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        @keyframes shine {
            to {
                transform: translateX(100%);
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_page(title: str, icon: str, show_sidebar: bool = True) -> None:
    st.set_page_config(page_title=title, page_icon=icon, layout="wide")
    inject_theme()
    init_state()
    if show_sidebar:
        render_sidebar()


def init_state() -> None:
    st.session_state.setdefault("last_query", "")
    st.session_state.setdefault("rich_response", None)
    st.session_state.setdefault("paper_detail", None)
    st.session_state.setdefault("auth_token", "")
    st.session_state.setdefault("auth_user_id", "")
    st.session_state.setdefault("auth_username", "")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
            <div class="nav-box">
                <div class="eyebrow">Research Console</div>
                <div style="font-size:1.15rem;font-weight:700;">MyoCortex</div>
                <div class="small muted">Biomedical evidence explorer with a repaired Streamlit frontend.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.page_link("dashboard.py", label="Home")
        st.page_link("pages/Papers.py", label="Papers")
        st.page_link("pages/Claims.py", label="Claims")
        st.page_link("pages/Contradictions.py", label="Contradictions")
        st.page_link("pages/Evaluation.py", label="Evaluation")
        st.page_link("pages/KnowledgeGraph.py", label="Knowledge Graph")
        st.page_link("pages/LiteratureReview.py", label="Literature Review")
        st.page_link("pages/Workspace.py", label="Workspace")

        st.markdown("---")
        username = st.session_state.get("auth_username", "")
        if username:
            st.markdown(
                f'<div class="small muted">Signed in as <strong>{username}</strong></div>',
                unsafe_allow_html=True,
            )
            if st.button("Sign Out", use_container_width=True):
                for key in ["auth_token", "auth_user_id", "auth_username"]:
                    st.session_state[key] = ""
                st.rerun()

        st.caption(f"API base: {API_BASE}")


def render_hero(title: str, subtitle: str, eyebrow: str = "Biomedical Research Workspace") -> None:
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{eyebrow}</div>
            <div style="font-size:2.3rem;font-weight:800;max-width:820px;">{title}</div>
            <div class="muted" style="font-size:1rem;max-width:820px;margin-top:0.6rem;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def call_health(timeout: int = 6) -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE}/health", headers=get_auth_headers(), timeout=timeout)
        response.raise_for_status()
        return {"ok": True, "data": response.json()}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def call_query(query: str, timeout: int = 120) -> dict[str, Any]:
    try:
        response = requests.post(
            f"{API_BASE}/query/rich",
            json={"query": query},
            headers=get_auth_headers(),
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        st.session_state["last_query"] = query
        st.session_state["rich_response"] = data
        return {"ok": True, "data": data}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def fetch_paper(pmid: str, timeout: int = 20) -> dict[str, Any]:
    try:
        response = requests.get(f"{API_BASE}/paper/{pmid}", headers=get_auth_headers(), timeout=timeout)
        response.raise_for_status()
        data = response.json()
        st.session_state["paper_detail"] = data
        return {"ok": True, "data": data}
    except requests.RequestException as exc:
        return {"ok": False, "error": str(exc)}


def render_metric_card(label: str, value: Any, hint: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-hint">{hint}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_confidence_bar(score: int) -> None:
    bounded = max(0, min(100, int(score)))
    st.markdown(
        f"""
        <div class="progress-shell">
            <div class="progress-fill" style="width:{bounded}%;"></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_evidence_cards(cards: list[dict[str, Any]]) -> None:
    if not cards:
        st.markdown('<div class="empty-card">No evidence cards are available yet.</div>', unsafe_allow_html=True)
        return

    for card in cards:
        stance = str(card.get("stance", "neutral")).lower()
        label = STANCE_LABELS.get(stance, stance.title())
        css_class = STANCE_CLASS.get(stance, "stance-neutral")
        sample = card.get("sample_size")
        sample_text = f"Sample size: {sample}" if sample else "Sample size unavailable"
        score_pct = round(float(card.get("relevance_score", 0.0)) * 100)
        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="card-topline">
                    <div class="muted small">PMID {card.get("pmid", "unknown")}</div>
                    <div class="stance-pill {css_class}">{label}</div>
                </div>
                <div class="card-title">{card.get("title", "Untitled paper")}</div>
                <div class="muted small">{sample_text} • Relevance {score_pct}%</div>
                <div style="margin-top:0.85rem;line-height:1.6;">{card.get("claim_text", "No extracted claim available.")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_claim_links(claim_links: list[dict[str, Any]]) -> None:
    if not claim_links:
        st.markdown('<div class="empty-card">No extracted claims are available for this run.</div>', unsafe_allow_html=True)
        return

    for item in claim_links:
        stance = str(item.get("stance", "neutral")).lower()
        label = STANCE_LABELS.get(stance, stance.title())
        css_class = STANCE_CLASS.get(stance, "stance-neutral")
        st.markdown(
            f"""
            <div class="evidence-card">
                <div class="card-topline">
                    <div class="card-title" style="font-size:1rem;">{item.get("paper_title", "Unknown paper")}</div>
                    <div class="stance-pill {css_class}">{label}</div>
                </div>
                <div class="muted small">PMID {item.get("pmid", "unknown")}</div>
                <div style="margin-top:0.75rem;line-height:1.6;">{item.get("claim_text", "")}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def get_response() -> dict[str, Any] | None:
    return st.session_state.get("rich_response")


def get_paper_detail() -> dict[str, Any] | None:
    return st.session_state.get("paper_detail")


def load_eval_log(limit: int = 100) -> list[dict[str, Any]]:
    if not EVAL_LOG_PATH.exists():
        return []

    rows: list[dict[str, Any]] = []
    with EVAL_LOG_PATH.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows[-limit:]
