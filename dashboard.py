from __future__ import annotations

import streamlit as st
import requests

from ui import (
    inject_theme, init_state, render_sidebar,
    call_health, call_query, get_response,
    render_confidence_bar, render_evidence_cards, render_hero, render_metric_card,
    is_logged_in, API_BASE,
)


# ── Always init config + theme + state ──────────────────────
st.set_page_config(page_title="MyoCortex | Home", page_icon="MC", layout="wide")
inject_theme()
init_state()


# ══════════════════════════════════════════════════════════════
# NOT LOGGED IN — show login/register form, NO sidebar
# ══════════════════════════════════════════════════════════════
if not is_logged_in():
    render_hero(
        "Welcome to MyoCortex",
        "Biomedical evidence explorer with multi-agent orchestration, knowledge graphs, and collaborative research workspaces.",
    )

    # Session state for auth mode
    if "auth_mode" not in st.session_state:
        st.session_state["auth_mode"] = "login"

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        mode = st.radio(
            "Action",
            ["Login", "Register"],
            horizontal=True,
            index=0 if st.session_state["auth_mode"] == "login" else 1,
            key="auth_mode_radio",
        )
        st.session_state["auth_mode"] = mode.lower()

    if st.session_state["auth_mode"] == "login":
        st.markdown('<div class="section-title">Sign In</div>', unsafe_allow_html=True)
        with st.form("login-form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter your username")
            password = st.text_input("Password", type="password", placeholder="Enter your password")
            submitted = st.form_submit_button("Sign In", use_container_width=True)

        if submitted and username and password:
            try:
                resp = requests.post(f"{API_BASE}/auth/login", json={
                    "username": username,
                    "password": password,
                }, timeout=10)
                if resp.status_code == 200:
                    data = resp.json()
                    st.session_state["auth_token"] = data["token"]
                    st.session_state["auth_user_id"] = data["user_id"]
                    st.session_state["auth_username"] = data["username"]
                    st.rerun()
                elif resp.status_code == 401:
                    st.error("Invalid username or password.")
                else:
                    detail = resp.json().get("detail", resp.text[:200]) if resp.text else "Unknown error"
                    st.error(f"Error: {detail}")
            except requests.ConnectionError:
                st.error("Cannot connect to API server. Make sure it's running on port 8000.")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.markdown('<div class="section-title">Create Account</div>', unsafe_allow_html=True)
        with st.form("register-form", clear_on_submit=False):
            reg_username = st.text_input("Username", placeholder="Choose a username (3-30 chars)", key="reg_user")
            reg_email = st.text_input("Email", placeholder="your@email.com", key="reg_email")
            reg_password = st.text_input("Password", type="password", placeholder="At least 6 characters", key="reg_pass")
            reg_password2 = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_pass2")
            registered = st.form_submit_button("Create Account", use_container_width=True)

        if registered:
            if not reg_username or not reg_email or not reg_password:
                st.error("All fields are required.")
            elif reg_password != reg_password2:
                st.error("Passwords do not match.")
            elif len(reg_password) < 6:
                st.error("Password must be at least 6 characters.")
            else:
                try:
                    resp = requests.post(f"{API_BASE}/auth/register", json={
                        "username": reg_username,
                        "email": reg_email,
                        "password": reg_password,
                    }, timeout=10)
                    if resp.status_code == 200:
                        data = resp.json()
                        st.session_state["auth_token"] = data["token"]
                        st.session_state["auth_user_id"] = data["user_id"]
                        st.session_state["auth_username"] = data["username"]
                        st.rerun()
                    elif resp.status_code == 400:
                        detail = resp.json().get("detail", "Registration failed") if resp.text else "Registration failed"
                        st.error(detail)
                    else:
                        detail = resp.json().get("detail", resp.text[:200]) if resp.text else "Unknown error"
                        st.error(f"Error: {detail}")
                except requests.ConnectionError:
                    st.error("Cannot connect to API server. Make sure it's running on port 8000.")
                except Exception as e:
                    st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown(
        '<div class="empty-card">'
        "Create an account to get started. Your artifacts and workspaces are private to your account."
        "</div>",
        unsafe_allow_html=True,
    )
    st.stop()


# ══════════════════════════════════════════════════════════════
# LOGGED IN — full dashboard with sidebar
# ══════════════════════════════════════════════════════════════
render_sidebar()

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
