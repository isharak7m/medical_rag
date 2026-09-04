from __future__ import annotations

import streamlit as st
import requests

from ui import init_page, render_hero, API_BASE

init_page("MyoCortex | Login", "🔐", show_sidebar=False)

# If already logged in, go to home
if st.session_state.get("auth_token"):
    st.rerun()

render_hero(
    "Welcome to MyoCortex",
    "Biomedical evidence explorer with multi-agent orchestration, knowledge graphs, and collaborative research workspaces.",
)

# ── Session state init ──────────────────────────────────────
if "auth_mode" not in st.session_state:
    st.session_state["auth_mode"] = "login"

# ── Toggle login/register ──────────────────────────────────
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

# ── Login form ──────────────────────────────────────────────
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
                st.error(f"Error: {resp.text[:200]}")
        except requests.ConnectionError:
            st.error("Cannot connect to API server. Make sure it's running on port 8000.")
        except Exception as e:
            st.error(f"Error: {e}")

# ── Register form ───────────────────────────────────────────
else:
    st.markdown('<div class="section-title">Create Account</div>', unsafe_allow_html=True)

    with st.form("register-form", clear_on_submit=False):
        reg_username = st.text_input("Username", placeholder="Choose a username (3-30 characters)", key="reg_user")
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
                    st.error("Username or email already exists.")
                else:
                    st.error(f"Error: {resp.text[:200]}")
            except requests.ConnectionError:
                st.error("Cannot connect to API server. Make sure it's running on port 8000.")
            except Exception as e:
                st.error(f"Error: {e}")

# ── Info ────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<div class="empty-card">'
    "New here? Create an account to get started. Your research artifacts, workspaces, and documents are private to your account."
    "</div>",
    unsafe_allow_html=True,
)
