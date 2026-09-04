from __future__ import annotations

import streamlit as st
import requests

from ui import init_page, render_hero, render_metric_card, API_BASE


def extract_text_from_file(uploaded_file) -> str:
    """Extract text content from uploaded files."""
    name = uploaded_file.name.lower()

    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() or ""
            return text
        except ImportError:
            return "[PDF upload requires PyPDF2: pip install PyPDF2]"
        except Exception as e:
            return f"[PDF extraction failed: {e}]"

    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        except ImportError:
            return "[DOCX upload requires python-docx: pip install python-docx]"
        except Exception as e:
            return f"[DOCX extraction failed: {e}]"

    if name.endswith(".csv"):
        return uploaded_file.read().decode("utf-8", errors="replace")

    return uploaded_file.read().decode("utf-8", errors="replace")

init_page("MyoCortex | Research Workspace", "WS")

render_hero(
    "Research Workspace",
    "Create notes, hypotheses, and manuscripts with Git-like version history. Collaborate with your team through comments and shared artifacts.",
)

# ── Tabs ────────────────────────────────────────────────────
tab_artifacts, tab_workspace = st.tabs(["Research Artifacts", "Collaboration Workspace"])

# ── Artifacts Tab ───────────────────────────────────────────
with tab_artifacts:
    st.markdown('<div class="section-title">Create Artifact</div>', unsafe_allow_html=True)

    upload_mode = st.radio(
        "Input method",
        ["Type content", "Upload document"],
        horizontal=True,
        key="artifact_input_mode",
    )

    uploaded_content = ""

    if upload_mode == "Upload document":
        uploaded_file = st.file_uploader(
            "Upload a document",
            type=["txt", "md", "pdf", "docx", "csv"],
            help="Supported formats: TXT, MD, PDF, DOCX, CSV",
            key="artifact_file_uploader",
        )
        if uploaded_file:
            uploaded_content = extract_text_from_file(uploaded_file)
            st.success(f"Extracted {len(uploaded_content)} characters from {uploaded_file.name}")
            with st.expander("Preview uploaded content"):
                st.text_area("Preview", uploaded_content[:2000], height=200, disabled=True, key="file_preview")

    with st.form("artifact-form"):
        art_type = st.selectbox("Type", ["note", "hypothesis", "manuscript", "review"])
        title = st.text_input("Title", placeholder="Brief descriptive title")

        if upload_mode == "Type content":
            content = st.text_area("Content", height=200, placeholder="Write your research artifact here...")
        else:
            content = uploaded_content if uploaded_content else st.text_area("Content (from upload)", height=200, disabled=True)

        author = st.text_input("Author", value="researcher")
        tags_input = st.text_input("Tags (comma-separated)", placeholder="creatine, meta-analysis, safety")
        message = st.form_submit_button("Create Artifact")

    if message and title and content:
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        try:
            resp = requests.post(f"{API_BASE}/artifacts", json={
                "artifact_type": art_type,
                "title": title,
                "content": content,
                "author": author,
                "tags": tags,
            }, timeout=10)
            if resp.status_code == 200:
                st.success(f"Created: {resp.json().get('artifact_id')}")
                st.rerun()
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    st.markdown('<div class="section-title">Your Artifacts</div>', unsafe_allow_html=True)

    filter_type = st.selectbox("Filter by type", ["all", "note", "hypothesis", "manuscript", "review"])
    try:
        params = {} if filter_type == "all" else {"artifact_type": filter_type}
        resp = requests.get(f"{API_BASE}/artifacts", params=params, timeout=10)
        if resp.status_code == 200:
            artifacts = resp.json()
            if artifacts:
                for art in artifacts:
                    tags = art.get("tags", [])
                    tags_html = " ".join(f'<span class="stance-pill stance-neutral">{t}</span>' for t in tags)
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <div class="card-topline">
                                <div class="card-title">{art.get('title', 'Untitled')}</div>
                                <div class="stance-pill stance-support">{art.get('artifact_type', 'note')}</div>
                            </div>
                            <div class="muted small">ID: {art.get('artifact_id')} | v{art.get('current_version', 1)} | {art.get('author', 'unknown')}</div>
                            <div style="margin-top:0.6rem;">{art.get('content', '')[:300]}{'...' if len(art.get('content', '')) > 300 else ''}</div>
                            <div style="margin-top:0.5rem;">{tags_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Show version history
                    with st.expander(f"Version History ({art.get('artifact_id')})"):
                        try:
                            hist_resp = requests.get(f"{API_BASE}/artifacts/{art['artifact_id']}/history", timeout=10)
                            if hist_resp.status_code == 200:
                                for v in hist_resp.json():
                                    diff = v.get("diff_from_previous", "")
                                    st.markdown(f"**v{v.get('version_id', '?')}** — {v.get('message', '')} ({v.get('created_at', '')[:10]})")
                                    if diff:
                                        st.code(diff, language="diff")
                        except Exception:
                            st.caption("Could not load version history")

                    # Update form
                    with st.expander(f"Update {art.get('title', '')[:30]}"):
                        with st.form(key=f"update-{art['artifact_id']}"):
                            new_content = st.text_area("New content", value=art.get("content", ""), height=150)
                            update_msg = st.text_input("Update message", value="Updated")
                            update_btn = st.form_submit_button("Save Version")
                            if update_btn and new_content:
                                try:
                                    up_resp = requests.put(
                                        f"{API_BASE}/artifacts/{art['artifact_id']}",
                                        json={"content": new_content, "message": update_msg},
                                        timeout=10,
                                    )
                                    if up_resp.status_code == 200:
                                        st.success("Updated!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(str(e))
            else:
                st.markdown('<div class="empty-card">No artifacts yet. Create one above.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")

# ── Workspace Tab ───────────────────────────────────────────
with tab_workspace:
    st.markdown('<div class="section-title">Collaboration Workspaces</div>', unsafe_allow_html=True)

    with st.form("ws-form"):
        ws_name = st.text_input("Workspace Name", placeholder="My Research Team")
        ws_desc = st.text_input("Description", placeholder="Collaborative space for our research project")
        ws_author = st.text_input("Your Name", value="researcher")
        ws_create = st.form_submit_button("Create Workspace")

    if ws_create and ws_name:
        try:
            resp = requests.post(f"{API_BASE}/workspaces", json={
                "name": ws_name,
                "description": ws_desc,
                "created_by": ws_author,
            }, timeout=10)
            if resp.status_code == 200:
                st.success(f"Created workspace: {resp.json().get('workspace_id')}")
                st.rerun()
        except Exception as e:
            st.error(str(e))

    try:
        resp = requests.get(f"{API_BASE}/workspaces", timeout=10)
        if resp.status_code == 200:
            workspaces = resp.json()
            for ws in workspaces:
                members = ws.get("members", [])
                member_list = ", ".join(m.get("username", "?") for m in members)
                st.markdown(
                    f"""
                    <div class="evidence-card">
                        <div class="card-topline">
                            <div class="card-title">{ws.get('name', 'Unnamed')}</div>
                            <div class="stance-pill stance-support">{ws.get('workspace_id', '')}</div>
                        </div>
                        <div class="muted small">Created by {ws.get('created_by', '?')} | Members: {member_list}</div>
                        <div style="margin-top:0.5rem;">{ws.get('description', '')}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                with st.expander(f"Comments ({ws.get('workspace_id')})"):
                    try:
                        c_resp = requests.get(f"{API_BASE}/workspaces/{ws['workspace_id']}/comments", timeout=10)
                        if c_resp.status_code == 200:
                            for c in c_resp.json():
                                st.markdown(f"**{c.get('author', '?')}** ({c.get('created_at', '')[:10]}): {c.get('content', '')}")
                    except Exception:
                        pass

                    with st.form(key=f"comment-{ws['workspace_id']}"):
                        comment_author = st.text_input("Name", value="researcher", key=f"ca-{ws['workspace_id']}")
                        comment_text = st.text_area("Comment", placeholder="Add a comment or @mention a colleague", key=f"ct-{ws['workspace_id']}")
                        comment_btn = st.form_submit_button("Post Comment")
                        if comment_btn and comment_text:
                            try:
                                requests.post(f"{API_BASE}/workspaces/{ws['workspace_id']}/comment", json={
                                    "author": comment_author,
                                    "content": comment_text,
                                }, timeout=10)
                                st.rerun()
                            except Exception as e:
                                st.error(str(e))
    except Exception as e:
        st.error(f"Failed to load workspaces: {e}")
