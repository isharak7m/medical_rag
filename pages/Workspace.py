from __future__ import annotations

import io
import streamlit as st
import requests

from ui import init_page, render_hero, render_metric_card, API_BASE, get_auth_headers, require_login


def extract_text_from_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".txt") or name.endswith(".md"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    if name.endswith(".pdf"):
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(uploaded_file)
            return "".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            return f"[PDF extraction failed: {e}]"
    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(uploaded_file)
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            return f"[DOCX extraction failed: {e}]"
    if name.endswith(".csv"):
        return uploaded_file.read().decode("utf-8", errors="replace")
    return uploaded_file.read().decode("utf-8", errors="replace")


init_page("MyoCortex | Research Workspace", "WS")
require_login()

render_hero(
    "Research Workspace",
    "Collaborative research environment. Create workspaces, invite team members, upload and share documents with version control.",
)


# ── Session state ──────────────────────────────────────────
if "active_workspace" not in st.session_state:
    st.session_state["active_workspace"] = None
if "ws_tab" not in st.session_state:
    st.session_state["ws_tab"] = "workspaces"


# ── Sidebar: Quick actions ─────────────────────────────────
st.sidebar.markdown("### Workspace")
ws_action = st.sidebar.radio(
    "Navigate",
    ["All Workspaces", "My Artifacts"],
    label_visibility="collapsed",
    key="ws_nav",
)


# ══════════════════════════════════════════════════════════════
# MY ARTIFACTS (personal research documents)
# ══════════════════════════════════════════════════════════════
if ws_action == "My Artifacts":
    st.markdown('<div class="section-title">My Research Artifacts</div>', unsafe_allow_html=True)

    create_mode = st.radio(
        "Create from",
        ["Upload file", "Type content"],
        horizontal=True,
        key="artifact_create_mode",
    )

    if create_mode == "Upload file":
        uploaded_file = st.file_uploader(
            "Upload document",
            type=["txt", "md", "pdf", "docx", "csv"],
            help="TXT, MD, PDF, DOCX, CSV",
            key="artifact_upload",
        )
        if uploaded_file:
            uploaded_content = extract_text_from_file(uploaded_file)
            st.session_state["uploaded_artifact_content"] = uploaded_content
            st.success(f"Extracted {len(uploaded_content)} characters from {uploaded_file.name}")
    else:
        st.session_state["uploaded_artifact_content"] = ""

    with st.form("artifact-form", clear_on_submit=True):
        art_type = st.selectbox("Type", ["note", "hypothesis", "manuscript", "review"])
        title = st.text_input("Title", placeholder="Brief descriptive title")

        existing_content = st.session_state.get("uploaded_artifact_content", "")
        content = st.text_area(
            "Content",
            height=200,
            placeholder="Write your research artifact here..." if not existing_content else "Content loaded from upload",
            value=existing_content,
        )

        author = st.text_input("Author", value="researcher")
        tags_input = st.text_input("Tags (comma-separated)", placeholder="creatine, meta-analysis")
        submitted = st.form_submit_button("Create Artifact")

    if submitted and title and content:
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]
        try:
            resp = requests.post(f"{API_BASE}/artifacts", json={
                "artifact_type": art_type,
                "title": title,
                "content": content,
                "author": author,
                "tags": tags,
            }, headers=get_auth_headers(), timeout=10)
            if resp.status_code == 200:
                st.session_state["uploaded_artifact_content"] = ""
                st.success(f"Created: {resp.json().get('artifact_id')}")
                st.rerun()
            else:
                st.error(f"Error: {resp.text}")
        except Exception as e:
            st.error(f"Connection error: {e}")

    st.markdown("---")
    st.markdown('<div class="section-title">All Artifacts</div>', unsafe_allow_html=True)

    filter_type = st.selectbox("Filter", ["all", "note", "hypothesis", "manuscript", "review"])
    try:
        params = {} if filter_type == "all" else {"artifact_type": filter_type}
        resp = requests.get(f"{API_BASE}/artifacts", params=params, headers=get_auth_headers(), timeout=10)
        if resp.status_code == 200:
            artifacts = resp.json()
            if artifacts:
                for art in artifacts:
                    tags = art.get("tags", [])
                    tags_html = " ".join(f'<span class="stance-pill stance-neutral">{t}</span>' for t in tags)
                    content_preview = art.get("content", "")[:300]
                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <div class="card-topline">
                                <div class="card-title">{art.get('title', 'Untitled')}</div>
                                <div class="stance-pill stance-support">{art.get('artifact_type', 'note')}</div>
                            </div>
                            <div class="muted small">ID: {art.get('artifact_id')} | v{art.get('current_version', 1)} | {art.get('author', '?')}</div>
                            <div style="margin-top:0.6rem;">{content_preview}{'...' if len(art.get('content', '')) > 300 else ''}</div>
                            <div style="margin-top:0.5rem;">{tags_html}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Download button
                    st.download_button(
                        label="Download",
                        data=art.get("content", ""),
                        file_name=f"{art.get('title', 'artifact').replace(' ', '_')[:50]}.md",
                        mime="text/markdown",
                        key=f"dl-{art['artifact_id']}",
                    )

                    # Version history
                    with st.expander(f"Version History ({art.get('artifact_id')})"):
                        try:
                            hist_resp = requests.get(f"{API_BASE}/artifacts/{art['artifact_id']}/history", headers=get_auth_headers(), timeout=10)
                            if hist_resp.status_code == 200:
                                for v in hist_resp.json():
                                    diff = v.get("diff_from_previous", "")
                                    st.markdown(f"**v{v.get('version_id', '?')}** — {v.get('message', '')} ({v.get('created_at', '')[:10]})")
                                    if diff:
                                        st.code(diff, language="diff")
                        except Exception:
                            st.caption("Could not load version history")

                    # Update with file upload
                    with st.expander(f"Update {art.get('title', '')[:30]}"):
                        update_mode = st.radio(
                            "Update from",
                            ["Type content", "Upload file"],
                            horizontal=True,
                            key=f"update_mode-{art['artifact_id']}",
                        )

                        update_key = f"update_content_{art['artifact_id']}"
                        if update_mode == "Upload file":
                            update_file = st.file_uploader(
                                "Upload updated document",
                                type=["txt", "md", "pdf", "docx", "csv"],
                                key=f"update_file-{art['artifact_id']}",
                            )
                            if update_file:
                                st.session_state[update_key] = extract_text_from_file(update_file)
                                st.success(f"Extracted {len(st.session_state[update_key])} characters")

                        with st.form(key=f"update-{art['artifact_id']}"):
                            if update_mode == "Type content":
                                new_content = st.text_area("New content", value=art.get("content", ""), height=150)
                            else:
                                existing = st.session_state.get(update_key, "")
                                new_content = st.text_area("Content from upload", value=existing, height=150)

                            update_msg = st.text_input("Update message", value="Updated")
                            update_btn = st.form_submit_button("Save Version")
                            if update_btn and new_content:
                                try:
                                    up_resp = requests.put(
                                        f"{API_BASE}/artifacts/{art['artifact_id']}",
                                        json={"content": new_content, "message": update_msg},
                                        headers=get_auth_headers(),
                                        timeout=10,
                                    )
                                    if up_resp.status_code == 200:
                                        st.session_state.pop(update_key, None)
                                        st.success("Updated!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(str(e))
            else:
                st.markdown('<div class="empty-card">No artifacts yet. Create one above.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load artifacts: {e}")


# ══════════════════════════════════════════════════════════════
# ALL WORKSPACES
# ══════════════════════════════════════════════════════════════
else:
    st.markdown('<div class="section-title">Research Workspaces</div>', unsafe_allow_html=True)

    # ── Create workspace ─────────────────────────────────────
    with st.expander("Create New Workspace"):
        with st.form("ws-create"):
            ws_name = st.text_input("Workspace Name", placeholder="My Research Team")
            ws_desc = st.text_input("Description", placeholder="Collaborative space for our project")
            ws_author = st.text_input("Your Name", value="researcher")
            ws_create = st.form_submit_button("Create Workspace")

        if ws_create and ws_name:
            try:
                resp = requests.post(f"{API_BASE}/workspaces", json={
                    "name": ws_name,
                    "description": ws_desc,
                    "created_by": ws_author,
                }, headers=get_auth_headers(), timeout=10)
                if resp.status_code == 200:
                    st.success(f"Created: {resp.json().get('workspace_id')}")
                    st.rerun()
            except Exception as e:
                st.error(str(e))

    # ── List workspaces ──────────────────────────────────────
    try:
        resp = requests.get(f"{API_BASE}/workspaces", headers=get_auth_headers(), timeout=10)
        if resp.status_code == 200:
            workspaces = resp.json()
            if workspaces:
                for ws in workspaces:
                    ws_id = ws.get("workspace_id", "")
                    members = ws.get("members", [])
                    member_names = ", ".join(m.get("username", "?") for m in members)
                    member_count = len(members)

                    st.markdown(
                        f"""
                        <div class="evidence-card">
                            <div class="card-topline">
                                <div class="card-title">{ws.get('name', 'Unnamed')}</div>
                                <div class="stance-pill stance-support">{member_count} member{'s' if member_count != 1 else ''}</div>
                            </div>
                            <div class="muted small">Created by {ws.get('created_by', '?')} | ID: {ws_id}</div>
                            <div style="margin-top:0.5rem;">{ws.get('description', '')}</div>
                            <div style="margin-top:0.5rem;" class="muted small">Members: {member_names}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # ── Workspace detail tabs ──────────────────
                    tab_docs, tab_members, tab_comments = st.tabs(["Documents", "Members", "Discussion"])

                    # ── Documents tab ─────────────────────────
                    with tab_docs:
                        st.markdown("**Upload Document to Workspace**")
                        doc_col1, doc_col2 = st.columns([3, 1])
                        with doc_col1:
                            ws_file = st.file_uploader(
                                "Upload",
                                type=["txt", "md", "pdf", "docx", "csv"],
                                key=f"ws-upload-{ws_id}",
                            )
                        with doc_col2:
                            ws_doc_type = st.selectbox("Type", ["note", "manuscript", "review"], key=f"ws-type-{ws_id}")
                            ws_doc_title = st.text_input("Title", value="", placeholder="Document title", key=f"ws-title-{ws_id}")
                            ws_author_field = st.text_input("Author", value="researcher", key=f"ws-author-{ws_id}")

                        if ws_file and ws_doc_title:
                            doc_content = extract_text_from_file(ws_file)
                            if st.button(f"Upload to {ws.get('name')}", key=f"ws-btn-{ws_id}"):
                                try:
                                    art_resp = requests.post(f"{API_BASE}/artifacts", json={
                                        "artifact_type": ws_doc_type,
                                        "title": ws_doc_title,
                                        "content": doc_content,
                                        "author": ws_author_field,
                                        "tags": [ws.get("name", "")],
                                    }, headers=get_auth_headers(), timeout=10)
                                    if art_resp.status_code == 200:
                                        artifact_id = art_resp.json().get("artifact_id")
                                        share_resp = requests.post(
                                            f"{API_BASE}/workspaces/{ws_id}/share",
                                            json={"artifact_id": artifact_id},
                                            headers=get_auth_headers(),
                                            timeout=10,
                                        )
                                        st.success(f"Uploaded {ws_doc_title} to workspace!")
                                        st.rerun()
                                except Exception as e:
                                    st.error(str(e))

                        # List shared artifacts
                        shared_ids = ws.get("shared_artifacts", [])
                        if shared_ids:
                            st.markdown("**Shared Documents:**")
                            for art_id in shared_ids:
                                try:
                                    art_resp = requests.get(f"{API_BASE}/artifacts/{art_id}", headers=get_auth_headers(), timeout=10)
                                    if art_resp.status_code == 200:
                                        art = art_resp.json()
                                        c1, c2 = st.columns([4, 1])
                                        with c1:
                                            st.markdown(f"**{art.get('title', 'Untitled')}** ({art.get('artifact_type', 'note')}) by {art.get('author', '?')}")
                                        with c2:
                                            dl_url = f"{API_BASE}/artifacts/{art_id}/download"
                                            st.markdown(f"[Download]({dl_url})")
                                except Exception:
                                    pass

                        # Show artifacts tagged with this workspace
                        try:
                            arts_resp = requests.get(f"{API_BASE}/artifacts", params={"tag": ws.get("name", "")}, headers=get_auth_headers(), timeout=10)
                            if arts_resp.status_code == 200:
                                tagged = arts_resp.json()
                                if tagged:
                                    st.markdown("**Workspace Documents:**")
                                    for art in tagged:
                                        c1, c2 = st.columns([4, 1])
                                        with c1:
                                            st.markdown(f"**{art.get('title', 'Untitled')}** (v{art.get('current_version', 1)}) by {art.get('author', '?')}")
                                        with c2:
                                            dl_url = f"{API_BASE}/artifacts/{art['artifact_id']}/download"
                                            st.markdown(f"[Download]({dl_url})")
                        except Exception:
                            pass

                    # ── Members tab ──────────────────────────
                    with tab_members:
                        st.markdown("**Current Members:**")
                        for m in members:
                            role_color = "#62f2bc" if m.get("role") == "admin" else "#8cb8ff" if m.get("role") == "editor" else "#ffd66e"
                            st.markdown(
                                f'<span class="stance-pill" style="color:{role_color};border-color:{role_color}40;">'
                                f'{m.get("username", "?")} ({m.get("role", "viewer")})</span>',
                                unsafe_allow_html=True,
                            )

                        st.markdown("---")
                        st.markdown("**Add Member:**")
                        with st.form(key=f"add-member-{ws_id}"):
                            new_member = st.text_input("Username", placeholder="Enter teammate's name")
                            new_role = st.selectbox("Role", ["viewer", "editor", "admin"])
                            add_btn = st.form_submit_button("Add Member")
                            if add_btn and new_member:
                                try:
                                    member_resp = requests.post(
                                        f"{API_BASE}/workspaces/{ws_id}/members",
                                        json={"username": new_member, "role": new_role},
                                        headers=get_auth_headers(),
                                        timeout=10,
                                    )
                                    if member_resp.status_code == 200:
                                        st.success(f"Added {new_member} as {new_role}")
                                        st.rerun()
                                    else:
                                        st.error(f"Failed: {member_resp.text}")
                                except Exception as e:
                                    st.error(str(e))

                    # ── Discussion tab ───────────────────────
                    with tab_comments:
                        try:
                            c_resp = requests.get(f"{API_BASE}/workspaces/{ws_id}/comments", headers=get_auth_headers(), timeout=10)
                            if c_resp.status_code == 200:
                                comments = c_resp.json()
                                if comments:
                                    for c in comments:
                                        author = c.get("author", "?")
                                        created = c.get("created_at", "")[:10]
                                        content = c.get("content", "")
                                        st.markdown(
                                            f"""<div class="evidence-card" style="padding:0.8rem;">
                                            <div class="card-topline">
                                                <div class="muted small">{author}</div>
                                                <div class="muted small">{created}</div>
                                            </div>
                                            <div style="margin-top:0.4rem;">{content}</div>
                                            </div>""",
                                            unsafe_allow_html=True,
                                        )
                                else:
                                    st.caption("No comments yet.")
                        except Exception:
                            st.caption("Could not load comments.")

                        with st.form(key=f"comment-{ws_id}"):
                            comment_author = st.text_input("Your name", value="researcher", key=f"ca-{ws_id}")
                            comment_text = st.text_area("Comment", placeholder="Discussion, @mentions, notes...", key=f"ct-{ws_id}")
                            comment_btn = st.form_submit_button("Post")
                            if comment_btn and comment_text:
                                try:
                                    requests.post(f"{API_BASE}/workspaces/{ws_id}/comment", json={
                                        "author": comment_author,
                                        "content": comment_text,
                                    }, headers=get_auth_headers(), timeout=10)
                                    st.rerun()
                                except Exception as e:
                                    st.error(str(e))

            else:
                st.markdown('<div class="empty-card">No workspaces yet. Create one above to start collaborating.</div>', unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Failed to load workspaces: {e}")
