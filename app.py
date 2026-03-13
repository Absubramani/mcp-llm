import os
import re
import tempfile
from datetime import date
import streamlit as st

from agent.orchestrator import run_agent_stream
from agent.auth import (
    delete_token, exchange_code_for_token,
    get_auth_url, get_user_email,
    load_token, save_token,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.markdown("""
<style>
    .stApp { background-color: #0f0f0f; }
    section[data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .loader-container {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 0;
    margin-top: 4px;
}
.loader-text {
    color: #aaaaaa;
    font-style: italic;
    font-size: 15px;
    line-height: 1;
}
.loader-spinner {
    width: 14px;
    height: 14px;
    border: 2px solid #333333;
    border-top: 2px solid #888888;
    border-radius: 50%;
    animation: spin 0.8s linear infinite;
    flex-shrink: 0;
    margin-top: 1px;
}
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a smart, friendly and patient AI assistant with access to Google Drive and Gmail tools."
}

LOADING_HTML = """
<div style="display:flex;align-items:center;gap:8px;padding:6px 0;">
    <span style="color:#aaaaaa;font-style:italic;font-size:15px;">Working on it...</span>
    <div style="
        width:14px;height:14px;
        border:2px solid #333333;
        border-top:2px solid #aaaaaa;
        border-radius:50%;
        animation:spin 0.8s linear infinite;
        flex-shrink:0;
    "></div>
</div>
<style>@keyframes spin{0%{transform:rotate(0deg)}100%{transform:rotate(360deg)}}</style>
"""

# ── Helper: strip hidden id comments before display ───────────────────────────
def clean_for_display(text: str) -> str:
    """Remove <!-- ... --> comment lines used to pass attachment ids to the LLM."""
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL).strip()

# ── Session State Init ────────────────────────────────────────────────────────
defaults = {
    "user_email": None,
    "user_creds": None,
    "saved_email": None,
    "oauth_flow": None,
    "auth_url": None,
    "conversation_history": [SYSTEM_PROMPT],
    "messages": [],
    "uploaded_file_paths": [],
    "uploaded_file_names": [],
    "uploader_key": 0,
    "just_used_files": False,
    "working_key_index": 0,
    "working_key_date": str(date.today()),
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ── Daily Reset ───────────────────────────────────────────────────────────────
if st.session_state.working_key_date != str(date.today()):
    st.session_state.working_key_index = 0
    st.session_state.working_key_date = str(date.today())

# ── Handle OAuth Callback ─────────────────────────────────────────────────────
query_params = st.query_params
if "code" in query_params and st.session_state.user_email is None:
    code = query_params["code"]
    state = query_params.get("state", "")
    try:
        with st.spinner("Logging you in..."):
            creds = exchange_code_for_token(state, code)
            email = get_user_email(creds)
            if email == "unknown":
                st.error("Could not retrieve your email. Please try again.")
                st.stop()
            save_token(email, creds)
            st.session_state.user_email = email
            st.session_state.user_creds = creds
            st.session_state.saved_email = email
            st.session_state.auth_url = None
            st.query_params.clear()
            st.rerun()
    except Exception as e:
        st.error(f"Login failed: {str(e)}")
        st.session_state.auth_url = None
        st.stop()

# ── Try Loading Existing Token ────────────────────────────────────────────────
if st.session_state.user_email is None:
    saved_email = st.session_state.saved_email
    if saved_email:
        creds = load_token(saved_email)
        if creds:
            st.session_state.user_email = saved_email
            st.session_state.user_creds = creds

# ── Login Page ────────────────────────────────────────────────────────────────
if st.session_state.user_email is None:
    if not st.session_state.auth_url:
        try:
            auth_url, state = get_auth_url()
            st.session_state.auth_url = auth_url
        except Exception as e:
            st.error(f"Failed to generate login URL: {str(e)}")
            st.stop()

    st.title("🤖 AI Assistant")
    st.caption("Your personal Gmail & Google Drive assistant")
    st.divider()
    st.markdown("### Welcome! Please login to continue.")
    st.write("This app helps you manage your Gmail and Google Drive using AI.")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f'''
        <a href="{st.session_state.auth_url}" target="_self" style="
            display: inline-block;
            background-color: #4285f4;
            color: white;
            padding: 12px 24px;
            border-radius: 8px;
            font-size: 16px;
            font-weight: 500;
            text-decoration: none;
            cursor: pointer;
        ">🔐 Login with Google</a>
        ''',
        unsafe_allow_html=True
    )
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.caption("Gmail & Google Drive")
    st.divider()

    st.markdown(f"👤 **{st.session_state.user_email}**")
    st.divider()

    st.markdown("#### 📎 Attach File")
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "png", "jpg", "jpeg", "txt", "docx", "xlsx", "csv"],
        label_visibility="collapsed",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}"
    )

    if uploaded_files and not st.session_state.just_used_files:
        st.session_state.uploaded_file_paths = []
        st.session_state.uploaded_file_names = []
        for uploaded_file in uploaded_files:
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.session_state.uploaded_file_paths.append(tmp_path)
            st.session_state.uploaded_file_names.append(uploaded_file.name)
        st.success(f"✅ {len(uploaded_files)} file(s) ready!")
        st.caption("Now type your message!")
    elif st.session_state.just_used_files:
        st.session_state.just_used_files = False
        st.session_state.uploaded_file_paths = []
        st.session_state.uploaded_file_names = []
        st.caption("Upload to attach to email or Drive")
    else:
        st.caption("Upload to attach to email or Drive")

    st.divider()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.conversation_history = [SYSTEM_PROMPT]
        st.session_state.uploaded_file_paths = []
        st.session_state.uploaded_file_names = []
        st.session_state.just_used_files = False
        st.session_state.uploader_key += 1
        st.session_state.working_key_index = 0
        st.session_state.working_key_date = str(date.today())
        st.rerun()

    if st.button("🚪 Logout", use_container_width=True):
        delete_token(st.session_state.user_email)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

# ── Main Chat ─────────────────────────────────────────────────────────────────
st.title("🤖 AI Assistant")
st.caption("Manage your Gmail and Google Drive — just ask!")

if st.session_state.uploaded_file_names:
    names = ", ".join(st.session_state.uploaded_file_names)
    st.info(f"📎 **{names}** — ready to use")

# Render chat history — strip hidden id comments before display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(clean_for_display(msg["content"]))

# ── Chat Input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

if user_input:
    full_input = user_input
    if st.session_state.uploaded_file_paths:
        files_str = " ".join([f"[FILE: {p}]" for p in st.session_state.uploaded_file_paths])
        full_input = f"{user_input} {files_str}"

    with st.chat_message("user"):
        if st.session_state.uploaded_file_names:
            names = ", ".join(st.session_state.uploaded_file_names)
            st.write(f"{user_input} 📎 {names}")
        else:
            st.write(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": user_input if not st.session_state.uploaded_file_names
        else f"{user_input} 📎 {', '.join(st.session_state.uploaded_file_names)}"
    })

    with st.chat_message("assistant"):
        reply_box = st.empty()
        full_reply = []
        updated_history = None
        working_key = st.session_state.working_key_index

        reply_box.markdown(LOADING_HTML, unsafe_allow_html=True)

        try:
            for token, hist, key in run_agent_stream(
                full_input,
                st.session_state.conversation_history,
                st.session_state.working_key_index,
                st.session_state.user_creds,
            ):
                if token is not None:
                    full_reply.append(token)
                    reply_box.markdown(clean_for_display("".join(full_reply)))
                if hist is not None:
                    updated_history = hist
                if key is not None:
                    working_key = key

        except Exception as e:
            error_msg = "Something went wrong. Please try again."
            reply_box.error(error_msg)
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg
            })

        else:
            if updated_history:
                st.session_state.conversation_history = updated_history
            st.session_state.working_key_index = working_key

            final_reply = "".join(full_reply)
            if final_reply and final_reply.strip():
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": final_reply
                })

    if st.session_state.uploaded_file_paths:
        st.session_state.just_used_files = True
        st.session_state.uploader_key += 1
        st.rerun()