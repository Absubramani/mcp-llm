import streamlit as st
from agent.orchestrator import run_agent
from datetime import date
import tempfile
import os

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
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a smart, friendly and patient AI assistant with access to Google Drive and Gmail tools."
}

# ── Session State Init ────────────────────────────────────────────────────────
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [SYSTEM_PROMPT]
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_paths" not in st.session_state:
    st.session_state.uploaded_file_paths = []
if "uploaded_file_names" not in st.session_state:
    st.session_state.uploaded_file_names = []
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "just_used_files" not in st.session_state:
    st.session_state.just_used_files = False
if "working_key_index" not in st.session_state:
    st.session_state.working_key_index = 0
if "working_key_date" not in st.session_state:
    st.session_state.working_key_date = str(date.today())

# ── Reset key index on new day ────────────────────────────────────────────────
if st.session_state.working_key_date != str(date.today()):
    st.session_state.working_key_index = 0
    st.session_state.working_key_date = str(date.today())

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AI Assistant")
    st.caption("Gmail & Google Drive")
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
        st.session_state.uploaded_file_paths = []
        st.session_state.uploaded_file_names = []
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

# ── Main Chat ─────────────────────────────────────────────────────────────────
st.title("🤖 AI Assistant")
st.caption("Manage your Gmail and Google Drive — just ask!")

if st.session_state.uploaded_file_names:
    names = ", ".join(st.session_state.uploaded_file_names)
    st.info(f"📎 **{names}** — ready to use")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ── Chat Input ────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything...")

if user_input:

    # Pass file paths to LLM if files uploaded — LLM decides what to do
    full_input = user_input
    if st.session_state.uploaded_file_paths:
        files_str = " ".join([f"[FILE: {p}]" for p in st.session_state.uploaded_file_paths])
        full_input = f"{user_input} {files_str}"

    # Show user message
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

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Working on it..."):
            try:
                reply, updated_history, working_key = run_agent(
                    full_input,
                    st.session_state.conversation_history,
                    st.session_state.working_key_index,
                )
                st.session_state.conversation_history = updated_history
                st.session_state.working_key_index = working_key
                st.write(reply)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply
                })
            except Exception as e:
                error_msg = "Something went wrong. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

    # Clear files after every message if files were attached
    if st.session_state.uploaded_file_paths:
        st.session_state.just_used_files = True
        st.session_state.uploader_key += 1
        st.rerun()