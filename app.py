import streamlit as st
from agent.orchestrator import run_agent

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Assistant")
st.caption("Manage your Gmail and Google Drive — just ask!")

SYSTEM_PROMPT = {
    "role": "system",
    "content": (
        "You are a smart, friendly and patient AI assistant with access to Google Drive and Gmail tools."
    ),
}

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = [SYSTEM_PROMPT]

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if user_input := st.chat_input("Ask me anything..."):
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        with st.spinner("Working on it..."):
            try:
                reply, updated_history = run_agent(
                    user_input,
                    st.session_state.conversation_history,
                )
                st.session_state.conversation_history = updated_history
                st.write(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                error_msg = "Something went wrong. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})

with st.sidebar:
    st.header("💡 What can I help with?")
    st.markdown("""
    **Gmail**
    - List, read, send emails
    - Reply, delete, search

    **Google Drive**
    - Create folders and files
    - Read, move, rename, delete
    - Search files

    **Examples:**
    - "Show my last 5 emails"
    - "Send email to john@gmail.com"
    - "Create folder called Projects"
    - "Read notes.txt in Projects"
    """)

    st.divider()

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.conversation_history = [SYSTEM_PROMPT]
        st.rerun()