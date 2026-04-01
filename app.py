import os
import re
import tempfile
from datetime import date
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

from agent.orchestrator import run_agent_stream
from agent.auth import (
    delete_token, exchange_code_for_token,
    get_auth_url, get_user_email,
    load_token, save_token,
    # GitHub
    get_github_auth_url, exchange_github_code_for_token,
    get_github_username, save_github_token,
    load_github_token, delete_github_token,
)


# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Constants ─────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = {
    "role": "system",
    "content": "You are a smart, friendly and patient AI assistant with access to Google Drive, Gmail and GitHub tools.",
}
LOADING_HTML = """
<div style="
    display:flex;
    align-items:center;
    gap:6px;
">
    <div style="
        color:#6060a0;
        font-size:15.5px;
        line-height:1.75;
        font-style:italic;
    ">
        Working on it...
    </div>
    <div style="
        width:12px;
        height:12px;
        border:2px solid #2a2a40;
        border-top-color:#5c5cf0;
        border-radius:50%;
        animation:spin 0.6s linear infinite;
    "></div>
</div>
"""
FILE_ICONS = {
    "pdf": "📄", "png": "🖼️", "jpg": "🖼️", "jpeg": "🖼️",
    "txt": "📝", "docx": "📘", "xlsx": "📗", "csv": "📊",
}

def get_file_icon(name):
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    return FILE_ICONS.get(ext, "📎")

def clean_for_display(text):
    return re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL).strip()

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "user_email": None, "user_creds": None, "saved_email": None,
    "auth_url": None,
    # GitHub
    "github_token": None, "github_username": None,
    "github_auth_url": None, "github_connecting": False,
    "conversation_history": [SYSTEM_PROMPT],
    "messages": [], "uploaded_file_paths": [], "uploaded_file_names": [],
    "uploader_key": 0, "just_used_files": False,
    "working_key_index": 0, "working_key_date": str(date.today()),
    "do_clear": False, "do_logout": False,
    "do_github_connect": False, "do_github_disconnect": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if st.session_state.working_key_date != str(date.today()):
    st.session_state.working_key_index = 0
    st.session_state.working_key_date = str(date.today())

# ── Handle actions triggered from drawer ─────────────────────────────────────
if st.session_state.do_clear:
    st.session_state.do_clear = False
    st.session_state.messages = []
    st.session_state.conversation_history = [SYSTEM_PROMPT]
    st.session_state.uploaded_file_paths = []
    st.session_state.uploaded_file_names = []
    st.session_state.just_used_files = False
    st.session_state.uploader_key += 1
    st.session_state.working_key_index = 0
    st.session_state.working_key_date = str(date.today())

if st.session_state.do_logout:
    st.session_state.do_logout = False
    if st.session_state.user_email:
        delete_token(st.session_state.user_email)
        delete_github_token(st.session_state.user_email)
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

if st.session_state.do_github_connect:
    st.session_state.do_github_connect = False
    auth_url, state = get_github_auth_url()
    st.session_state.github_auth_url = auth_url
    st.session_state.github_connecting = True
    st.rerun()

if st.session_state.do_github_disconnect:
    st.session_state.do_github_disconnect = False
    if st.session_state.user_email:
        delete_github_token(st.session_state.user_email)
    st.session_state.github_token = None
    st.session_state.github_username = None
    st.session_state.github_auth_url = None
    st.session_state.github_connecting = False
    st.rerun()

# ── Restore sessions from disk FIRST — before any OAuth callback handling ────
# This must run before OAuth callbacks so user_email is available for GitHub handler
if st.session_state.user_email is None and st.session_state.saved_email:
    _creds = load_token(st.session_state.saved_email)
    if _creds:
        st.session_state.user_email = st.session_state.saved_email
        st.session_state.user_creds = _creds

# ── OAuth Callbacks ───────────────────────────────────────────────────────────
query_params = st.query_params

# ── GitHub OAuth callback — detected by 'github_' prefix in state param ──────
# NOTE: We use state prefix instead of session flag because Streamlit loses
# session state on full page redirects (GitHub OAuth does a real browser redirect).
# IMPORTANT: Capture and clear query params FIRST before any processing
# to prevent Google handler from seeing this code on any subsequent rerun.
if "code" in query_params and query_params.get("state", "").startswith("github_"):
    _gh_code  = query_params["code"]
    _gh_state = query_params["state"]
    st.query_params.clear()  # clear immediately before anything else
    try:
        with st.spinner("Connecting GitHub..."):
            token_data = exchange_github_code_for_token(_gh_state, _gh_code)
            username   = get_github_username(token_data)

            # Restore user email — try session first, then temp file saved before redirect
            if not st.session_state.user_email:
                _session_file = Path(__file__).parent / "tokens" / "github_session.txt"
                if _session_file.exists():
                    _saved_email = _session_file.read_text().strip()
                    if _saved_email:
                        _creds = load_token(_saved_email)
                        if _creds:
                            st.session_state.user_email  = _saved_email
                            st.session_state.saved_email = _saved_email
                            st.session_state.user_creds  = _creds
                try:
                    _session_file.unlink(missing_ok=True)
                except Exception:
                    pass

            if st.session_state.user_email:
                save_github_token(st.session_state.user_email, token_data)

            st.session_state.github_token     = token_data
            st.session_state.github_username  = username
            st.session_state.github_connecting = False
            st.session_state.github_auth_url  = None
            st.rerun()
    except Exception as e:
        st.error(f"GitHub login failed: {e}")
        st.session_state.github_connecting = False
        st.stop()

# ── Google OAuth callback ─────────────────────────────────────────────────────
elif "code" in query_params and st.session_state.user_email is None:
    try:
        with st.spinner("Logging you in..."):
            creds = exchange_code_for_token(
                query_params.get("state", ""), query_params["code"]
            )
            email = get_user_email(creds)
            if email == "unknown":
                st.error("Could not retrieve your email.")
                st.stop()
            save_token(email, creds)
            st.session_state.update(
                user_email=email, user_creds=creds,
                saved_email=email, auth_url=None
            )
            st.query_params.clear()
            st.rerun()
    except Exception as e:
        st.error(f"Login failed: {e}")
        st.session_state.auth_url = None
        st.stop()

# ── Restore sessions from disk ────────────────────────────────────────────────
# (already handled above before OAuth callbacks — this is a safety fallback)
if st.session_state.user_email is None and st.session_state.saved_email:
    _creds = load_token(st.session_state.saved_email)
    if _creds:
        st.session_state.user_email = st.session_state.saved_email
        st.session_state.user_creds = _creds

# Restore GitHub token from disk if logged in but github_token not in session
if st.session_state.user_email and st.session_state.github_token is None:
    saved_gh = load_github_token(st.session_state.user_email)
    if saved_gh:
        st.session_state.github_token = saved_gh
        st.session_state.github_username = get_github_username(saved_gh)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

html, body, [class*="css"] { 
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important; 
}

.stApp { background-color: #0e0e14 !important; }

#MainMenu, footer, header { visibility: hidden !important; }

button[data-testid="stSidebarCollapseButton"],
[data-testid="collapsedControl"],
section[data-testid="stSidebar"] { 
    display: none !important; 
}

/* ── Container ── */
.block-container {
    max-width: 760px !important;
    padding: 2rem 1rem 6rem 1rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: #1c1c2e !important; 
    color: #7878a0 !important;
    border: 1px solid #2a2a3e !important; 
    border-radius: 8px !important;
    font-size: 13px !important; 
    padding: 8px 16px !important;
    width: 100% !important; 
}
.stButton > button:hover {
    background-color: #22223a !important; 
    border-color: #5c5cf0 !important;
    color: #ddddf4 !important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 18px 0 !important;
    margin: 0 !important;
    box-shadow: none !important;
}

[data-testid="stChatMessage"] + [data-testid="stChatMessage"] {
    border-top: 1px solid #1c1c28 !important;
}

/* Avatar */
[data-testid="stChatMessage"] [data-testid="stChatMessageAvatarContainer"] {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    border-radius: 6px !important;
    flex-shrink: 0 !important;
    margin-top: 2px !important; /* 🔥 aligns with text */
}

/* 🔥 CRITICAL FIX — remove hidden spacing */
[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] {
    margin: 0 !important;
    padding: 0 !important;
}

/* 🔥 Remove internal paragraph offset */
[data-testid="stChatMessage"] div[data-testid="stMarkdownContainer"] p {
    margin-top: 0 !important;
}

/* Text styling */
[data-testid="stChatMessage"] p {
    color: #e0e0f0 !important;
    font-size: 15.5px !important;
    line-height: 1.75 !important;
    margin: 0 0 10px 0 !important;
}

[data-testid="stChatMessage"] p:last-child { 
    margin-bottom: 0 !important; 
}

/* ── Chat input ── */
[data-testid="stChatInput"] > div {
    background-color: #13131c !important;
    border: 1.5px solid #2a2a3e !important;
    border-radius: 14px !important;
    padding: 4px 8px !important;
    display: flex !important;
    align-items: center !important;
}

[data-testid="stChatInput"] textarea {
    color: #d0d0e8 !important;
    font-size: 15px !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    padding: 10px 8px !important;
    line-height: 1.5 !important;
}

/* Spinner animation */
@keyframes spin { 
    from { transform: rotate(0deg); } 
    to { transform: rotate(360deg); } 
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-thumb { background: #2a2a3e; border-radius: 4px; }

</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# LOGIN PAGE
# ══════════════════════════════════════════════════════════════════════════════
if st.session_state.user_email is None:
    if not st.session_state.auth_url:
        try:
            auth_url, state = get_auth_url()
            st.session_state.auth_url = auth_url
        except Exception as e:
            st.error(f"Failed to generate login URL: {e}")
            st.stop()
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style="max-width:400px;margin:0 auto;background:#13131c;
        border:1px solid #1e1e2e;border-radius:18px;padding:2.5rem 2rem;
        text-align:center;box-shadow:0 24px 64px rgba(0,0,0,0.6);">
        <div style="font-size:48px;margin-bottom:14px;">🤖</div>
        <div style="font-size:24px;font-weight:600;color:#eeeef8;letter-spacing:-0.5px;
            margin-bottom:10px;font-family:'Plus Jakarta Sans',sans-serif;">AI Assistant</div>
        <div style="font-size:13.5px;color:#5a5a78;line-height:1.75;margin-bottom:2rem;
            font-family:'Plus Jakarta Sans',sans-serif;">
            Manage your Gmail, Google Drive &amp; GitHub<br>intelligently — just by asking.
        </div>
        <a href="{st.session_state.auth_url}" target="_self" style="display:block;
            background:#5c5cf0;color:#fff;padding:13px 24px;border-radius:10px;
            font-size:14.5px;font-weight:500;text-decoration:none;
            font-family:'Plus Jakarta Sans',sans-serif;
            box-shadow:0 4px 24px rgba(92,92,240,0.45);">
            🔐 &nbsp; Sign in with Google
        </a>
        <div style="font-size:11.5px;color:#33334a;margin-top:1.25rem;
            font-family:'Plus Jakarta Sans',sans-serif;">
            🔒 &nbsp; We only access what you explicitly allow
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# DRAWER — injected into parent document via components.html
# ══════════════════════════════════════════════════════════════════════════════
email = st.session_state.user_email or ""
initial = email[0].upper() if email else "?"
github_username = st.session_state.github_username or ""
github_connected = bool(st.session_state.github_token)

# GitHub connect — direct redirect to GitHub OAuth
# We save the user email to a temp file on disk before redirecting
# so we can restore it after GitHub sends us back (session is wiped on redirect)
if st.session_state.github_connecting and st.session_state.github_auth_url:
    # Save current user email to disk so we can restore after redirect
    if st.session_state.user_email:
        _session_file = Path(__file__).parent / "tokens" / "github_session.txt"
        _session_file.write_text(st.session_state.user_email)
    auth_url = st.session_state.github_auth_url
    st.markdown(
        f'<meta http-equiv="refresh" content="0;url={auth_url}">',
        unsafe_allow_html=True
    )
    st.stop()

components.html(f"""
<!DOCTYPE html>
<html>
<head>
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
</style>
</head>
<body>
<script>
(function() {{
  var p = window.parent.document;

  // ───────────────────────── Styles ─────────────────────────
  if (!p.getElementById('drawer-styles')) {{
    var style = p.createElement('style');
    style.id = 'drawer-styles';
    style.textContent = `
      #st-drawer-overlay {{
        display:none; position:fixed; inset:0;
        background:rgba(0,0,0,0.55); z-index:9998;
      }}
      #st-drawer-overlay.open {{ display:block; }}

      #st-drawer {{
        position:fixed; top:0; left:0; bottom:0; width:300px;
        background:#111120; border-right:1px solid #1e1e30;
        box-shadow:8px 0 40px rgba(0,0,0,0.6);
        z-index:9999;
        transform:translateX(-100%);
        transition:transform 0.28s ease;
        display:flex; flex-direction:column;
        font-family:'Plus Jakarta Sans',sans-serif;
      }}
      #st-drawer.open {{ transform:translateX(0); }}

      #st-drawer-header {{
        display:flex; justify-content:space-between;
        padding:18px 16px; border-bottom:1px solid #1e1e30;
      }}

      #st-drawer-close {{
        width:32px; height:32px;
        background:#1c1c2e;
        border:1px solid #2a2a3e;
        border-radius:8px;
        cursor:pointer;
        display:flex; align-items:center; justify-content:center;
        color:#7878a0;
      }}

      #st-drawer-body {{
        flex:1; overflow-y:auto; padding:16px;
      }}

      .st-drawer-btn {{
        width:100%;
        background:#1c1c2e;
        border:1px solid #2a2a3e;
        border-radius:8px;
        padding:9px 14px;
        margin-bottom:8px;
        cursor:pointer;
        color:#7878a0;
        font-size:13px;
        text-align:left;
      }}

      #st-hamburger {{
        position:fixed; top:14px; left:14px; z-index:9997;
        width:36px; height:36px;
        background:#1c1c2e;
        border:1px solid #2a2a3e;
        border-radius:8px;
        cursor:pointer;
        display:flex; align-items:center; justify-content:center;
      }}
    `;
    p.head.appendChild(style);
  }}

  // ───────────────────────── Hamburger ─────────────────────────
  if (!p.getElementById('st-hamburger')) {{
    var btn = p.createElement('div');
    btn.id = 'st-hamburger';
    btn.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24"><path d="M3 18h18v-2H3v2zm0-5h18v-2H3v2zm0-7v2h18V6H3z"/></svg>';
    btn.onclick = openDrawer;
    p.body.appendChild(btn);
  }}

  // ───────────────────────── Overlay ─────────────────────────
  if (!p.getElementById('st-drawer-overlay')) {{
    var overlay = p.createElement('div');
    overlay.id = 'st-drawer-overlay';
    overlay.onclick = closeDrawer;
    p.body.appendChild(overlay);
  }}

  // ───────────────────────── Drawer ─────────────────────────
  if (!p.getElementById('st-drawer')) {{
    var drawer = p.createElement('div');
    drawer.id = 'st-drawer';

    drawer.innerHTML = `
        <div id="st-drawer-header">
            <div style="font-size:16px;font-weight:600;color:#dddde8;">
            🤖 AI Assistant
            </div>
            <div id="st-drawer-close">✕</div>
        </div>

        <div id="st-drawer-body" style="display:flex;flex-direction:column;height:100%;">

            <!-- Top Content -->
            <div>
            <!-- User Card -->
            <div style="
                display:flex;align-items:center;gap:12px;
                background:#1c1c2c;
                border:1px solid #252538;
                border-radius:12px;
                padding:12px;
                margin-bottom:12px;
            ">
                <div style="
                width:36px;height:36px;border-radius:50%;
                background:#2a2a44;
                display:flex;align-items:center;justify-content:center;
                font-weight:600;color:#8888ee;">
                {initial}
                </div>

                <div style="flex:1;min-width:0;">
                <div style="font-size:13px;color:#d0d0e8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                    {email}
                </div>
                <div style="font-size:11px;color:#34c77a;">
                    ● Google Connected
                </div>
                </div>
            </div>

            <!-- GitHub Connection Card -->
            <div style="
                background:#1c1c2c;
                border:1px solid #252538;
                border-radius:12px;
                padding:12px;
                margin-bottom:18px;
            ">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:{'8px' if not github_connected else '0'};">
                <div style="font-size:18px;">🐙</div>
                <div style="flex:1;min-width:0;">
                    <div style="font-size:13px;color:#d0d0e8;">GitHub</div>
                    <div style="font-size:11px;color:{'#34c77a' if github_connected else '#ff8080'};">
                    {'● Connected as @' + github_username if github_connected else '○ Not connected'}
                    </div>
                </div>
                </div>
                {'<button class="st-drawer-btn" id="btn-github-disconnect" style="background:#2a1c1c;border:1px solid #3a2a2a;color:#ff8080;margin-bottom:0;">🔌 Disconnect GitHub</button>' if github_connected else '<button class="st-drawer-btn" id="btn-github-connect" style="background:#1a2a1a;border:1px solid #2a3a2a;color:#34c77a;margin-bottom:0;">🔗 Connect GitHub</button>'}
            </div>

            <!-- Section: Capabilities -->
            <div style="font-size:11px;color:#5a5a80;margin-bottom:10px;">
                CAPABILITIES
            </div>

            <div style="font-size:13px;color:#8080b0;line-height:2;">
                📧 Read, send, reply<br>
                📝 Manage drafts<br>
                📅 Schedule emails<br>
                🔍 Search emails<br>
                📁 Manage Drive files<br>
                📤 Upload files<br>
                🔔 Mark read/unread<br>
                🐙 GitHub repos &amp; issues
            </div>

            </div>

            <!-- Bottom Actions -->
            <div style="margin-top:auto;">

            <div style="height:1px;background:#1e1e2e;margin:12px 0;"></div>

            <button class="st-drawer-btn" id="btn-clear" style="
                background:#1a1a28;
                border:1px solid #2a2a3e;
            ">
                🗑️ Clear Chat
            </button>

            <button class="st-drawer-btn" id="btn-logout" style="
                background:#2a1c1c;
                border:1px solid #3a2a2a;
                color:#ff8080;
            ">
                🚪 Logout
            </button>

            </div>

        </div>
        `;

    p.body.appendChild(drawer);

    p.getElementById('st-drawer-close').onclick = closeDrawer;

    // Clear
    p.getElementById('btn-clear').onclick = function() {{
      closeDrawer();
      var btns = p.querySelectorAll('button[kind="secondary"]');
      btns.forEach(b => {{ if (b.innerText.includes('Clear')) b.click(); }});
    }};

    // Logout
    p.getElementById('btn-logout').onclick = function() {{
      closeDrawer();
      var btns = p.querySelectorAll('button[kind="secondary"]');
      btns.forEach(b => {{ if (b.innerText.includes('Logout')) b.click(); }});
    }};

    var ghConnect = p.getElementById('btn-github-connect');
    if (ghConnect) {{
      ghConnect.onclick = function() {{
        closeDrawer();
        var btns = p.querySelectorAll('button[kind="secondary"]');
        btns.forEach(b => {{ if (b.innerText.includes('GitHub Connect')) b.click(); }});
      }};
    }}

    var ghDisconnect = p.getElementById('btn-github-disconnect');
    if (ghDisconnect) {{
      ghDisconnect.onclick = function() {{
        closeDrawer();
        var btns = p.querySelectorAll('button[kind="secondary"]');
        btns.forEach(b => {{ if (b.innerText.includes('GitHub Disconnect')) b.click(); }});
      }};
    }}
  }}

  function openDrawer() {{
    p.getElementById('st-drawer').classList.add('open');
    p.getElementById('st-drawer-overlay').classList.add('open');
  }}

  function closeDrawer() {{
    p.getElementById('st-drawer').classList.remove('open');
    p.getElementById('st-drawer-overlay').classList.remove('open');
  }}

  // ───────────────────────── FINAL ALIGNMENT FIX ─────────────────────────
  function fixMessageAlignment() {{
    var msgs = p.querySelectorAll('[data-testid="stChatMessage"]');

    msgs.forEach(function(msg) {{
      var row = msg.firstElementChild;
      if (!row) return;

      row.style.display = "flex";
      row.style.alignItems = "center";
      row.style.gap = "12px";

      var avatar = row.querySelector('[data-testid="stChatMessageAvatarContainer"]');
      if (avatar) {{
        avatar.style.alignSelf = "center";
        avatar.style.marginTop = "0";
      }}

      var content = row.querySelector('[data-testid="stMarkdownContainer"]');
      if (content) {{
        content.style.margin = "0";
        content.style.padding = "0";
        content.style.display = "flex";
        content.style.flexDirection = "column";
        content.style.justifyContent = "center";
      }}
    }});
  }}

  setTimeout(fixMessageAlignment, 300);
  setTimeout(fixMessageAlignment, 800);

  new MutationObserver(fixMessageAlignment)
    .observe(p.body, {{ childList: true, subtree: true }});

}})();
</script>
</body>
</html>
""", height=0)


# Hidden Streamlit buttons wired to drawer actions
col_h1, col_h2, col_h3, col_h4 = st.columns(4)
with col_h1:
    if st.button("🗑️ Clear Chat", key="hidden_clear"):
        st.session_state.messages = []
        st.session_state.conversation_history = [SYSTEM_PROMPT]
        st.session_state.uploaded_file_paths = []
        st.session_state.uploaded_file_names = []
        st.session_state.just_used_files = False
        st.session_state.uploader_key += 1
        st.session_state.working_key_index = 0
        st.session_state.working_key_date = str(date.today())
        st.rerun()
with col_h2:
    if st.button("🚪 Logout", key="hidden_logout"):
        delete_token(st.session_state.user_email)
        delete_github_token(st.session_state.user_email)
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
with col_h3:
    if st.button("🔗 GitHub Connect", key="hidden_github_connect"):
        st.session_state.do_github_connect = True
        st.rerun()
with col_h4:
    if st.button("🔌 GitHub Disconnect", key="hidden_github_disconnect"):
        st.session_state.do_github_disconnect = True
        st.rerun()

# Hide these buttons visually
st.markdown("""
<style>
div[data-testid="stHorizontalBlock"]:first-of-type {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
    height: 0 !important;
    overflow: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHAT
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center;padding:0.5rem 0 1.5rem;">
    <div style="font-size:28px;font-weight:700;color:#dddde8;
        font-family:'Plus Jakarta Sans',sans-serif;letter-spacing:-0.5px;">
        🤖 AI Assistant
    </div>
    <div style="font-size:13px;color:#44445a;margin-top:4px;
        font-family:'Plus Jakarta Sans',sans-serif;">
        Manage your Gmail, Google Drive and GitHub — just ask!
    </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.uploaded_file_names:
    names = " · ".join(
        f"{get_file_icon(n)} {n}" for n in st.session_state.uploaded_file_names
    )
    st.info(f"📎 **{names}** — ready to use")

# ── File uploader in sidebar-like expander ────────────────────────────────────
with st.expander("📎 Attach files", expanded=False):
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "png", "jpg", "jpeg", "txt", "docx", "xlsx", "csv"],
        label_visibility="collapsed",
        accept_multiple_files=True,
        key=f"file_uploader_{st.session_state.uploader_key}",
    )
    if uploaded_files and not st.session_state.just_used_files:
        st.session_state.uploaded_file_paths = []
        st.session_state.uploaded_file_names = []
        for uf in uploaded_files:
            tmp_dir = tempfile.mkdtemp()
            tmp_path = os.path.join(tmp_dir, uf.name)
            with open(tmp_path, "wb") as f:
                f.write(uf.getbuffer())
            st.session_state.uploaded_file_paths.append(tmp_path)
            st.session_state.uploaded_file_names.append(uf.name)
        st.success(f"✅ {len(uploaded_files)} file(s) ready!")
        st.caption("Now type your message!")
    elif st.session_state.just_used_files:
        st.session_state.just_used_files = False
        st.session_state.uploaded_file_paths = []
        st.session_state.uploaded_file_names = []

st.divider()

# ── Chat messages ─────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(clean_for_display(msg["content"]))

user_input = st.chat_input("Ask me anything...")

if user_input:
    full_input = user_input
    if st.session_state.uploaded_file_paths:
        files_str = " ".join(
            f"[FILE: {p}]" for p in st.session_state.uploaded_file_paths
        )
        full_input = f"{user_input} {files_str}"

    with st.chat_message("user"):
        if st.session_state.uploaded_file_names:
            st.markdown(
                f"{user_input} 📎 {', '.join(st.session_state.uploaded_file_names)}"
            )
        else:
            st.markdown(user_input)

    st.session_state.messages.append({
        "role": "user",
        "content": (
            f"{user_input} 📎 {', '.join(st.session_state.uploaded_file_names)}"
            if st.session_state.uploaded_file_names else user_input
        ),
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
                st.session_state.github_token,
            ):
                if token is not None:
                    full_reply.append(token)
                    reply_box.markdown(clean_for_display("".join(full_reply)))
                if hist is not None:
                    updated_history = hist
                if key is not None:
                    working_key = key
        except Exception:
            error_msg = "Something went wrong. Please try again."
            reply_box.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
        else:
            if updated_history:
                st.session_state.conversation_history = updated_history
            st.session_state.working_key_index = working_key
            final_reply = "".join(full_reply)
            if final_reply and final_reply.strip():
                st.session_state.messages.append({"role": "assistant", "content": final_reply})

    if st.session_state.uploaded_file_paths:
        st.session_state.just_used_files = True
        st.session_state.uploader_key += 1
        st.rerun()