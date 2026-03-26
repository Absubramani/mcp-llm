import os
import pickle
import hashlib
import json
import requests
from pathlib import Path
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build

BASE_DIR = Path(__file__).parent.parent
CLIENT_SECRET_FILE = BASE_DIR / "oauth-client.json"
TOKENS_DIR = BASE_DIR / "tokens"
TOKENS_DIR.mkdir(exist_ok=True)

FLOW_STATE_FILE = TOKENS_DIR / "flow_state.json"

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/gmail.modify",
    "https://mail.google.com/",
    "https://www.googleapis.com/auth/userinfo.email",
    "openid",
]

REDIRECT_URI = "http://localhost:8501"

# ── GitHub OAuth Config ───────────────────────────────────────────────────────
GITHUB_CLIENT_ID     = os.getenv("GITHUB_CLIENT_ID", "")
GITHUB_CLIENT_SECRET = os.getenv("GITHUB_CLIENT_SECRET", "")
GITHUB_SCOPES        = "repo user"
GITHUB_REDIRECT_URI  = "http://localhost:8501"


# ══════════════════════════════════════════════════════════════════════════════
# GOOGLE AUTH
# ══════════════════════════════════════════════════════════════════════════════

def get_user_token_path(user_id: str) -> Path:
    """Get Google token file path for a specific user."""
    safe_id = hashlib.md5(user_id.encode()).hexdigest()
    return TOKENS_DIR / f"{safe_id}.pickle"


def get_auth_url() -> tuple[str, str]:
    """Generate Google OAuth URL — returns auth_url and state."""
    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_FILE),
        scopes=SCOPES,
        redirect_uri=REDIRECT_URI
    )
    auth_url, state = flow.authorization_url(
        access_type="offline",
        include_granted_scopes="true",
        prompt="consent"
    )
    with open(FLOW_STATE_FILE, "w") as f:
        json.dump({"state": state}, f)

    return auth_url, state


def restore_flow(state: str) -> Flow:
    """Restore Google OAuth flow from saved state."""
    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_FILE),
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI
    )
    return flow


def exchange_code_for_token(state: str, code: str):
    """Exchange Google authorization code for credentials."""
    try:
        flow = restore_flow(state)
        flow.fetch_token(code=code)
        return flow.credentials
    except Exception as e:
        raise RuntimeError(f"Failed to exchange token: {e}")


def get_user_email(creds) -> str:
    """Get user email from Google credentials."""
    try:
        service = build("oauth2", "v2", credentials=creds)
        user_info = service.userinfo().get().execute()
        email = user_info.get("email", "")
        if email:
            return email
    except Exception:
        pass

    try:
        if hasattr(creds, "id_token") and creds.id_token:
            import jwt
            decoded = jwt.decode(
                creds.id_token,
                options={"verify_signature": False}
            )
            email = decoded.get("email", "")
            if email:
                return email
    except Exception:
        pass

    return "unknown"


def save_token(user_id: str, creds) -> None:
    """Save Google token to disk."""
    token_path = get_user_token_path(user_id)
    with open(token_path, "wb") as f:
        pickle.dump(creds, f)


def load_token(user_id: str):
    """Load and refresh Google token from disk."""
    token_path = get_user_token_path(user_id)
    if not token_path.exists():
        return None

    try:
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
    except Exception:
        token_path.unlink(missing_ok=True)
        return None

    if not creds:
        return None

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_token(user_id, creds)
        except Exception:
            token_path.unlink(missing_ok=True)
            return None

    return creds if creds.valid else None


def delete_token(user_id: str) -> None:
    """Delete Google token — logout."""
    token_path = get_user_token_path(user_id)
    token_path.unlink(missing_ok=True)

    if FLOW_STATE_FILE.exists():
        try:
            FLOW_STATE_FILE.unlink()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# GITHUB AUTH
# ══════════════════════════════════════════════════════════════════════════════

def get_github_token_path(user_id: str) -> Path:
    """Get GitHub token file path for a specific user."""
    safe_id = hashlib.md5(user_id.encode()).hexdigest()
    return TOKENS_DIR / f"github_{safe_id}.json"


def get_github_auth_url() -> tuple[str, str]:
    """Generate GitHub OAuth URL — returns auth_url and state."""
    import secrets
    state = "github_" + secrets.token_urlsafe(16)

    # Save state to disk — survives Streamlit page reload
    github_state_file = TOKENS_DIR / "github_flow_state.json"
    with open(github_state_file, "w") as f:
        json.dump({"state": state}, f)

    params = {
        "client_id":    GITHUB_CLIENT_ID,
        "redirect_uri": GITHUB_REDIRECT_URI,
        "scope":        GITHUB_SCOPES,
        "state":        state,
    }
    query = "&".join(f"{k}={v}" for k, v in params.items())
    auth_url = f"https://github.com/login/oauth/authorize?{query}"
    return auth_url, state


def exchange_github_code_for_token(state: str, code: str) -> dict:
    """Exchange GitHub authorization code for access token."""
    github_state_file = TOKENS_DIR / "github_flow_state.json"

    # Verify state matches
    if github_state_file.exists():
        with open(github_state_file) as f:
            saved = json.load(f)
        if saved.get("state") != state:
            raise RuntimeError("GitHub OAuth state mismatch — possible CSRF attack.")

    response = requests.post(
        "https://github.com/login/oauth/access_token",
        headers={"Accept": "application/json"},
        data={
            "client_id":     GITHUB_CLIENT_ID,
            "client_secret": GITHUB_CLIENT_SECRET,
            "code":          code,
            "redirect_uri":  GITHUB_REDIRECT_URI,
        },
        timeout=10,
    )
    data = response.json()

    if "access_token" not in data:
        raise RuntimeError(f"GitHub token exchange failed: {data.get('error_description', data)}")

    # Clean up state file
    github_state_file.unlink(missing_ok=True)
    return data


def get_github_username(token_data: dict) -> str:
    """Get GitHub username from token."""
    access_token = token_data.get("access_token", "")
    response = requests.get(
        "https://api.github.com/user",
        headers={
            "Authorization": f"Bearer {access_token}",
            "Accept":        "application/vnd.github+json",
        },
        timeout=10,
    )
    if response.status_code == 200:
        return response.json().get("login", "unknown")
    return "unknown"


def save_github_token(user_id: str, token_data: dict) -> None:
    """Save GitHub token to disk."""
    token_path = get_github_token_path(user_id)
    with open(token_path, "w") as f:
        json.dump(token_data, f)


def load_github_token(user_id: str) -> dict | None:
    """Load GitHub token from disk."""
    token_path = get_github_token_path(user_id)
    if not token_path.exists():
        return None
    try:
        with open(token_path) as f:
            return json.load(f)
    except Exception:
        token_path.unlink(missing_ok=True)
        return None


def delete_github_token(user_id: str) -> None:
    """Delete GitHub token — disconnect GitHub."""
    token_path = get_github_token_path(user_id)
    token_path.unlink(missing_ok=True)