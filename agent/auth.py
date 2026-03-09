import os
import pickle
import hashlib
import json
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


def get_user_token_path(user_id: str) -> Path:
    """Get token file path for a specific user."""
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
    # Save state to disk — survives Streamlit page reload
    with open(FLOW_STATE_FILE, "w") as f:
        json.dump({"state": state}, f)

    return auth_url, state


def restore_flow(state: str) -> Flow:
    """Restore OAuth flow from saved state."""
    flow = Flow.from_client_secrets_file(
        str(CLIENT_SECRET_FILE),
        scopes=SCOPES,
        state=state,
        redirect_uri=REDIRECT_URI
    )
    return flow


def exchange_code_for_token(state: str, code: str):
    """Exchange authorization code for credentials."""
    try:
        flow = restore_flow(state)
        flow.fetch_token(code=code)
        return flow.credentials
    except Exception as e:
        raise RuntimeError(f"Failed to exchange token: {e}")


def get_user_email(creds) -> str:
    """Get user email from credentials."""
    # Try userinfo API first
    try:
        service = build("oauth2", "v2", credentials=creds)
        user_info = service.userinfo().get().execute()
        email = user_info.get("email", "")
        if email:
            return email
    except Exception:
        pass

    # Fallback — decode id_token without verification
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
    """Save user token to disk."""
    token_path = get_user_token_path(user_id)
    with open(token_path, "wb") as f:
        pickle.dump(creds, f)


def load_token(user_id: str):
    """Load and refresh user token from disk."""
    token_path = get_user_token_path(user_id)
    if not token_path.exists():
        return None

    try:
        with open(token_path, "rb") as f:
            creds = pickle.load(f)
    except Exception:
        # Corrupted token file — delete and return None
        token_path.unlink(missing_ok=True)
        return None

    if not creds:
        return None

    if creds.expired and creds.refresh_token:
        try:
            creds.refresh(Request())
            save_token(user_id, creds)
        except Exception:
            # Refresh failed — token invalid, delete it
            token_path.unlink(missing_ok=True)
            return None

    return creds if creds.valid else None


def delete_token(user_id: str) -> None:
    """Delete user token — logout."""
    token_path = get_user_token_path(user_id)
    token_path.unlink(missing_ok=True)

    # Also clean up flow state
    if FLOW_STATE_FILE.exists():
        try:
            FLOW_STATE_FILE.unlink()
        except Exception:
            pass