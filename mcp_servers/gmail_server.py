import os
import pickle
import base64
import re
import threading
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from mcp.server.fastmcp import FastMCP
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# ================= CONFIG =================

BASE_DIR = Path(__file__).parent.parent
CLIENT_SECRET_FILE = BASE_DIR / "oauth-client.json"
TOKEN_FILE = BASE_DIR / "token.pickle"
SCOPES = [
    "https://www.googleapis.com/auth/gmail.modify",
    "https://www.googleapis.com/auth/drive",
    "https://mail.google.com/",
]

mcp = FastMCP("Gmail MCP")

# ================= AUTH =================

_local = threading.local()


def get_service():
    """Get gmail service for current request — lazy init per thread."""
    if not hasattr(_local, "gmail_service") or _local.gmail_service is None:
        _local.gmail_service = get_gmail_service()
    return _local.gmail_service


def get_gmail_service(creds=None):
    """Build and return a Gmail service using best available credentials."""
    # 1. Use directly passed creds
    if creds:
        return build("gmail", "v1", credentials=creds)

    # 2. Use creds from temp file passed by tool_executor
    creds_file = os.environ.get("MCP_CREDS_FILE")
    if creds_file and Path(creds_file).exists():
        try:
            with open(creds_file, "rb") as f:
                user_creds = pickle.load(f)
            if user_creds and user_creds.valid:
                return build("gmail", "v1", credentials=user_creds)
            # Try refresh if expired
            if user_creds and user_creds.expired and user_creds.refresh_token:
                user_creds.refresh(Request())
                return build("gmail", "v1", credentials=user_creds)
        except Exception:
            pass

    # 3. Fallback to local token.pickle for CLI/testing
    local_creds = None
    if TOKEN_FILE.exists():
        try:
            with open(TOKEN_FILE, "rb") as f:
                local_creds = pickle.load(f)
        except Exception:
            TOKEN_FILE.unlink(missing_ok=True)

    if local_creds and local_creds.expired and local_creds.refresh_token:
        try:
            local_creds.refresh(Request())
            with open(TOKEN_FILE, "wb") as f:
                pickle.dump(local_creds, f)
        except Exception:
            TOKEN_FILE.unlink(missing_ok=True)
            local_creds = None

    if not local_creds or not local_creds.valid:
        flow = InstalledAppFlow.from_client_secrets_file(
            str(CLIENT_SECRET_FILE), SCOPES
        )
        local_creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "wb") as f:
            pickle.dump(local_creds, f)

    return build("gmail", "v1", credentials=local_creds)


# ================= HELPERS =================

def build_message(to: str, subject: str, body: str, thread_id: str = None) -> dict:
    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    message.attach(MIMEText(body, "plain"))

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    payload = {"raw": raw}

    if thread_id:
        payload["threadId"] = thread_id

    return payload


def parse_headers(headers: list, key: str) -> str:
    for h in headers:
        if h["name"].lower() == key.lower():
            return h["value"]
    return ""


def get_email_body(payload: dict) -> str:
    """Extract clean plain text body from email payload."""
    # Handle multipart emails
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                if data:
                    return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
        # Fallback to first part if no plain text
        for part in payload["parts"]:
            data = part.get("body", {}).get("data", "")
            if data:
                text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                text = re.sub(r'<[^>]+>', ' ', text)
                text = re.sub(r'\s+', ' ', text).strip()
                return text
    else:
        data = payload.get("body", {}).get("data", "")
        if data:
            text = base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    return "No body content."


# ================= TOOLS =================

@mcp.tool()
def list_emails(max_results: str = "10", query: str = "", num: str = "") -> list:
    """
    List emails from Gmail inbox. Use max_results to specify count.
    Example: max_results='5' for last 5 emails.
    Optionally filter with query like 'from:someone@gmail.com' or 'subject:hello'.
    """
    gmail_service = get_service()
    try:
        count = int(num) if num else int(max_results) if max_results else 10
        q = "in:inbox " + query if query else "in:inbox"

        results = gmail_service.users().messages().list(
            userId="me", maxResults=count, q=q
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return [{"message": "No emails found."}]

        emails = []
        for msg in messages:
            detail = gmail_service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()

            headers = detail.get("payload", {}).get("headers", [])
            emails.append({
                "id": msg["id"],
                "thread_id": detail.get("threadId"),
                "from": parse_headers(headers, "From"),
                "subject": parse_headers(headers, "Subject"),
                "date": parse_headers(headers, "Date"),
                "snippet": detail.get("snippet", ""),
            })

        return emails
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def read_email(email_id: str) -> dict:
    """
    Read the full content of an email by its ID.
    Use the exact id string from list_emails results.
    Example: email_id='19c8a47e66de3a50'
    """
    gmail_service = get_service()
    try:
        msg = gmail_service.users().messages().get(
            userId="me", id=email_id, format="full"
        ).execute()

        headers = msg.get("payload", {}).get("headers", [])
        body = get_email_body(msg.get("payload", {}))

        # Truncate very long emails to save LLM tokens
        if len(body) > 3000:
            body = body[:3000]

        return {
            "id": email_id,
            "from": parse_headers(headers, "From"),
            "to": parse_headers(headers, "To"),
            "subject": parse_headers(headers, "Subject"),
            "date": parse_headers(headers, "Date"),
            "body": body,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def send_email(to: str, subject: str, body: str) -> dict:
    """
    Send an email.
    Example: to='someone@gmail.com', subject='Hello', body='Hi there!'
    """
    gmail_service = get_service()
    try:
        payload = build_message(to, subject, body)
        sent = gmail_service.users().messages().send(
            userId="me", body=payload
        ).execute()
        return {
            "status": "success",
            "message": f"Email sent to {to} successfully.",
            "id": sent["id"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def reply_to_email(email_id: str, body: str) -> dict:
    """
    Reply to an existing email using its id from list_emails results.
    Example: email_id='19c8a47e66de3a50', body='Thanks for your email!'
    """
    gmail_service = get_service()
    try:
        original = gmail_service.users().messages().get(
            userId="me", id=email_id, format="metadata",
            metadataHeaders=["From", "Subject"]
        ).execute()

        headers = original.get("payload", {}).get("headers", [])
        to = parse_headers(headers, "From")
        subject = parse_headers(headers, "Subject")
        if not subject.startswith("Re:"):
            subject = "Re: " + subject
        thread_id = original.get("threadId")

        payload = build_message(to, subject, body, thread_id=thread_id)
        sent = gmail_service.users().messages().send(
            userId="me", body=payload
        ).execute()

        return {
            "status": "success",
            "message": f"Reply sent to {to} successfully.",
            "id": sent["id"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def delete_email(email_id: str) -> dict:
    """
    Move an email to trash using its id from list_emails results.
    Example: email_id='19c8a47e66de3a50'
    """
    gmail_service = get_service()
    try:
        gmail_service.users().messages().trash(
            userId="me", id=email_id
        ).execute()
        return {
            "status": "success",
            "message": "Email moved to trash successfully."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def restore_email(email_id: str) -> dict:
    """
    Restore an email from trash using its id.
    Example: email_id='19c8a47e66de3a50'
    """
    gmail_service = get_service()
    try:
        gmail_service.users().messages().untrash(
            userId="me", id=email_id
        ).execute()
        return {
            "status": "success",
            "message": "Email restored from trash successfully."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def search_emails(query: str, max_results: str = "10") -> list:
    """
    Search emails using a query string.
    Examples: query='from:boss@company.com', query='subject:invoice', query='after:2024/01/01'
    """
    gmail_service = get_service()
    try:
        count = int(max_results) if max_results else 10
        results = gmail_service.users().messages().list(
            userId="me", maxResults=count, q=query
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return [{"message": f"No emails found for: {query}"}]

        emails = []
        for msg in messages:
            detail = gmail_service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()

            headers = detail.get("payload", {}).get("headers", [])
            emails.append({
                "id": msg["id"],
                "from": parse_headers(headers, "From"),
                "subject": parse_headers(headers, "Subject"),
                "date": parse_headers(headers, "Date"),
                "snippet": detail.get("snippet", ""),
            })

        return emails
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def send_email_with_attachment(to: str, subject: str, body: str, file_path: str, extra_file_paths: str = "") -> dict:
    """
    Send an email with one or more file attachments.
    to: recipient email address.
    subject: email subject.
    body: email body text.
    file_path: full local path of the first file to attach.
    extra_file_paths: comma separated paths of additional files (optional).
    """
    import mimetypes
    from email.mime.base import MIMEBase
    from email import encoders
    gmail_service = get_service()

    try:
        all_files = [file_path]
        if extra_file_paths:
            for p in extra_file_paths.split(","):
                p = p.strip()
                if p:
                    all_files.append(p)

        for fp in all_files:
            if not os.path.exists(fp):
                return {"status": "error", "message": f"File not found: {fp}"}

        msg = MIMEMultipart()
        msg["to"] = to
        msg["subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        attached_names = []
        for fp in all_files:
            file_name = os.path.basename(fp)
            mime_type, _ = mimetypes.guess_type(fp)
            if not mime_type:
                mime_type = "application/octet-stream"
            main_type, sub_type = mime_type.split("/", 1)

            with open(fp, "rb") as f:
                attachment = MIMEBase(main_type, sub_type)
                attachment.set_payload(f.read())
                encoders.encode_base64(attachment)
                attachment.add_header(
                    "Content-Disposition",
                    f"attachment; filename={file_name}"
                )
                msg.attach(attachment)
                attached_names.append(file_name)

        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = gmail_service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()

        return {
            "status": "success",
            "message": f"Email sent to {to} with attachments: {', '.join(attached_names)}",
            "id": sent.get("id")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()