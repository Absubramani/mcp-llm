import os
import pickle
import base64
import re
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

# ================= AUTH =================

creds = None
if TOKEN_FILE.exists():
    with open(TOKEN_FILE, "rb") as token:
        creds = pickle.load(token)

if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRET_FILE), SCOPES)
        creds = flow.run_local_server(port=0)
    with open(TOKEN_FILE, "wb") as token:
        pickle.dump(creds, token)

gmail_service = build("gmail", "v1", credentials=creds)
mcp = FastMCP("Gmail MCP")

# ================= HELPERS =================

def build_message(to: str, subject: str, body: str, reply_to_id: str = None) -> dict:
    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    message.attach(MIMEText(body, "plain"))

    raw = base64.urlsafe_b64encode(message.as_bytes()).decode()
    payload = {"raw": raw}

    if reply_to_id:
        payload["threadId"] = reply_to_id

    return payload


def parse_headers(headers: list, key: str) -> str:
    for h in headers:
        if h["name"].lower() == key.lower():
            return h["value"]
    return ""


def get_email_body(payload: dict) -> str:
    if "parts" in payload:
        for part in payload["parts"]:
            if part["mimeType"] == "text/plain":
                data = part["body"].get("data", "")
                return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
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
def list_emails(max_results: int = 10, query: str = "", num: int = None) -> list:
    """
    List emails from Gmail inbox. Use max_results parameter to specify count.
    Example: max_results=5 for last 5 emails.
    Optionally filter with query like 'from:someone@gmail.com' or 'subject:hello'.
    """
    # Handle model using wrong parameter name
    if num is not None:
        max_results = num

    q = "in:inbox " + query if query else "in:inbox"
    results = gmail_service.users().messages().list(
        userId="me", maxResults=max_results, q=q
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

@mcp.tool()
def read_email(email_id: str) -> dict:
    """
    Read the full content of an email by its ID.
    Use email_id parameter with the ID from list_emails results.
    Example: email_id='19c8a47e66de3a50'
    """
    try:
        msg = gmail_service.users().messages().get(
            userId="me", id=email_id, format="full"
        ).execute()

        headers = msg.get("payload", {}).get("headers", [])
        body = get_email_body(msg.get("payload", {}))

        # Truncate long emails to save tokens
        if len(body) > 2000:
            body = body[:2000] + "\n\n[Email truncated — showing first 2000 characters]"

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
    Send an email. Use to, subject, body parameters.
    Example: to='someone@gmail.com', subject='Hello', body='Hi there!'
    """
    try:
        payload = build_message(to, subject, body)
        sent = gmail_service.users().messages().send(
            userId="me", body=payload
        ).execute()

        return {
            "status": "success",
            "message": f"Email sent to {to}",
            "id": sent["id"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def reply_to_email(email_id: str, body: str) -> dict:
    """
    Reply to an existing email using email_id from list_emails results.
    Example: email_id='19c8a47e66de3a50', body='Thanks for your email!'
    """
    try:
        original = gmail_service.users().messages().get(
            userId="me", id=email_id, format="metadata",
            metadataHeaders=["From", "Subject", "Date"]
        ).execute()

        headers = original.get("payload", {}).get("headers", [])
        to = parse_headers(headers, "From")
        subject = "Re: " + parse_headers(headers, "Subject")
        thread_id = original.get("threadId")

        payload = build_message(to, subject, body, reply_to_id=thread_id)
        sent = gmail_service.users().messages().send(
            userId="me", body=payload
        ).execute()

        return {
            "status": "success",
            "message": f"Reply sent to {to}",
            "id": sent["id"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def delete_email(email_id: str) -> dict:
    """
    Delete an email permanently using email_id from list_emails results.
    Example: email_id='19c8a47e66de3a50'
    """
    try:
        gmail_service.users().messages().delete(
            userId="me", id=email_id
        ).execute()

        return {
            "status": "success",
            "message": f"Email {email_id} deleted successfully."
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def search_emails(query: str, max_results: int = 10) -> list:
    """
    Search emails using query. Use max_results to limit results.
    Examples: query='from:boss@company.com', query='subject:invoice', query='after:2024/01/01'
    """
    try:
        results = gmail_service.users().messages().list(
            userId="me", maxResults=max_results, q=query
        ).execute()

        messages = results.get("messages", [])
        if not messages:
            return [{"message": f"No emails found for query: {query}"}]

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
    to: recipient email address
    subject: email subject
    body: email body text
    file_path: full local path of the first file to attach
    extra_file_paths: comma separated paths of additional files (optional)
    """
    import os
    import mimetypes
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    from email.mime.base import MIMEBase
    from email import encoders

    try:
        # Build list of all files
        all_files = [file_path]
        if extra_file_paths:
            for p in extra_file_paths.split(","):
                p = p.strip()
                if p:
                    all_files.append(p)

        # Validate all files exist
        for fp in all_files:
            if not os.path.exists(fp):
                return {"status": "error", "message": f"File not found: {fp}"}

        # Build email
        msg = MIMEMultipart()
        msg["to"] = to
        msg["subject"] = subject
        msg.attach(MIMEText(body, "plain"))

        # Attach all files
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
            userId="me",
            body={"raw": raw}
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