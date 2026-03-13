import os
import pickle
import base64
import re
import threading
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from mcp.server.fastmcp import FastMCP
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import mimetypes

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
    if not hasattr(_local, "gmail_service") or _local.gmail_service is None:
        _local.gmail_service = get_gmail_service()
    return _local.gmail_service


def get_gmail_service(creds=None):
    if creds:
        return build("gmail", "v1", credentials=creds)

    creds_file = os.environ.get("MCP_CREDS_FILE")
    if creds_file and Path(creds_file).exists():
        try:
            with open(creds_file, "rb") as f:
                user_creds = pickle.load(f)
            if user_creds and user_creds.valid:
                return build("gmail", "v1", credentials=user_creds)
            if user_creds and user_creds.expired and user_creds.refresh_token:
                user_creds.refresh(Request())
                return build("gmail", "v1", credentials=user_creds)
        except Exception:
            pass

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

def build_message(
    to: str,
    subject: str,
    body: str,
    thread_id: str = None,
    cc: str = "",
    bcc: str = "",
) -> dict:
    message = MIMEMultipart()
    message["to"] = to
    message["subject"] = subject
    if cc:
        message["cc"] = cc
    if bcc:
        message["bcc"] = bcc
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
    if "parts" in payload:
        # Search recursively for text/plain
        def find_plain(parts):
            for part in parts:
                if part["mimeType"] == "text/plain":
                    data = part["body"].get("data", "")
                    if data:
                        return base64.urlsafe_b64decode(data).decode("utf-8", errors="ignore")
                if "parts" in part:
                    result = find_plain(part["parts"])
                    if result:
                        return result
            return None

        text = find_plain(payload["parts"])
        if text:
            return text

        # Fallback to first part with data
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


# ── NEW: strip placeholder values before sending ───────────────────────────────
_SKIP_MARKERS = {
    "(no subject)", "no subject", "(no body)", "no body",
    "none", "skip", "n/a", "(none)", "",
}

def _clean_field(value: str) -> str:
    """Return empty string if value is a skip placeholder, else return as-is."""
    return "" if value.strip().lower() in _SKIP_MARKERS else value.strip()
# ──────────────────────────────────────────────────────────────────────────────


def _attach_files(msg: MIMEMultipart, file_paths: list) -> list:
    """Attach files to a MIME message. Returns list of attached filenames."""
    attached = []
    for fp in file_paths:
        fp = fp.strip()
        if not fp or not os.path.exists(fp):
            continue
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
            attached.append(file_name)
    return attached


# ================= TOOLS =================

@mcp.tool()
def list_emails(max_results: str = "10", query: str = "", num: str = "") -> list:
    """
    List emails from Gmail inbox.
    max_results: number of emails to return (default 10).
    query: optional filter like 'from:someone@gmail.com' or 'subject:hello'.
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
    """
    gmail_service = get_service()
    try:
        msg = gmail_service.users().messages().get(
            userId="me", id=email_id, format="full"
        ).execute()

        headers = msg.get("payload", {}).get("headers", [])
        body = get_email_body(msg.get("payload", {}))

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
def send_email(
    to: str,
    subject: str = "",
    body: str = "",
    cc: str = "",
    bcc: str = ""
) -> dict:
    """
    Send an email with optional CC and BCC.
    to: recipient email address.
    subject: email subject — leave empty if not yet collected from user.
    body: email body text — leave empty if not yet collected from user.
    cc: comma separated CC email addresses (optional).
    bcc: comma separated BCC email addresses (optional).
    If subject and body are both empty, this tool will ask the user for them.
    Example: send_email(to='a@gmail.com', subject='Hi', body='Hello', cc='b@gmail.com')
    """
    # Strip placeholder values
    subject = _clean_field(subject)
    body = _clean_field(body)

    # Only ask if BOTH subject and body are empty — if at least one is provided, send immediately
    if not subject and not body:
        return {
            "status": "need_subject_and_body",
            "message": 'What should the **subject** and **message body** be?\n(Say "no subject" or "no body" to skip either)',
        }

    gmail_service = get_service()
    try:
        payload = build_message(to, subject, body, cc=cc, bcc=bcc)
        sent = gmail_service.users().messages().send(
            userId="me", body=payload
        ).execute()
        return {
            "status": "success",
            "message": "Email sent successfully.",
            "id": sent["id"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def reply_to_email(email_id: str, body: str) -> dict:
    """
    Reply to an existing email using its id.
    email_id: exact id from list_emails results.
    body: reply text.
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
        if not subject.lower().startswith("re:"):
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
def forward_email(email_id: str, to_email: str, note: str = "") -> dict:
    """
    Forward an email to another recipient.
    email_id: exact id of the email to forward.
    to_email: recipient email address to forward to.
    note: optional message to add above the forwarded content.
    """
    gmail_service = get_service()
    try:
        original = gmail_service.users().messages().get(
            userId="me", id=email_id, format="full"
        ).execute()

        headers = original.get("payload", {}).get("headers", [])
        original_from = parse_headers(headers, "From")
        original_subject = parse_headers(headers, "Subject")
        original_date = parse_headers(headers, "Date")
        original_body = get_email_body(original.get("payload", {}))

        subject = original_subject
        if not subject.lower().startswith("fwd:"):
            subject = "Fwd: " + subject

        forward_body = ""
        if note:
            forward_body = note + "\n\n"
        forward_body += (
            "---------- Forwarded message ----------\n"
            f"From: {original_from}\n"
            f"Date: {original_date}\n"
            f"Subject: {original_subject}\n"
            "\n"
            f"{original_body}"
        )

        payload = build_message(to_email, subject, forward_body)
        sent = gmail_service.users().messages().send(
            userId="me", body=payload
        ).execute()

        return {
            "status": "success",
            "message": f"Email forwarded to {to_email} successfully.",
            "id": sent["id"]
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def delete_email(email_id: str, confirmed: str = "") -> dict:
    """
    Move email to trash.
    Always call search_emails first to confirm correct email.
    Only call with confirmed='true' after user confirms.
    """
    gmail_service = get_service()
    try:
        if confirmed != "true":
            msg = gmail_service.users().messages().get(
                userId="me", id=email_id, format="metadata",
                metadataHeaders=["From", "Subject"]
            ).execute()
            headers = msg.get("payload", {}).get("headers", [])
            subject = parse_headers(headers, "Subject")
            sender = parse_headers(headers, "From")
            return {
                "status": "confirmation_required",
                "message": f"Found email from {sender} | Subject: {subject}. Move to trash?",
                "email_id": email_id,
                "subject": subject,
            }

        gmail_service.users().messages().trash(
            userId="me", id=email_id
        ).execute()
        return {"status": "success", "message": "Email moved to trash successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def restore_email(email_id: str, confirmed: str = "") -> dict:
    """
    Restore email from trash.
    Only call with confirmed='true' after user confirms.
    """
    gmail_service = get_service()
    try:
        if confirmed != "true":
            return {
                "status": "confirmation_required",
                "message": "Are you sure you want to restore this email?",
                "email_id": email_id,
            }

        gmail_service.users().messages().untrash(
            userId="me", id=email_id
        ).execute()
        return {"status": "success", "message": "Email restored successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def search_emails(query: str, max_results: str = "10") -> list:
    """
    Search emails using Gmail query syntax.
    Examples: 'from:boss@gmail.com', 'subject:invoice', 'in:trash test', 'after:2024/01/01'
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
def send_email_with_attachment(
    to: str,
    subject: str,
    body: str,
    file_path: str,
    extra_file_paths: str = "",
    cc: str = "",
    bcc: str = ""
) -> dict:
    """
    Send an email with one or more file attachments.
    to: recipient email.
    subject: email subject.
    body: email body text.
    file_path: full local path of the file to attach.
    extra_file_paths: comma separated paths for additional files (optional).
    cc: CC email addresses (optional).
    bcc: BCC email addresses (optional).
    """
    gmail_service = get_service()
    try:
        # Build full file list
        all_files = [file_path.strip()]
        if extra_file_paths:
            for p in extra_file_paths.split(","):
                p = p.strip()
                if p:
                    all_files.append(p)

        # Validate all files exist before doing anything
        missing = [fp for fp in all_files if not os.path.exists(fp)]
        if missing:
            return {
                "status": "error",
                "message": f"File(s) not found: {', '.join(missing)}. "
                           f"Use the exact full path returned by download_file."
            }

        # Build message
        msg = MIMEMultipart()
        msg["to"] = to
        msg["subject"] = subject
        if cc:
            msg["cc"] = cc
        if bcc:
            msg["bcc"] = bcc
        msg.attach(MIMEText(body, "plain"))

        # Attach files
        attached_names = _attach_files(msg, all_files)
        if not attached_names:
            return {
                "status": "error",
                "message": "Failed to attach files. Please check the file paths."
            }

        # Encode and send
        raw = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        sent = gmail_service.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()

        if not sent.get("id"):
            return {
                "status": "error",
                "message": "Email send failed — no message ID returned."
            }

        return {
            "status": "success",
            "message": f"Email sent to {to} with {', '.join(attached_names)} successfully.",
            "id": sent.get("id")
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def get_email_attachments(email_id: str) -> list:
    """
    List all attachments in an email.
    email_id: exact id from list_emails results.
    Returns list of attachments with their ids, names, and sizes.
    """
    gmail_service = get_service()
    try:
        msg = gmail_service.users().messages().get(
            userId="me", id=email_id, format="full"
        ).execute()

        attachments = []

        def find_attachments(parts):
            for part in parts:
                if part.get("filename") and part["filename"].strip():
                    attachment_id = part.get("body", {}).get("attachmentId", "")
                    size = part.get("body", {}).get("size", 0)
                    attachments.append({
                        "attachment_id": attachment_id,
                        "filename": part["filename"],
                        "mime_type": part.get("mimeType", "application/octet-stream"),
                        "size_bytes": size,
                        "size_kb": round(size / 1024, 1) if size else 0,
                    })
                if "parts" in part:
                    find_attachments(part["parts"])

        payload = msg.get("payload", {})
        if "parts" in payload:
            find_attachments(payload["parts"])

        if not attachments:
            return [{"message": "No attachments found in this email."}]

        return attachments
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def download_email_attachment(
    email_id: str,
    attachment_id: str,
    filename: str,
    download_folder: str = ""
) -> dict:
    """
    Download a specific attachment from an email.
    email_id: exact email id.
    attachment_id: attachment_id from get_email_attachments result.
    filename: filename from get_email_attachments result.
    download_folder: local folder to save (optional, defaults to ~/Downloads).
    """
    gmail_service = get_service()
    try:
        attachment = gmail_service.users().messages().attachments().get(
            userId="me",
            messageId=email_id,
            id=attachment_id
        ).execute()

        data = attachment.get("data", "")
        if not data:
            return {"status": "error", "message": "Attachment data is empty."}

        file_data = base64.urlsafe_b64decode(data)

        if not download_folder:
            download_folder = str(Path.home() / "Downloads")
        os.makedirs(download_folder, exist_ok=True)

        save_path = os.path.join(download_folder, filename)
        with open(save_path, "wb") as f:
            f.write(file_data)

        return {
            "status": "success",
            "message": f"Attachment '{filename}' downloaded successfully.",
            "saved_to": save_path,
            "attachment_path": save_path,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()