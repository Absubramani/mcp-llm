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
def list_emails(max_results: str = "10", query: str = "", num: str = "", page_token: str = "") -> dict:
    """
    List emails from Gmail inbox with pagination support.
    max_results: number of emails to return (default 10).
    query: optional filter like 'from:someone@gmail.com' or 'subject:hello'.
    page_token: pass next_page_token from previous result to get next page.
    Returns: {emails, next_page_token, has_more}
    """
    gmail_service = get_service()
    try:
        count = int(num) if num else int(max_results) if max_results else 10
        q = "in:inbox " + query if query else "in:inbox"

        kwargs = dict(userId="me", maxResults=count, q=q)
        if page_token:
            kwargs["pageToken"] = page_token

        results = gmail_service.users().messages().list(**kwargs).execute()
        messages = results.get("messages", [])
        next_token = results.get("nextPageToken", "")

        if not messages:
            return {"emails": [], "message": "No emails found.", "next_page_token": "", "has_more": False}

        emails = []
        for msg in messages:
            detail = gmail_service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()
            headers = detail.get("payload", {}).get("headers", [])
            label_ids = detail.get("labelIds", [])
            emails.append({
                "id": msg["id"],
                "thread_id": detail.get("threadId"),
                "from": parse_headers(headers, "From"),
                "subject": parse_headers(headers, "Subject"),
                "date": parse_headers(headers, "Date"),
                "snippet": detail.get("snippet", ""),
                "unread": "UNREAD" in label_ids,
            })

        return {
            "emails": emails,
            "next_page_token": next_token,
            "has_more": bool(next_token),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


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
def search_emails(query: str, max_results: str = "10", page_token: str = "") -> dict:
    """
    Search emails using Gmail query syntax with pagination support.
    Examples: 'from:boss@gmail.com', 'subject:invoice', 'in:trash test', 'after:2024/01/01'
    page_token: pass next_page_token from previous result to get next page.
    Returns: {emails, next_page_token, has_more}
    """
    gmail_service = get_service()
    try:
        count = int(max_results) if max_results else 10

        kwargs = dict(userId="me", maxResults=count, q=query)
        if page_token:
            kwargs["pageToken"] = page_token

        results = gmail_service.users().messages().list(**kwargs).execute()
        messages = results.get("messages", [])
        next_token = results.get("nextPageToken", "")

        if not messages:
            return {"emails": [], "message": f"No emails found for: {query}", "next_page_token": "", "has_more": False}

        emails = []
        for msg in messages:
            detail = gmail_service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()
            headers = detail.get("payload", {}).get("headers", [])
            label_ids = detail.get("labelIds", [])
            emails.append({
                "id": msg["id"],
                "from": parse_headers(headers, "From"),
                "subject": parse_headers(headers, "Subject"),
                "date": parse_headers(headers, "Date"),
                "snippet": detail.get("snippet", ""),
                "unread": "UNREAD" in label_ids,
            })

        return {
            "emails": emails,
            "next_page_token": next_token,
            "has_more": bool(next_token),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


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


# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULED EMAIL — SQLite-based (production grade)
#
# Architecture: MCP server is stateless (new process per tool call).
# Solution: write jobs to SQLite here; scheduler.py reads and sends them.
# SQLite is the shared state between the stateless MCP server and scheduler.
#
# scheduler.py runs as a long-lived companion process (started via run.sh).
# Jobs survive MCP restarts. Scheduler fires at exact time via APScheduler.
# ══════════════════════════════════════════════════════════════════════════════

import sqlite3 as _sqlite3
import uuid as _uuid
import logging as _logging
from datetime import datetime as _dt, timezone as _tz, timedelta as _td

_sched_log = _logging.getLogger("gmail_scheduler")
_DB_PATH   = BASE_DIR / "scheduled_emails.db"


def _db():
    """Open SQLite connection with row factory. Creates table if missing."""
    conn = _sqlite3.connect(str(_DB_PATH))
    conn.row_factory = _sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scheduled_emails (
            job_id       TEXT PRIMARY KEY,
            to_addr      TEXT NOT NULL,
            subject      TEXT,
            body         TEXT,
            cc           TEXT,
            bcc          TEXT,
            send_at_utc  TEXT NOT NULL,
            display_time TEXT,
            creds_file   TEXT NOT NULL,
            created_at   TEXT NOT NULL,
            sent         INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()
    return conn


@mcp.tool()
def schedule_email(
    to: str,
    send_at: str,
    subject: str = "",
    body: str    = "",
    cc: str      = "",
    bcc: str     = "",
) -> dict:
    """
    Schedule an email to be sent at a future date/time.
    to       : recipient email address.
    send_at  : natural language — "tomorrow 9am", "Friday 3pm", "2025-06-01 10:00".
    subject  : email subject  — leave "" if not yet collected from user.
    body     : email body     — leave "" if not yet collected from user.
    cc       : CC addresses (optional).
    bcc      : BCC addresses (optional).
    If both subject and body are empty the tool asks for them before scheduling.
    """
    import dateparser

    subject = _clean_field(subject)
    body    = _clean_field(body)

    # Ask for subject/body if both missing
    if not subject and not body:
        return {
            "status":  "need_subject_and_body",
            "message": "What should the **subject** and **message body** be?\n"
                       "(Say \"no subject\" or \"no body\" to skip either)",
        }

    try:
        parsed_dt = dateparser.parse(
            send_at,
            settings={
                "PREFER_DATES_FROM":        "future",
                "RETURN_AS_TIMEZONE_AWARE": True,
                "TIMEZONE":                 "Asia/Kolkata",
                "TO_TIMEZONE":              "UTC",
            },
        )
        if not parsed_dt:
            return {
                "status":  "error",
                "message": f"Could not understand \"{send_at}\". Try \"tomorrow 9am\" or \"Friday 3pm\".",
            }

        now_utc = _dt.now(_tz.utc)
        if parsed_dt <= now_utc + _td(minutes=1):
            return {
                "status":  "error",
                "message": "Scheduled time must be at least 1 minute in the future.",
            }

        job_id     = str(_uuid.uuid4())
        creds_file = os.environ.get("MCP_CREDS_FILE", str(TOKEN_FILE))
        ist        = _tz(offset=_td(hours=5, minutes=30))
        display_time = parsed_dt.astimezone(ist).strftime("%A, %B %d at %I:%M %p IST")

        conn = _db()
        conn.execute("""
            INSERT INTO scheduled_emails
                (job_id, to_addr, subject, body, cc, bcc,
                 send_at_utc, display_time, creds_file, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)
        """, (
            job_id, to, subject, body, cc or "", bcc or "",
            parsed_dt.astimezone(_tz.utc).isoformat(),
            display_time, creds_file,
            now_utc.isoformat(),
        ))
        conn.commit()
        conn.close()

        _sched_log.info(f"[schedule] Queued job_id={job_id} to={to} at={display_time}")

        return {
            "status":        "success",
            "message":       "Email scheduled successfully.",
            "scheduled_for": display_time,
            "job_id":        job_id,
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_scheduled_emails(max_results: str = "10") -> list:
    """
    List all emails currently queued for scheduled delivery.
    Returns job_id, recipient, subject, body, and scheduled send time.
    """
    try:
        count = int(max_results) if max_results else 10
        conn  = _db()
        rows  = conn.execute(
            """SELECT job_id, to_addr, subject, display_time
               FROM scheduled_emails
               WHERE sent = 0
               ORDER BY send_at_utc ASC LIMIT ?""",
            (count,),
        ).fetchall()
        conn.close()

        if not rows:
            return [{"message": "No scheduled emails found."}]

        return [
            {
                "job_id":        row["job_id"],
                "to":            row["to_addr"],
                "subject":       row["subject"] or "(no subject)",
                "body":          row["body"] or "(no body)",
                "scheduled_for": row["display_time"],
            }
            for row in rows
        ]
    except Exception as e:
        return [{"status": "error", "message": str(e)}]


@mcp.tool()
def cancel_scheduled_email(job_id: str) -> dict:
    """
    Cancel a scheduled email before it is sent. Cancels immediately — no confirmation needed.
    job_id: exact job_id from list_scheduled_emails.
    """
    try:
        conn = _db()
        row  = conn.execute(
            "SELECT * FROM scheduled_emails WHERE job_id = ? AND sent = 0",
            (job_id,),
        ).fetchone()

        if not row:
            conn.close()
            return {
                "status":  "error",
                "message": "Scheduled email not found — it may have already been sent or cancelled.",
            }

        conn.execute("DELETE FROM scheduled_emails WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()

        return {
            "status":  "success",
            "message": f"Scheduled email to **{row['to_addr']}** | Subject: **{row['subject'] or '(no subject)'}** | was cancelled successfully.",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def get_user_timezone() -> dict:
    """
    Get the system local timezone.
    Only call this when user explicitly asks what their timezone is.
    """
    try:
        import datetime
        utc_offset = datetime.datetime.now().astimezone().utcoffset()
        total_seconds = int(utc_offset.total_seconds())
        hours, remainder = divmod(abs(total_seconds), 3600)
        minutes = remainder // 60
        sign = "+" if total_seconds >= 0 else "-"
        offset_str = f"UTC{sign}{hours:02d}:{minutes:02d}"
        tz_name = datetime.datetime.now().astimezone().tzname()
        return {
            "status": "success",
            "timezone": tz_name,
            "utc_offset": offset_str,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# UNREAD EMAILS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def list_unread_emails(max_results: str = "10", page_token: str = "") -> dict:
    """
    List only unread emails from Gmail inbox with pagination.
    max_results: number of emails to return (default 10).
    page_token: pass next_page_token from previous result to get next page.
    Returns: {emails, next_page_token, has_more, unread_count}
    """
    gmail_service = get_service()
    try:
        count = int(max_results) if max_results else 10
        kwargs = dict(userId="me", maxResults=count, q="in:inbox is:unread")
        if page_token:
            kwargs["pageToken"] = page_token

        results = gmail_service.users().messages().list(**kwargs).execute()
        messages = results.get("messages", [])
        next_token = results.get("nextPageToken", "")

        if not messages:
            return {
                "emails": [],
                "message": "No unread emails found.",
                "next_page_token": "",
                "has_more": False,
                "unread_count": 0,
            }

        emails = []
        for msg in messages:
            detail = gmail_service.users().messages().get(
                userId="me", id=msg["id"], format="metadata",
                metadataHeaders=["From", "Subject", "Date"]
            ).execute()
            headers = detail.get("payload", {}).get("headers", [])
            emails.append({
                "id":       msg["id"],
                "from":     parse_headers(headers, "From"),
                "subject":  parse_headers(headers, "Subject"),
                "date":     parse_headers(headers, "Date"),
                "snippet":  detail.get("snippet", ""),
                "unread":   True,
            })

        return {
            "emails":          emails,
            "next_page_token": next_token,
            "has_more":        bool(next_token),
            "unread_count":    len(emails),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def mark_as_read(email_ids: str) -> dict:
    """
    Mark one or more emails as read.
    email_ids: single email id OR comma-separated list of ids for bulk operation.
    Example: mark_as_read('id1') or mark_as_read('id1,id2,id3')
    CRITICAL: Use EXACT id values from list_emails or list_unread_emails results.
    NEVER pass a number like '1' or '2' — always use the full id string.
    """
    gmail_service = get_service()
    try:
        ids = [i.strip() for i in email_ids.split(",") if i.strip()]
        if not ids:
            return {"status": "error", "message": "No email ids provided."}

        # Reject if any id looks like a position number instead of a real Gmail id
        invalid = [i for i in ids if i.isdigit() or len(i) < 8]
        if invalid:
            return {
                "status":  "error",
                "message": (
                    f"Invalid email id(s): {', '.join(invalid)}. "
                    "You must use the exact id from list_emails results (e.g. '19ce68f85eb2120e'), "
                    "not a position number. Call list_unread_emails first to get the real ids."
                ),
            }

        success, failed = [], []
        subjects = []
        for email_id in ids:
            try:
                gmail_service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={"removeLabelIds": ["UNREAD"]}
                ).execute()
                # Fetch subject for confirmation message
                try:
                    detail = gmail_service.users().messages().get(
                        userId="me", id=email_id, format="metadata",
                        metadataHeaders=["Subject"]
                    ).execute()
                    subj = parse_headers(detail.get("payload", {}).get("headers", []), "Subject")
                    subjects.append(subj or email_id)
                except Exception:
                    subjects.append(email_id)
                success.append(email_id)
            except Exception as e:
                failed.append({"id": email_id, "error": str(e)})

        if failed:
            return {
                "status":  "partial",
                "message": f"Marked {len(success)} email(s) as read. {len(failed)} failed.",
                "success": success,
                "failed":  failed,
            }
        subject_str = subjects[0] if len(subjects) == 1 else f"{len(subjects)} emails"
        return {
            "status":   "success",
            "message":  f"Marked {len(success)} email(s) as read successfully.",
            "subject":  subject_str,
            "count":    len(success),
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def mark_as_unread(email_ids: str) -> dict:
    """
    Mark one or more emails as unread.
    email_ids: single email id OR comma-separated list of ids for bulk operation.
    Example: mark_as_unread('id1') or mark_as_unread('id1,id2,id3')
    CRITICAL: Use EXACT id values from list_emails or list_unread_emails results.
    NEVER pass a number like '1' or '2' — always use the full id string.
    """
    gmail_service = get_service()
    try:
        ids = [i.strip() for i in email_ids.split(",") if i.strip()]
        if not ids:
            return {"status": "error", "message": "No email ids provided."}

        # Reject if any id looks like a position number instead of a real Gmail id
        invalid = [i for i in ids if i.isdigit() or len(i) < 8]
        if invalid:
            return {
                "status":  "error",
                "message": (
                    f"Invalid email id(s): {', '.join(invalid)}. "
                    "You must use the exact id from list_emails results (e.g. '19ce68f85eb2120e'), "
                    "not a position number. Call list_unread_emails first to get the real ids."
                ),
            }

        success, failed = [], []
        for email_id in ids:
            try:
                gmail_service.users().messages().modify(
                    userId="me",
                    id=email_id,
                    body={"addLabelIds": ["UNREAD"]}
                ).execute()
                success.append(email_id)
            except Exception as e:
                failed.append({"id": email_id, "error": str(e)})

        if failed:
            return {
                "status":  "partial",
                "message": f"Marked {len(success)} email(s) as unread. {len(failed)} failed.",
                "success": success,
                "failed":  failed,
            }
        return {
            "status":  "success",
            "message": f"Marked {len(success)} email(s) as unread successfully.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════════
# DRAFT EMAILS
# ══════════════════════════════════════════════════════════════════════════════

@mcp.tool()
def save_draft(
    to: str,
    subject: str = "",
    body: str = "",
    cc: str = "",
    bcc: str = ""
) -> dict:
    """
    Save an email as a draft without sending it.
    to: recipient email address.
    subject: email subject — leave empty if not yet collected from user.
    body: email body text — leave empty if not yet collected from user.
    cc: CC addresses (optional).
    bcc: BCC addresses (optional).
    If only subject given — ask for body. If only body given — ask for subject.
    If both empty — ask for both. If both given — save immediately.
    """
    subject = _clean_field(subject)
    body    = _clean_field(body)

    # Both missing — ask for both
    if not subject and not body:
        return {
            "status":  "need_subject_and_body",
            "message": 'What should the **subject** and **message body** be?\n(Say "no subject" or "no body" to skip either)',
        }

    # Only subject given — ask for body
    if subject and not body:
        return {
            "status":  "need_body",
            "message": f'What should the **message body** be for this draft?\n(Say "no body" to skip)',
            "subject": subject,
        }

    # Only body given — ask for subject
    if body and not subject:
        return {
            "status":  "need_subject",
            "message": f'What should the **subject** be for this draft?\n(Say "no subject" to skip)',
            "body": body,
        }

    gmail_service = get_service()
    try:
        payload = build_message(to, subject, body, cc=cc, bcc=bcc)
        draft = gmail_service.users().drafts().create(
            userId="me",
            body={"message": {"raw": payload["raw"]}}
        ).execute()

        return {
            "status":   "success",
            "message":  "Draft saved successfully.",
            "draft_id": draft["id"],
            "to":       to,
            "subject":  subject or "(no subject)",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def list_drafts(max_results: str = "10") -> list:
    """
    List all saved email drafts.
    max_results: number of drafts to return (default 10).
    Returns list of drafts with draft_id, to, subject, snippet.
    """
    gmail_service = get_service()
    try:
        count = int(max_results) if max_results else 10
        results = gmail_service.users().drafts().list(
            userId="me", maxResults=count
        ).execute()

        drafts_list = results.get("drafts", [])
        if not drafts_list:
            return [{"message": "No drafts found."}]

        drafts = []
        for d in drafts_list:
            try:
                detail = gmail_service.users().drafts().get(
                    userId="me", id=d["id"], format="metadata"
                ).execute()
                msg     = detail.get("message", {})
                headers = msg.get("payload", {}).get("headers", [])
                drafts.append({
                    "draft_id": d["id"],
                    "to":       parse_headers(headers, "To"),
                    "subject":  parse_headers(headers, "Subject") or "(no subject)",
                    "snippet":  msg.get("snippet", ""),
                    "date":     parse_headers(headers, "Date"),
                })
            except Exception:
                continue

        if not drafts:
            return [{"message": "No drafts found."}]

        return drafts
    except Exception as e:
        return [{"status": "error", "message": str(e)}]



@mcp.tool()
def update_draft(
    draft_id: str,
    subject: str = "",
    body: str = "",
    to: str = "",
    cc: str = "",
    bcc: str = ""
) -> dict:
    """
    Update an existing draft — change subject, body, recipient, cc, or bcc.
    draft_id: exact draft_id from list_drafts results.
    Only provide fields you want to update — others are kept as-is.
    CRITICAL: ALWAYS call list_drafts first to get the real draft_id.
    NEVER pass a number like 1 or 2 as draft_id.
    """
    gmail_service = get_service()
    try:
        # Reject invalid ids — auto-fetch list so LLM can pick the right one
        if draft_id.isdigit() or len(draft_id) < 8 or draft_id.startswith('['):
            results = gmail_service.users().drafts().list(userId="me", maxResults=10).execute()
            drafts_list = results.get("drafts", [])
            if not drafts_list:
                return {"status": "error", "message": "No drafts found."}
            drafts = []
            for d in drafts_list:
                try:
                    detail = gmail_service.users().drafts().get(
                        userId="me", id=d["id"], format="metadata"
                    ).execute()
                    msg = detail.get("message", {})
                    headers = msg.get("payload", {}).get("headers", [])
                    drafts.append({
                        "draft_id": d["id"],
                        "to":      parse_headers(headers, "To"),
                        "subject": parse_headers(headers, "Subject") or "(no subject)",
                    })
                except Exception:
                    continue
            return {
                "status":  "need_draft_id",
                "message": "Invalid draft_id. Use the exact draft_id from the list below.",
                "drafts":  drafts,
            }

        # Fetch existing draft content
        detail  = gmail_service.users().drafts().get(
            userId="me", id=draft_id, format="full"
        ).execute()
        msg     = detail.get("message", {})
        headers = msg.get("payload", {}).get("headers", [])

        # Use existing values for fields not being updated
        final_to      = _clean_field(to)      or parse_headers(headers, "To")
        final_subject = _clean_field(subject) or parse_headers(headers, "Subject") or ""
        final_cc      = _clean_field(cc)      or parse_headers(headers, "Cc") or ""
        final_bcc     = _clean_field(bcc)     or parse_headers(headers, "Bcc") or ""

        # Get existing body if not updating
        final_body = _clean_field(body)
        if not final_body:
            final_body = get_email_body(msg.get("payload", {}))
            if final_body == "No body content.":
                final_body = ""

        # Build updated message
        payload = build_message(final_to, final_subject, final_body, cc=final_cc, bcc=final_bcc)

        # Update the draft
        gmail_service.users().drafts().update(
            userId="me",
            id=draft_id,
            body={"message": {"raw": payload["raw"]}}
        ).execute()

        return {
            "status":   "success",
            "message":  f"Draft updated successfully.",
            "draft_id": draft_id,
            "to":       final_to,
            "subject":  final_subject or "(no subject)",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def send_draft(draft_id: str) -> dict:
    """
    Send an existing draft email and delete it after sending.
    draft_id: exact draft_id from list_drafts results.
    CRITICAL: ALWAYS call list_drafts first to get the real draft_id.
    NEVER pass a number like '1' or '2' — always use the full draft_id string.
    """
    gmail_service = get_service()
    try:
        # Reject position numbers — force LLM to use real ids
        if draft_id.isdigit() or len(draft_id) < 8:
            # Auto-fetch drafts and return them so LLM can pick the right one
            gmail_service2 = get_service()
            results = gmail_service2.users().drafts().list(userId="me", maxResults=10).execute()
            drafts_list = results.get("drafts", [])
            if not drafts_list:
                return {"status": "error", "message": "No drafts found."}
            drafts = []
            for d in drafts_list:
                try:
                    detail = gmail_service2.users().drafts().get(
                        userId="me", id=d["id"], format="metadata"
                    ).execute()
                    msg = detail.get("message", {})
                    headers = msg.get("payload", {}).get("headers", [])
                    drafts.append({
                        "draft_id": d["id"],
                        "to":      parse_headers(headers, "To"),
                        "subject": parse_headers(headers, "Subject") or "(no subject)",
                    })
                except Exception:
                    continue
            return {
                "status":  "need_draft_id",
                "message": "Please use the exact draft_id from the list below, not a number.",
                "drafts":  drafts,
            }

        # Fetch draft details for confirmation message
        detail  = gmail_service.users().drafts().get(
            userId="me", id=draft_id, format="metadata"
        ).execute()
        msg     = detail.get("message", {})
        headers = msg.get("payload", {}).get("headers", [])
        to      = parse_headers(headers, "To")
        subject = parse_headers(headers, "Subject") or "(no subject)"

        # Send the draft — Gmail auto-deletes draft on send
        gmail_service.users().drafts().send(
            userId="me",
            body={"id": draft_id}
        ).execute()

        return {
            "status":  "success",
            "message": f"Draft sent to **{to}** successfully and removed from drafts.",
            "to":      to,
            "subject": subject,
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
def delete_draft(draft_id: str, confirmed: str = "") -> dict:
    """
    Delete a draft email permanently.
    draft_id: exact draft_id from list_drafts results.
    Call first without confirmed to show details.
    Call with confirmed='true' after user confirms.
    """
    gmail_service = get_service()
    try:
        detail  = gmail_service.users().drafts().get(
            userId="me", id=draft_id, format="metadata"
        ).execute()
        msg     = detail.get("message", {})
        headers = msg.get("payload", {}).get("headers", [])
        to      = parse_headers(headers, "To")
        subject = parse_headers(headers, "Subject") or "(no subject)"

        if confirmed != "true":
            return {
                "status":  "confirmation_required",
                "message": f"Delete draft to **{to}** | Subject: **{subject}**?",
                "draft_id": draft_id,
            }

        gmail_service.users().drafts().delete(userId="me", id=draft_id).execute()
        return {"status": "success", "message": f"Draft '{subject}' deleted successfully."}
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    mcp.run()