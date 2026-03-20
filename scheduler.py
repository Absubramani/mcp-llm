"""
scheduler.py — Long-lived companion process for scheduled email delivery.

Started automatically by run.sh alongside app.py.
Reads pending jobs from scheduled_emails.db every 30s,
registers them with APScheduler for exact-time delivery.
Survives indefinitely — never restarts in production.
"""

import os
import sys
import pickle
import base64
import sqlite3
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.jobstores.memory import MemoryJobStore
from apscheduler.executors.pool import ThreadPoolExecutor
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR

from google.auth.transport.requests import Request
from googleapiclient.discovery import build

# ── Config ────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DB_PATH    = BASE_DIR / "scheduled_emails.db"
POLL_SECS  = 30   # how often to check DB for new jobs

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("scheduler")

# ── APScheduler ───────────────────────────────────────────────────────────────
scheduler = BlockingScheduler(
    jobstores    = {"default": MemoryJobStore()},
    executors    = {"default": ThreadPoolExecutor(max_workers=10)},
    job_defaults = {
        "coalesce":           True,
        "max_instances":      1,
        "misfire_grace_time": 120,   # fire up to 2 min late (covers DB poll lag)
    },
    timezone = "UTC",
)

_registered_jobs: set = set()   # track job_ids already in APScheduler


# ── DB helpers ────────────────────────────────────────────────────────────────
def _db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
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


def _mark_sent(job_id: str):
    try:
        conn = _db()
        conn.execute("UPDATE scheduled_emails SET sent = 1 WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"Failed to mark job {job_id} as sent: {e}")


def _delete_job(job_id: str):
    try:
        conn = _db()
        conn.execute("DELETE FROM scheduled_emails WHERE job_id = ?", (job_id,))
        conn.commit()
        conn.close()
    except Exception as e:
        log.error(f"Failed to delete job {job_id}: {e}")


# ── Gmail sender ──────────────────────────────────────────────────────────────
def _get_gmail_service(creds_file: str):
    """Build Gmail service. Falls back to token.pickle if temp creds file is gone."""
    if not Path(creds_file).exists():
        fallback = BASE_DIR / "token.pickle"
        if fallback.exists():
            log.warning(f"Creds file gone, falling back to token.pickle")
            creds_file = str(fallback)
        else:
            raise FileNotFoundError(f"No valid creds file found")
    with open(creds_file, "rb") as f:
        creds = pickle.load(f)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        with open(creds_file, "wb") as f:
            pickle.dump(creds, f)
    return build("gmail", "v1", credentials=creds)


def _send_email(job: sqlite3.Row):
    """Send one scheduled email. Called by APScheduler in a thread."""
    job_id = job["job_id"]
    to     = job["to_addr"]
    try:
        svc = _get_gmail_service(job["creds_file"])

        msg = MIMEMultipart()
        msg["to"]      = to
        msg["subject"] = job["subject"] or ""
        if job["cc"]:
            msg["cc"]  = job["cc"]
        if job["bcc"]:
            msg["bcc"] = job["bcc"]
        msg.attach(MIMEText(job["body"] or "", "plain"))

        raw    = base64.urlsafe_b64encode(msg.as_bytes()).decode()
        result = svc.users().messages().send(
            userId="me", body={"raw": raw}
        ).execute()

        log.info(
            f"✅ Sent | to={to} | subject='{job['subject']}' | "
            f"msg_id={result.get('id')} | job_id={job_id}"
        )
        _delete_job(job_id)
        _registered_jobs.discard(job_id)

    except Exception as exc:
        log.error(f"❌ Failed | to={to} | job_id={job_id} | error={exc}")
        raise   # APScheduler logs EVENT_JOB_ERROR


# ── Poll DB and register new jobs ─────────────────────────────────────────────
def _poll_and_register():
    """
    Called every POLL_SECS seconds.
    Finds unsent jobs not yet registered and adds them to APScheduler.
    """
    try:
        conn = _db()
        rows = conn.execute(
            "SELECT * FROM scheduled_emails WHERE sent = 0"
        ).fetchall()
        conn.close()

        now_utc = datetime.now(timezone.utc)

        for row in rows:
            job_id = row["job_id"]
            if job_id in _registered_jobs:
                continue   # already scheduled

            run_date = datetime.fromisoformat(row["send_at_utc"])
            if run_date.tzinfo is None:
                run_date = run_date.replace(tzinfo=timezone.utc)

            # Already overdue — send immediately
            if run_date <= now_utc:
                log.warning(
                    f"Job {job_id} is overdue (was {row['display_time']}), sending now"
                )
                run_date = now_utc + timedelta(seconds=3)

            scheduler.add_job(
                _send_email,
                trigger          = "date",
                run_date         = run_date,
                id               = job_id,
                args             = [dict(row)],
                replace_existing = True,
            )
            _registered_jobs.add(job_id)
            log.info(
                f"Registered job_id={job_id} | to={row['to_addr']} | "
                f"at={row['display_time']}"
            )

    except Exception as e:
        log.error(f"Poll error: {e}")


# ── APScheduler event listeners ───────────────────────────────────────────────
def _on_executed(event):
    log.info(f"[apscheduler] Job {event.job_id} executed OK")

def _on_error(event):
    log.error(f"[apscheduler] Job {event.job_id} raised: {event.exception}")

scheduler.add_listener(_on_executed, EVENT_JOB_EXECUTED)
scheduler.add_listener(_on_error,    EVENT_JOB_ERROR)


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info(f"Scheduler starting | DB={DB_PATH} | poll every {POLL_SECS}s")

    # Poll immediately on startup to catch any jobs that were queued while
    # scheduler was not running (e.g. first start, manual restart)
    _poll_and_register()

    # Register recurring poll job
    scheduler.add_job(
        _poll_and_register,
        trigger      = "interval",
        seconds      = POLL_SECS,
        id           = "__poll__",
        replace_existing = True,
    )

    try:
        scheduler.start()   # blocks forever
    except (KeyboardInterrupt, SystemExit):
        log.info("Scheduler stopped")