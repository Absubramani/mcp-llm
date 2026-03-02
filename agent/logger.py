import logging
import time
from pathlib import Path
from datetime import datetime

# ── Log Directory ─────────────────────────────────────────────────────────────
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"

# ── Logger Setup ──────────────────────────────────────────────────────────────
logger = logging.getLogger("mcp_llm")
logger.setLevel(logging.DEBUG)

# Avoid duplicate handlers
if not logger.handlers:
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)

    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


# ── Logging Functions ─────────────────────────────────────────────────────────

def log_request(user_input: str):
    logger.info(f"REQUEST | user_input={user_input!r}")


def log_response(user_input: str, reply: str, duration_sec: float):
    logger.info(
        f"RESPONSE | duration={duration_sec:.2f}s | "
        f"input_len={len(user_input)} | reply_len={len(reply)}"
    )


def log_tool_call(tool_name: str, args: dict, result: str, duration_sec: float):
    # Truncate long results for readability
    result_preview = result[:200] + "..." if len(result) > 200 else result
    logger.info(
        f"TOOL | name={tool_name} | args={args} | "
        f"duration={duration_sec:.2f}s | result={result_preview!r}"
    )


def log_error(error: str, context: str = ""):
    logger.error(f"ERROR | context={context!r} | error={error!r}")


def log_rate_limit(wait_seconds: int, model: str):
    logger.warning(f"RATE_LIMIT | model={model} | retry_after={wait_seconds}s")