import logging
from pathlib import Path
from datetime import datetime

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"app_{datetime.now().strftime('%Y-%m-%d')}.log"

logger = logging.getLogger("mcp_llm")
logger.setLevel(logging.DEBUG)

if not logger.handlers:
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(file_handler)


def _sep():
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write("\n")


def log_request(user_input: str):
    _sep()
    logger.info(f"REQUEST  | {user_input.strip()!r}")


def log_tool_call(tool_name: str, args: dict, result: str, duration_sec: float):
    status = "FAIL" if '"status": "error"' in result else "OK"
    logger.info(f"TOOL     | [{status}] {tool_name} | args={args} | {duration_sec:.2f}s")


def log_response(user_input: str, reply: str, duration_sec: float, tools_called: list = []):
    if not tools_called:
        logger.warning(f"RESPONSE | {duration_sec:.2f}s | NO TOOLS CALLED | {reply[:80].strip()!r}")
    else:
        tools_str = ", ".join(tools_called)
        logger.info(f"RESPONSE | {duration_sec:.2f}s | tools=[{tools_str}] | {reply[:80].strip()!r}")


def log_error(error: str, context: str = ""):
    if "429" in error or "rate_limit_exceeded" in error:
        import re
        m = re.search(r'try again in (\d+)m(\d+)', error)
        msg = f"RATE LIMIT | retry in {m.group(1)}m {m.group(2)}s" if m else "RATE LIMIT"
    elif "413" in error or "too large" in error.lower():
        msg = "TOKEN LIMIT | request too large"
    elif "tool_use_failed" in error:
        msg = "TOOL PARSE FAIL | invalid tool call syntax"
    elif "Invalid id value" in error:
        msg = "INVALID ID | placeholder used instead of real id"
    else:
        msg = error[:120]
    logger.error(f"ERROR    | {msg} | context={context}")


def log_rate_limit(wait_seconds: int, model: str):
    logger.warning(f"RATE LIMIT | {model} | retry in {wait_seconds // 60}m {wait_seconds % 60}s")