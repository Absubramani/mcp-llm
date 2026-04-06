import logging
import re
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


def log_response(user_input: str, reply: str, duration_sec: float, tools_called: list = None):
    if tools_called is None:
        tools_called = []
    if not tools_called:
        logger.warning(f"RESPONSE | {duration_sec:.2f}s | NO TOOLS CALLED | {reply[:80].strip()!r}")
    else:
        tools_str = ", ".join(tools_called)
        logger.info(f"RESPONSE | {duration_sec:.2f}s | tools=[{tools_str}] | {reply[:80].strip()!r}")


def log_error(error: str, context: str = ""):
    raw_error = error
    lower     = error.lower()
    if any(x in lower for x in [
        "429", "rate_limit_exceeded", "rate_limit",
        "tokens per day", "tokens per minute",
        "requests per minute", "too many requests"
    ]):
        m = re.search(r'try again in (?:(\d+)m)?\s*(\d+)s?', error)
        if m:
            mins = m.group(1) or "0"
            secs = m.group(2)
            msg = f"RATE LIMIT | retry in {mins}m {secs}s"
        else:
            msg = "RATE LIMIT"
    elif "413" in error or "too large" in lower:
        msg = "TOKEN LIMIT | request too large"
    elif "tool_use_failed" in lower or "failed_generation" in lower:
        msg = "TOOL PARSE FAIL | invalid tool call syntax"
    elif "invalid id value" in lower:
        msg = "INVALID ID | placeholder used instead of real id"
    elif "failed to call a function" in lower:
        msg = "TOOL SCHEMA FAIL | Groq rejected tool schema"
    elif "invalid_api_key" in lower or "401" in error or "unauthorized" in lower:
        msg = "AUTH ERROR | invalid or expired API key"
    else:
        msg = error[:120]
    logger.error(f"ERROR    | {msg} | context={context} | RAW={raw_error}")


def log_rate_limit(wait_seconds: int, model: str):
    logger.warning(f"RATE LIMIT | {model} | retry in {wait_seconds // 60}m {wait_seconds % 60}s")


def log_llm_fallback(key_num: int, error_str: str):
    lower = error_str.lower()
    if any(x in lower for x in [
        "rate_limit", "429", "tokens per day",
        "rate_limit_exceeded", "tokens per minute",
        "requests per minute", "too many requests"
    ]):
        reason = "Rate limited"
    elif any(x in lower for x in ["invalid_api_key", "401", "unauthorized"]):
        reason = "Invalid API key"
    elif "tool_use_failed" in lower or "failed_generation" in lower:
        reason = "Tool parse failed"
    elif "400" in error_str and "failed to call a function" in lower:
        reason = "Tool schema rejected"
    else:
        reason = "Unknown error"

    if key_num == -2:
        logger.warning(f"LLM      | Mistral failed — {reason} → falling back")
    else:
        logger.warning(f"LLM      | Groq Key {key_num} failed — {reason} → falling back")

def log_llm_selected(provider: str, key_number: int = 0):
    if key_number:
        logger.info(f"LLM      | Using Groq Key {key_number}")
    elif provider == "mistral":
        logger.info("LLM      | Using Mistral")
    elif provider == "ollama":
        logger.info(f"LLM      | Using Ollama")
    else:
        logger.info(f"LLM      | Using {provider}")