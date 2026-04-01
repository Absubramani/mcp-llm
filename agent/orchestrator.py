import os
import re
import time
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import Field, create_model
from agent.tool_schema import fetch_tools
from agent.tool_executor import execute_tool, get_last_tool_result
from agent.prompt import get_prompt


# ── Section Router ────────────────────────────────────────────────────────────
def _detect_sections(user_input: str) -> list:
    """
    Detect which tool sections are needed based on user input.
    Returns list of active sections — only those sections are included
    in the system prompt, keeping token count low.
    """
    lower = user_input.lower()

    gmail_words  = [
        "email", "mail", "inbox", "draft", "send", "reply", "forward",
        "attachment", "schedule", "unread", "read", "mark", "trash",
        "restore", "cc", "bcc", "subject", "compose", "gmail",
    ]
    drive_words  = [
        "file", "folder", "drive", "upload", "download", "document",
        "sheet", "spreadsheet", "slides", "pdf", "docx", "xlsx", "pptx",
        "txt", "csv", "create", "rename", "move", "copy", "share",
        "recent", "search", "find", "open", "read", "summarize",
    ]
    github_words = [
        "repo", "repository", "repositories", "issue", "issues",
        "pull request", "pull requests", "pr", "prs", "github", "branch", "commit",
        "readme", "code", "bug", "feature", "merge", "create repo",
        "new repo", "files in", "list files", "open pr", "create pr",
    ]

    sections = []
    if any(w in lower for w in gmail_words):
        sections.append("gmail")
    if any(w in lower for w in drive_words):
        sections.append("drive")
    if any(w in lower for w in github_words):
        sections.append("github")

    # If nothing matched — include all (greetings, unclear input, etc.)
    if not sections:
        sections = ["gmail", "drive", "github"]

    return sections
from agent.logger import log_llm_selected
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


# ── LangChain 0.2.x NoneType tool input bug ───────────────────────────────────
# Fixed directly in .venv/lib/python3.11/site-packages/langchain/agents/
# output_parsers/tools.py — two line changes:
#   if "__arg1" in _tool_input  →  if _tool_input and "__arg1" in _tool_input
#   tool_input = _tool_input    →  tool_input = _tool_input or {}
# No monkey-patch needed here since the source file is fixed.
# ─────────────────────────────────────────────────────────────────────────────


# ── Input Preprocessor ────────────────────────────────────────────────────────
def _preprocess_input(user_input: str) -> str:
    processed = user_input.strip()

    file_extensions = [
        '.docx', '.pdf', '.xlsx', '.txt', '.csv', '.md',
        '.pptx', '.json', '.xml', '.html', '.py', '.js',
        '.yaml', '.yml', '.sql', '.sh',
    ]
    lower = processed.lower()
    for ext in file_extensions:
        if lower.endswith(ext):
            processed = processed[:-len(ext)]
            break

    processed = re.sub(
        r'([A-Za-z0-9])_([A-Za-z0-9])',
        lambda m: m.group(1) + ' ' + m.group(2),
        processed
    )

    return processed


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm(key_index: int = 0):
    provider = os.getenv("LLM_PROVIDER", "groq")

    if provider == "groq":
        groq_keys = [
            os.getenv("GROQ_API_KEY_1"),
            os.getenv("GROQ_API_KEY_2"),
            os.getenv("GROQ_API_KEY_3"),
        ]
        valid_keys = [k for k in groq_keys if k]

        if 0 <= key_index < len(valid_keys):
            log_llm_selected("groq", key_index + 1)
            return ChatGroq(
                api_key=valid_keys[key_index],
                model="llama-3.3-70b-versatile",
                temperature=0,
            ), key_index

    # Mistral fallback — ONLY when explicitly requested via key_index == -2
    if key_index == -2:
        mistral_key = os.getenv("MISTRAL_API_KEY")
        if mistral_key:
            try:
                from langchain_openai import ChatOpenAI
                log_llm_selected("mistral")
                return ChatOpenAI(
                    api_key=mistral_key,
                    base_url="https://api.mistral.ai/v1",
                    model="mistral-large-latest",
                    temperature=0,
                ), -2
            except Exception:
                pass

    # Ollama — last resort only (key_index == -1 always lands here)
    from langchain_ollama import ChatOllama
    log_llm_selected("ollama")
    return ChatOllama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0,
    ), -1


# ── Tools Builder ─────────────────────────────────────────────────────────────
def build_langchain_tools(
    all_tools: list,
    tool_server_map: dict,
    creds=None,
    github_token: dict = None,
) -> list:
    # Compressed descriptions sent to LLM to stay within token limits.
    # Full docstrings are preserved in the MCP server files for code readability.
    _SHORT_DESC = {
        # Gmail
        "list_emails":                "List inbox emails. max_results, query, page_token optional.",
        "read_email":                 "Read full email by id.",
        "send_email":                 "Send email. to required. subject/body empty triggers ask flow. cc/bcc optional.",
        "reply_to_email":             "Reply to email by id. body required.",
        "forward_email":              "Forward email to another address. email_id, to_email required.",
        "delete_email":               "Move email to trash. confirmed='true' required.",
        "restore_email":              "Restore email from trash. confirmed='true' required.",
        "search_emails":              "Search emails by Gmail query. e.g. from:x@y.com, subject:foo.",
        "send_email_with_attachment": "Send email with local file attachment. file_path required.",
        "get_email_attachments":      "List attachments in an email by id.",
        "download_email_attachment":  "Download attachment. email_id, attachment_id, filename required.",
        "list_unread_emails":         "List unread inbox emails. max_results, page_token optional.",
        "mark_as_read":               "Mark emails as read. email_ids: exact id or comma-separated ids.",
        "mark_as_unread":             "Mark emails as unread. email_ids: exact id or comma-separated ids.",
        "save_draft":                 "Save email as draft. to required. subject/body empty triggers ask flow.",
        "list_drafts":                "List all saved drafts.",
        "update_draft":               "Update draft subject/body/to. Call list_drafts in the same step to get draft_id before calling this.",
        "send_draft":                 "Send a draft. Call list_drafts in the same step to get draft_id before calling this.",
        "delete_draft":               "Delete a draft. confirmed='true' required.",
        "schedule_email":             "Schedule email for later. to, send_at required. subject/body empty triggers ask.",
        "list_scheduled_emails":      "List all scheduled emails.",
        "cancel_scheduled_email":     "Cancel a scheduled email by job_id.",
        "get_user_timezone":          "Get system timezone. Only call when user asks for their timezone.",
        # Drive
        "create_folder":              "Create folder or nested folders. path required.",
        "list_files":                 "List files in folder. path empty = root.",
        "create_text_file":           "Create text file in Drive. path and content required.",
        "read_file":                  "Read file content from Drive. path required.",
        "delete_file":                "Move file to trash. confirmed='true' required.",
        "restore_file":               "Restore file from trash. confirmed='true' required.",
        "move_file":                  "Move file to another folder.",
        "rename_file":                "Rename a file or folder.",
        "copy_file":                  "Copy file to another folder.",
        "search_files":               "Search files by keyword. NEVER include extension in query.",
        "get_file_info":              "Get file details: name, type, size, modified date.",
        "upload_file":                "Upload local file to Drive. file_path required.",
        "download_file":              "Download Drive file to local machine.",
        "list_recent_files":          "List recently modified files.",
        "share_file":                 "Share file with email. role: reader/commenter/writer.",
        # GitHub
        "list_repos":                 "List authenticated user's GitHub repos. limit optional.",
        "create_repo":                "Create a new GitHub repo. name required. description, private, auto_init optional.",
        "search_repos":               "Search GitHub repositories by keyword. query required.",
        "list_repo_files":            "List files and folders in a repo directory. repo required. path and branch optional.",
        "read_file_from_repo":        "Read a file from a GitHub repo. repo and file_path required. branch optional.",
        "list_issues":                "List issues in a repo. repo required. state: open/closed/all. limit optional.",
        "create_issue":               "Create a new issue. repo and title required. body and labels optional.",
        "read_issue":                 "Read a specific issue. repo and issue_number required.",
        "list_pull_requests":         "List pull requests in a repo. repo required. state: open/closed/all.",
        "create_pull_request":        "Create a pull request. repo, title, head required. base defaults to main. body optional.",
    }

    langchain_tools = []

    for tool in all_tools:
        func_info = tool["function"]
        tool_name = func_info["name"]
        tool_desc = _SHORT_DESC.get(tool_name, func_info["description"][:120])
        server_name = tool_server_map.get(tool_name, "drive")
        parameters = func_info.get("parameters", {})
        properties = parameters.get("properties", {})
        required = parameters.get("required", [])

        fields = {}
        for prop_name, prop_info in properties.items():
            if prop_name in required:
                fields[prop_name] = (str, Field(..., description=prop_info.get("description", "")))
            else:
                default_val = prop_info.get("default", "")
                if default_val is None:
                    default_val = ""
                fields[prop_name] = (str, Field(
                    default=str(default_val),
                    description=prop_info.get("description", "")
                ))

        if not fields:
            fields["dummy"] = (str, Field(default=""))

        ArgsModel = create_model(f"{tool_name}Args", **fields)

        def make_tool_fn(t_name, s_name, user_creds, gh_token):
            # GitHub tools — never drop args, even empty strings
            # because required fields like 'name' must always reach execute_tool
            _gh_tools = {
                "list_repos", "create_repo", "search_repos", "list_repo_files",
                "read_file_from_repo", "list_issues", "create_issue",
                "read_issue", "list_pull_requests", "create_pull_request",
            }
            def tool_fn(**kwargs) -> str:
                kwargs.pop("dummy", None)
                if t_name not in _gh_tools:
                    kwargs = {k: v for k, v in kwargs.items() if v != ""}
                return execute_tool(
                    s_name, t_name, kwargs,
                    creds=user_creds,
                    github_token=gh_token,
                )
            tool_fn.__name__ = t_name
            tool_fn.__doc__ = tool_desc
            return tool_fn

        langchain_tool = StructuredTool.from_function(
            func=make_tool_fn(tool_name, server_name, creds, github_token),
            name=tool_name,
            description=tool_desc,
            args_schema=ArgsModel,
        )
        langchain_tools.append(langchain_tool)

    return langchain_tools


# ── Shared Helpers ────────────────────────────────────────────────────────────
def _build_agent_executor(llm, langchain_tools, sections: list = None):
    agent = create_openai_tools_agent(llm, langchain_tools, get_prompt(sections))
    return AgentExecutor(
        agent=agent,
        tools=langchain_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        early_stopping_method="generate",
    )


def _build_chat_history(conversation_history, fallback_hint=None, sections=None):
    """
    Build chat history keeping only recent messages.
    Limit varies by active sections to stay within Groq token limits:
    - GitHub only: 4 messages (prompt is smaller but tool results can be large)
    - Gmail only: 4 messages (prompt is larger)
    - Drive only: 4 messages
    - Mixed/all: 2 messages (largest prompt, least history to fit)
    """
    if sections is None:
        sections = ["gmail", "drive", "github"]

    # Determine history limit based on active sections
    if len(sections) >= 3:
        limit = 2   # all sections — smallest history to fit within tokens
    else:
        limit = 4   # single or dual section — can afford more history

    chat_history = []
    for msg in conversation_history[-limit:]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg.get("content"):
            chat_history.append(AIMessage(content=msg["content"]))
    if fallback_hint:
        chat_history.append(HumanMessage(content=fallback_hint))
    return chat_history


def _classify_error(error_str: str) -> str:
    lower = error_str.lower()
    # 413 token limit — request too large, retrying won't help
    if "413" in error_str or ("too large" in lower and "token" in lower):
        return "token_limit"
    if any(x in lower for x in [
        "rate_limit", "rate limit", "429",
        "rate_limit_exceeded", "tokens per day",
        "tokens per minute", "requests per minute",
        "too many requests",
    ]):
        return "rate_limit"
    if any(x in lower for x in [
        "invalid_api_key", "invalid api key", "401",
        "authentication", "unauthorized",
    ]):
        return "auth_error"
    if any(x in lower for x in [
        "tool_use_failed", "failed_generation",
        "pydantic", "serializ", "unexpectedvalue",
    ]):
        return "parse_fail"
    if "400" in error_str and (
        "failed to call a function" in lower or
        "decommissioned" in lower or
        "does not exist" in lower
    ):
        return "groq_400"
    return "unknown"


def _get_groq_keys() -> list:
    return [k for k in [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ] if k]


def _parse_tool_result(raw: str) -> list:
    """
    Parse tool result into a list of dicts.
    Handles both JSON array and newline-separated JSON objects
    (MCP returns each item as a separate JSON object on its own line).
    """
    import json
    if not raw or not raw.strip():
        return []
    raw = raw.strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            return [result]
    except Exception:
        pass
    items = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict):
                items.append(obj)
        except Exception:
            continue
    return items


def _format_repos(raw: str) -> str:
    """Format list_repos tool result into clean markdown."""
    repos = _parse_tool_result(raw)
    if not repos:
        return ""
    NL = "\n"
    lines = ["🐙 Your GitHub repositories:" + NL]
    for i, r in enumerate(repos, 1):
        name  = r.get("name", "")
        lang  = r.get("language") or "—"
        stars = r.get("stars", 0)
        priv  = " 🔒" if r.get("private") else ""
        lines.append(f"{i}. **{name}**{priv} | ⭐ {stars} | {lang}")
    lines.append(NL + "Reply with a repo name to list issues, PRs, or read a file.")
    return NL.join(lines)


def _format_issues(raw: str) -> str:
    """Format list_issues tool result into clean markdown."""
    issues = _parse_tool_result(raw)
    if not issues:
        return ""
    NL = "\n"
    lines = ["🐛 Issues:" + NL]
    for i, iss in enumerate(issues, 1):
        num   = iss.get("number", "")
        title = iss.get("title", "")
        state = iss.get("state", "")
        lines.append(f"{i}. **#{num}** {title} | {state}")
    lines.append(NL + "Reply with a number to read the full issue.")
    return NL.join(lines)


def _format_prs(raw: str) -> str:
    """Format list_pull_requests tool result into clean markdown."""
    prs = _parse_tool_result(raw)
    if not prs:
        return ""
    NL = "\n"
    lines = ["🔀 Pull requests:" + NL]
    for i, pr in enumerate(prs, 1):
        num   = pr.get("number", "")
        title = pr.get("title", "")
        state = pr.get("state", "")
        lines.append(f"{i}. **#{num}** {title} | {state}")
    return NL.join(lines)


def _format_search_repos(raw: str) -> str:
    """Format search_repos tool result into clean markdown."""
    repos = _parse_tool_result(raw)
    if not repos:
        return ""
    NL = "\n"
    lines = ["🔍 Search results:" + NL]
    for i, r in enumerate(repos, 1):
        name  = r.get("name", "")
        lang  = r.get("language") or "—"
        stars = r.get("stars", 0)
        lines.append(f"{i}. **{name}** | ⭐ {stars} | {lang}")
    return NL.join(lines)


def _smart_reply_from_tools(tools_done: list, user_input: str) -> str:
    """Generate a clean success/summary message based on which tools ran, without needing LLM."""
    import re
    tools_set = set(tools_done)

    def extract_filename(pattern: str) -> str:
        """Extract filename from user_input preserving original case."""
        m = re.search(pattern, user_input, re.IGNORECASE)
        return m.group(1).strip() if m else "the file"

    # Upload to Drive flow (download + upload both ran)
    if "upload_file" in tools_set:
        filename = extract_filename(r"upload\s+(.+?)\s+to\s+drive")
        if filename == "the file":
            # try "upload 1", "upload 2" etc
            m2 = re.search(r"upload\s+(\d+)", user_input, re.IGNORECASE)
            filename = f"item {m2.group(1)}" if m2 else "the file"
        return f"✅ **{filename}** uploaded to **Google Drive** successfully! 📁"

    # Download only flow
    if "download_email_attachment" in tools_set:
        filename = extract_filename(r"download\s+(.+?)(?:\s+to|\s*$)")
        return f"✅ **{filename}** downloaded to your Downloads folder successfully!"

    # Search + get attachments = list flow — don't show anything, list already rendered
    if "search_emails" in tools_set and "get_email_attachments" in tools_set:
        return ""

    # GitHub smart replies — format directly from stored tool results
    if "list_repos" in tools_set:
        raw = get_last_tool_result("list_repos")
        formatted = _format_repos(raw)
        return formatted if formatted else "🐙 Repos retrieved — please try again to see them."

    if "list_issues" in tools_set:
        raw = get_last_tool_result("list_issues")
        formatted = _format_issues(raw)
        return formatted if formatted else "🐛 Issues retrieved — please try again to see them."

    if "list_pull_requests" in tools_set:
        raw = get_last_tool_result("list_pull_requests")
        formatted = _format_prs(raw)
        return formatted if formatted else "🔀 PRs retrieved — please try again to see them."

    if "search_repos" in tools_set:
        raw = get_last_tool_result("search_repos")
        formatted = _format_search_repos(raw)
        return formatted if formatted else "🔍 Results retrieved — please try again to see them."

    # GitHub smart replies
    if "create_repo" in tools_set:
        return "✅ GitHub repository created successfully! 🐙"
    if "create_issue" in tools_set:
        return "✅ GitHub issue created successfully! 🐛"
    if "create_pull_request" in tools_set:
        return "✅ Pull request created successfully! 🔀"

    # Generic fallback based on last tool
    last = tools_done[-1] if tools_done else ""
    if "upload" in last:
        return "✅ File uploaded to Google Drive successfully! 📁"
    if "download" in last:
        return "✅ File downloaded successfully!"
    if "send_email" in last:
        return "✅ Email sent successfully! 📧"
    if "search" in last or "list" in last:
        return ""

    return "✅ Done!"


def _handle_exception(
    error_str: str,
    key_index: int,
    groq_keys: list,
    has_mistral: bool,
    fallback_hint: str,
    get_tools_called_fn,
    log_llm_fallback_fn,
    log_error_fn,
    user_input: str,
):
    """
    Returns (next_key_index, should_break, fallback_hint, error_reply)
    """
    error_type = _classify_error(error_str)

    # Auth error — stop immediately
    if error_type == "auth_error":
        log_error_fn(error_str, context=f"user_input={user_input!r}")
        return key_index, True, fallback_hint, "I ran into an issue processing your request. Please try again."

    # Token limit (413) — request too large, retrying with same/other LLM won't help
    # If tools already ran, generate smart reply from results instead of hallucinating
    if error_type == "token_limit":
        tools_done = get_tools_called_fn()
        if tools_done:
            log_llm_fallback_fn(key_index + 1 if key_index >= 0 else key_index, error_str)
            smart_reply = _smart_reply_from_tools(tools_done, user_input)
            return key_index, True, fallback_hint, smart_reply
        log_error_fn(error_str, context=f"user_input={user_input!r}")
        return key_index, True, fallback_hint, "Your request is too large to process. Try asking for fewer items."

    # Mistral failed — if tools already ran, generate a smart success/summary message
    if key_index == -2:
        tools_done = get_tools_called_fn()
        if tools_done:
            log_llm_fallback_fn(-2, error_str)
            smart_reply = _smart_reply_from_tools(tools_done, user_input)
            return key_index, True, fallback_hint, smart_reply
        log_llm_fallback_fn(-2, error_str)
        return -1, False, fallback_hint, None

    # Ollama failed — stop
    if key_index == -1:
        log_error_fn(error_str, context=f"user_input={user_input!r}")
        return key_index, True, fallback_hint, "I ran into an issue processing your request. Please try again."

    # Groq keys — try next key
    if error_type != "unknown" and key_index >= 0:
        log_llm_fallback_fn(key_index + 1, error_str)
        tools_done = get_tools_called_fn()
        if tools_done:
            fallback_hint = (
                f"[System: Previous attempt already called tools: {', '.join(tools_done)}. "
                f"Do NOT call these tools again. Use results already obtained.]"
            )
        next_key = key_index + 1
        if next_key >= len(groq_keys):
            next_key = -2 if has_mistral else -1
        return next_key, False, fallback_hint, None

    # Unknown error — stop
    log_error_fn(error_str, context=f"user_input={user_input!r}")
    return key_index, True, fallback_hint, "I ran into an issue processing your request. Please try again."


# ── Garbage detection ─────────────────────────────────────────────────────────
_GARBAGE_MARKERS = [
    "type': 'reference'",
    "type': 'text'",
    "{'type':",
    '{"type":',
    "responded: get_email_attachments",
    "responded: search_emails",
    "responded: download",
    "responded: upload",
    "responded: list_emails",
    "responded: read_email",
    'search_emails": {',
    'list_emails": {',
    'get_email_attachments": {',
    'download_email_attachment": {',
    'upload_file": {',
    'download_file": {',
    'search_files": {',
    'read_file": {',
    'send_email": {',
    'upload_file">',
    'upload_file">![](',
    '">![]({"file_path"',
    '"file_path": "/',
    "Invoking: `upload_file`",
    "Invoking: `download",
    # GitHub garbage markers
    'list_repos": {',
    'create_repo": {',
    'search_repos": {',
    'list_repo_files": {',
    'read_file_from_repo": {',
    'list_issues": {',
    'create_issue": {',
    'read_issue": {',
    'list_pull_requests": {',
    'create_pull_request": {',
]


def _is_garbage_reply(reply: str) -> bool:
    if not reply:
        return False
    if "pydanticserializat" in reply.lower():
        return True
    if reply.strip().startswith("[") and "{'type'" in reply:
        return True
    # Mistral leaks raw tool calls or markdown image syntax with JSON args
    if reply.strip().startswith("upload_file") or reply.strip().startswith("download_"):
        return True
    if ">![]({"  in reply or '"file_path"' in reply:
        return True
    # Mistral garbled tool call — tool name directly followed by JSON without proper invoke
    import re
    if re.search(r'(search_emails|list_emails|get_email|download_|upload_file|send_email|search_files|read_file|list_repos|create_repo|search_repos|list_repo_files|read_file_from_repo|list_issues|create_issue|read_issue|list_pull_requests|create_pull_request)\s*[א-׿؀-ۿﬀ-﷿]?\s*\{', reply):
        return True
    # Raw JSON args with no tool invocation wrapper
    if re.search(r'^[a-z_]+\s*\{.*"query"', reply.strip()):
        return True
    for marker in _GARBAGE_MARKERS:
        if marker in reply:
            return True
    return False


# ── Standard Agent ────────────────────────────────────────────────────────────

# -- Standard Agent ------------------------------------------------------------
def run_agent(
    user_input: str,
    conversation_history: list[dict],
    start_key_index: int = 0,
    creds=None,
    github_token: dict = None,
) -> tuple[str, list[dict], int]:

    from agent.logger import log_request, log_response, log_error, log_llm_fallback
    from agent.tool_executor import reset_tools_called, get_tools_called

    user_input = _preprocess_input(user_input)
    log_request(user_input)
    reset_tools_called()
    start = time.time()

    sections = _detect_sections(user_input)
    all_tools, tool_server_map = fetch_tools()
    langchain_tools = build_langchain_tools(
        all_tools, tool_server_map,
        creds=creds,
        github_token=github_token,
    )
    groq_keys = _get_groq_keys()
    has_mistral = bool(os.getenv("MISTRAL_API_KEY"))

    key_index = start_key_index
    last_working_key = start_key_index
    reply = None
    fallback_hint = None
    retry_count = 0
    max_retries = 6

    while True:
        if retry_count >= max_retries:
            reply = "All AI providers are currently busy. Please try again in a minute."
            break
        retry_count += 1

        llm, key_index = get_llm(key_index)
        agent_executor = _build_agent_executor(llm, langchain_tools, sections)
        chat_history = _build_chat_history(conversation_history, fallback_hint, sections)

        try:
            result = agent_executor.invoke({
                "input": user_input,
                "chat_history": chat_history,
            })
            reply = result["output"]

            if _is_garbage_reply(reply):
                try:
                    result2 = agent_executor.invoke({
                        "input": user_input,
                        "chat_history": chat_history,
                    })
                    reply = result2["output"]
                except Exception:
                    reply = "I ran into an issue processing your request. Please try again."

            last_working_key = key_index
            break

        except Exception as e:
            next_key, should_break, fallback_hint, error_reply = _handle_exception(
                str(e), key_index, groq_keys, has_mistral,
                fallback_hint, get_tools_called, log_llm_fallback,
                log_error, user_input,
            )
            if should_break:
                reply = error_reply
                break
            key_index = next_key

    if reply is None:
        reply = "I ran into an issue processing your request. Please try again."

    duration = time.time() - start
    log_response(user_input, reply, duration, tools_called=get_tools_called())
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, conversation_history, last_working_key


# -- Streaming Agent -----------------------------------------------------------
def run_agent_stream(
    user_input: str,
    conversation_history: list[dict],
    start_key_index: int = 0,
    creds=None,
    github_token: dict = None,
):
    from agent.logger import log_request, log_response, log_error, log_llm_fallback
    from agent.tool_executor import reset_tools_called, get_tools_called

    user_input = _preprocess_input(user_input)
    log_request(user_input)
    reset_tools_called()
    start = time.time()

    sections = _detect_sections(user_input)
    all_tools, tool_server_map = fetch_tools()
    langchain_tools = build_langchain_tools(
        all_tools, tool_server_map,
        creds=creds,
        github_token=github_token,
    )
    groq_keys = _get_groq_keys()
    has_mistral = bool(os.getenv("MISTRAL_API_KEY"))

    key_index = start_key_index
    last_working_key = start_key_index
    reply = None
    fallback_hint = None
    retry_count = 0
    max_retries = 6

    while True:
        if retry_count >= max_retries:
            reply = "All AI providers are currently busy. Please try again in a minute."
            yield reply, None, None
            break
        retry_count += 1

        llm, key_index = get_llm(key_index)
        agent_executor = _build_agent_executor(llm, langchain_tools, sections)
        chat_history = _build_chat_history(conversation_history, fallback_hint, sections)

        try:
            full_reply = []

            for chunk in agent_executor.stream({
                "input": user_input,
                "chat_history": chat_history,
            }):
                if "output" in chunk:
                    token = chunk["output"]
                    if token:
                        full_reply.append(token)
                        yield token, None, None

            reply = "".join(full_reply)

            if _is_garbage_reply(reply):
                # Fall to next LLM — same LLM will produce same garbage
                full_reply = []
                tools_done = get_tools_called()
                if tools_done:
                    fallback_hint = (
                        f"[System: Previous attempt already called tools: {', '.join(tools_done)}. "
                        f"Do NOT call these tools again. Use results already obtained.]"
                    )
                if key_index >= 0:
                    next_key = key_index + 1
                    if next_key >= len(groq_keys):
                        next_key = -2 if has_mistral else -1
                elif key_index == -2:
                    next_key = -1
                else:
                    next_key = key_index
                log_llm_fallback(key_index, "garbage reply detected")
                key_index = next_key
                retry_count += 1
                if retry_count >= max_retries:
                    reply = "I ran into an issue processing your request. Please try again."
                    yield reply, None, None
                    break
                continue

            if not reply:
                reply = "I ran into an issue processing your request. Please try again."

            last_working_key = key_index
            break

        except Exception as e:
            import traceback
            traceback.print_exc()
            log_error(f"EXCEPTION [{type(e).__name__}]: {str(e)}")
            next_key, should_break, fallback_hint, error_reply = _handle_exception(
                str(e), key_index, groq_keys, has_mistral,
                fallback_hint, get_tools_called, log_llm_fallback,
                log_error, user_input,
            )
            if should_break:
                reply = error_reply or "I ran into an issue processing your request. Please try again."
                yield reply, None, None
                break
            key_index = next_key

    if reply is None:
        reply = "I ran into an issue processing your request. Please try again."
        yield reply, None, None

    duration = time.time() - start
    log_response(user_input, reply, duration, tools_called=get_tools_called())
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": reply})

    yield None, conversation_history, last_working_key



# -- CLI Mode ------------------------------------------------------------------
def start_agent():
    print("Multi MCP Agent ready! Type exit to quit.\n")
    conversation_history = []
    working_key = 0
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        reply, conversation_history, working_key = run_agent(
            user_input, conversation_history, working_key
        )
        print(f"\nAssistant: {reply}\n")