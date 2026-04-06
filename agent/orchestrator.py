import os
import re
import time
import threading
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from typing import Optional
from pydantic import Field, create_model
from agent.tool_schema import fetch_tools
from agent.tool_executor import execute_tool, get_last_tool_result
from agent.prompt import get_prompt
from agent.logger import log_llm_selected
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


# ── Section Router ────────────────────────────────────────────────────────────
def _detect_sections(user_input: str) -> list:
    lower = user_input.lower()

    gmail_words = [
        "email", "mail", "inbox", "draft", "send", "reply", "forward",
        "attachment", "schedule", "unread", "read", "mark", "trash",
        "restore", "cc", "bcc", "subject", "compose", "gmail",
    ]
    drive_words = [
        "file", "folder", "drive", "upload", "download", "document",
        "sheet", "spreadsheet", "slides", "pdf", "docx", "xlsx", "pptx",
        "txt", "csv", "create", "rename", "move", "copy", "share",
        "recent", "search", "find", "open", "summarize",
    ]
    github_words = [
        "repo", "repository", "repositories", "issue", "issues",
        "pull request", "pull requests", "pr", "prs", "github", "branch", "branches",
        "commit", "readme", "code", "bug", "feature", "merge",
        "project", "backlog", "ready", "in progress", "in review", "done",
        "board", "column", "move issue", "project board",
        "assign", "label", "start date", "end date", "starting date", "ending date",
    ]

    sections = []
    if any(w in lower for w in gmail_words):
        sections.append("gmail")
    if any(w in lower for w in drive_words):
        sections.append("drive")
    if any(w in lower for w in github_words):
        sections.append("github")
    if not sections:
        sections = ["gmail", "drive", "github"]
    return sections


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
# GROQ_MODEL env var overrides default.
# llama-3.3-70b-versatile handles this app's large prompts.
# Do NOT use llama-3.1-8b-instant — its 6000 TPM is too small.
_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


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
                model=_GROQ_MODEL,
                temperature=0,
                max_retries=1,
            ), key_index

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
                    max_retries=1,
                ), -2
            except Exception:
                pass

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
    _SHORT_DESC = {
        # Gmail
        "list_emails":                "List inbox emails. max_results, query, page_token optional.",
        "read_email":                 "Read full email by id.",
        "send_email":                 "Send email. to required. subject/body empty triggers ask flow.",
        "reply_to_email":             "Reply to email by id. body required.",
        "forward_email":              "Forward email. email_id, to_email required.",
        "delete_email":               "Move email to trash. confirmed='true' required.",
        "restore_email":              "Restore email from trash. confirmed='true' required.",
        "search_emails":              "Search emails by Gmail query.",
        "send_email_with_attachment": "Send email with local file attachment. file_path required.",
        "get_email_attachments":      "List attachments in an email by id.",
        "download_email_attachment":  "Download attachment. email_id, attachment_id, filename required.",
        "list_unread_emails":         "List unread inbox emails.",
        "mark_as_read":               "Mark emails as read. email_ids: exact id or comma-separated.",
        "mark_as_unread":             "Mark emails as unread. email_ids: exact id or comma-separated.",
        "save_draft":                 "Save email as draft.",
        "list_drafts":                "List all saved drafts.",
        "update_draft":               "Update draft. Call list_drafts first to get draft_id.",
        "send_draft":                 "Send a draft. Call list_drafts first to get draft_id.",
        "delete_draft":               "Delete a draft. confirmed='true' required.",
        "schedule_email":             "Schedule email. to, send_at required.",
        "list_scheduled_emails":      "List all scheduled emails.",
        "cancel_scheduled_email":     "Cancel a scheduled email by job_id.",
        "get_user_timezone":          "Get system timezone.",
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
        "get_file_info":              "Get file details.",
        "upload_file":                "Upload local file to Drive. file_path required.",
        "download_file":              "Download Drive file to local machine.",
        "list_recent_files":          "List recently modified files.",
        "share_file":                 "Share file with email. role: reader/commenter/writer.",
        # GitHub
        "list_repos":                 "List user GitHub repos. limit optional.",
        "create_repo":                "Create GitHub repo. name required.",
        "search_repos":               "Search GitHub repos. query required.",
        "list_repo_files":            "List files in repo. repo required (owner/repo or short name).",
        "read_file_from_repo":        "Read file from repo. repo and file_path required.",
        "list_issues":                "List issues in repo. repo required.",
        "create_issue":               "Create issue. repo and title required.",
        "read_issue":                 "Read specific issue. repo and issue_number required.",
        "list_branches":              "List all branches. repo required.",
        "list_pull_requests":         "List pull requests. repo required.",
        "create_pull_request":        "Create pull request. repo, title, head required.",
        "merge_pull_request":         "Merge a pull request. repo and pull_number required. merge_method: merge/squash/rebase (default merge).",
        "create_branch":              "Create new branch. repo, branch_name required.",
        "list_projects":              "List GitHub Projects v2. repo optional.",
        "create_project":             "Create a new GitHub Project v2. title required. repo optional to link it.",
        "get_project_columns":        "Get project board columns. project_id required (title or PVT_...).",
        "add_issue_to_project":       "Add issue to project board. project_id and issue_url required.",
        "move_issue_to_column":       "Move issue to column. project_id, item_id, column_name required.",
        "update_project_issue_fields": "Update start_date/end_date on project item. project_id, item_id required.",
        "list_project_issues":        "List all issues on project board with item_id. project_id required.",
        "update_project_issue_by_title": "Update/move project issue by title (no item_id needed). project_id, issue_title required. Can set assignee, labels, dates, move to column.",
        "create_project_issue":       "Create issue + add to project Backlog. repo, project_id, title required.",
    }

    _GH_TOOLS = {
        "list_repos", "create_repo", "search_repos", "list_repo_files",
        "read_file_from_repo", "list_branches", "list_issues", "create_issue",
        "read_issue", "list_pull_requests", "create_pull_request", "merge_pull_request",
        "create_branch", "list_projects", "create_project", "get_project_columns",
        "add_issue_to_project", "move_issue_to_column", "update_project_issue_fields",
        "list_project_issues", "update_project_issue_by_title", "create_project_issue",
    }

    langchain_tools = []
    for tool in all_tools:
        func_info   = tool["function"]
        tool_name   = func_info["name"]
        tool_desc   = _SHORT_DESC.get(tool_name, func_info["description"][:120])
        server_name = tool_server_map.get(tool_name, "drive")
        properties  = func_info.get("parameters", {}).get("properties", {})

        fields = {}
        for prop_name, prop_info in properties.items():
            fields[prop_name] = (Optional[str], Field(
                default="",
                description=prop_info.get("description", "")
            ))
        if not fields:
            fields["dummy"] = (Optional[str], Field(default=""))

        ArgsModel = create_model(f"{tool_name}Args", **fields)

        def make_tool_fn(t_name, s_name, user_creds, gh_token, is_gh):
            def tool_fn(**kwargs) -> str:
                kwargs.pop("dummy", None)
                if not is_gh:
                    kwargs = {k: v for k, v in kwargs.items() if v not in ("", None)}
                return execute_tool(
                    s_name, t_name, kwargs,
                    creds=user_creds,
                    github_token=gh_token,
                )
            tool_fn.__name__ = t_name
            tool_fn.__doc__  = tool_desc
            return tool_fn

        langchain_tool = StructuredTool.from_function(
            func=make_tool_fn(tool_name, server_name, creds, github_token, tool_name in _GH_TOOLS),
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
        max_iterations=10,
        early_stopping_method="generate",
    )


def _build_chat_history(conversation_history, fallback_hint=None, sections=None):
    if sections is None:
        sections = ["gmail", "drive", "github"]
    limit = 2 if len(sections) >= 3 else 4
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
    if any(x in lower for x in [
        "rate_limit", "rate limit", "429", "rate_limit_exceeded",
        "tokens per day", "tokens per minute", "requests per minute",
        "too many requests", "413",
    ]):
        return "rate_limit"
    if "too large" in lower and "token" in lower:
        return "token_limit"
    if any(x in lower for x in [
        "invalid_api_key", "invalid api key", "401", "authentication", "unauthorized",
    ]):
        return "auth_error"
    if any(x in lower for x in [
        "tool_use_failed", "failed_generation", "pydantic",
        "serializ", "unexpectedvalue", "tool call validation",
    ]):
        return "parse_fail"
    if "400" in error_str and (
        "failed to call a function" in lower or "decommissioned" in lower
    ):
        return "groq_400"
    return "unknown"


def _get_groq_keys() -> list:
    return [k for k in [
        os.getenv("GROQ_API_KEY_1"),
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ] if k]


# ── Result Formatters ─────────────────────────────────────────────────────────
def _parse_tool_result(raw: str) -> list:
    import json
    if not raw or not raw.strip():
        return []
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
    repos = _parse_tool_result(raw)
    if not repos:
        return ""
    NL = "\n"
    if len(repos) == 1 and "message" in repos[0]:
        return f"🐙 {repos[0]['message']}"
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
    issues = _parse_tool_result(raw)
    if not issues:
        return ""
    NL = "\n"
    if len(issues) == 1 and "message" in issues[0]:
        return f"🐛 {issues[0]['message']}"
    real = [i for i in issues if i.get("number") and i.get("title")]
    if not real:
        return "🐛 No issues found."
    lines = ["🐛 Issues:" + NL]
    for i, iss in enumerate(real, 1):
        lines.append(f"{i}. **#{iss.get('number')}** {iss.get('title')} | by {iss.get('author', '')}")
    lines.append(NL + "Reply with a number to read the full issue.")
    return NL.join(lines)


def _format_prs(raw: str) -> str:
    prs = _parse_tool_result(raw)
    if not prs:
        return ""
    NL = "\n"
    if len(prs) == 1 and "message" in prs[0]:
        return f"🔀 {prs[0]['message']}"
    real = [p for p in prs if p.get("number") and p.get("title")]
    if not real:
        return "🔀 No pull requests found."
    lines = ["🔀 Pull requests:" + NL]
    for i, pr in enumerate(real, 1):
        lines.append(f"{i}. **#{pr.get('number')}** {pr.get('title')} | {pr.get('head')} → {pr.get('base')}")
    return NL.join(lines)


def _format_branches(raw: str) -> str:
    branches = _parse_tool_result(raw)
    if not branches:
        return ""
    NL = "\n"
    lines = ["🌿 Branches:" + NL]
    for i, b in enumerate(branches, 1):
        default = " (default)" if b.get("default") else ""
        lines.append(f"{i}. **{b.get('name', '')}**{default}")
    return NL.join(lines)


def _format_search_repos(raw: str) -> str:
    repos = _parse_tool_result(raw)
    if not repos:
        return ""
    NL = "\n"
    if len(repos) == 1 and "message" in repos[0]:
        return f"🔍 {repos[0]['message']}"
    lines = ["🔍 Search results:" + NL]
    for i, r in enumerate(repos, 1):
        lines.append(f"{i}. **{r.get('name')}** | ⭐ {r.get('stars', 0)} | {r.get('language') or '—'}")
    return NL.join(lines)


def _format_project_issues(raw: str) -> str:
    items = _parse_tool_result(raw)
    if not items:
        return ""
    NL = "\n"
    if len(items) == 1 and "message" in items[0]:
        return f"📋 {items[0]['message']}"
    lines = ["📋 **Issues on Project Board**:" + NL]
    for i, item in enumerate(items, 1):
        num       = item.get("number", "")
        title     = item.get("title", "")
        status    = item.get("status", "")
        labels    = ", ".join(item.get("labels", [])) or "—"
        assignees = ", ".join(item.get("assignees", [])) or "—"
        start     = item.get("start_date", "") or "—"
        end       = item.get("end_date", "") or "—"
        item_id   = item.get("item_id", "")
        lines.append(f"{i}. **{title}** (Issue #{num}) | Status: {status} | Assignees: {assignees}")
        lines.append(f"   🏷️ Labels: {labels} | 📅 {start} → {end}")
        if item_id:
            lines.append(f"   <!-- item_id:{item_id} -->")
        lines.append("")
    lines.append("Which issue would you like to move or update?")
    return NL.join(lines)


def _format_projects(raw: str) -> str:
    projects = _parse_tool_result(raw)
    if not projects:
        return ""
    NL = "\n"
    if len(projects) == 1 and "message" in projects[0]:
        return f"📋 {projects[0]['message']}"
    lines = ["📋 Your GitHub Projects:" + NL]
    for i, p in enumerate(projects, 1):
        lines.append(f"{i}. **{p.get('title')}** (Project #{p.get('number')})")
    lines.append(NL + "Reply with a project name to view its board.")
    return NL.join(lines)


def _format_file_content(raw: str) -> str:
    """
    Format read_file_from_repo result — ALWAYS inside a code block.
    Prevents # comment lines rendering as giant Markdown headers.
    """
    import json
    if not raw or not raw.strip():
        return ""
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            data = data[0]
        if isinstance(data, dict):
            if data.get("status") == "error":
                return f"❌ {data.get('message', 'Could not read file.')}"
            content   = data.get("content", "")
            file_name = data.get("file", "")
            repo      = data.get("repo", "")
            if content:
                ext  = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
                lang_map = {
                    "py": "python", "js": "javascript", "ts": "typescript",
                    "md": "markdown", "json": "json", "yaml": "yaml", "yml": "yaml",
                    "sh": "bash", "html": "html", "css": "css", "rb": "ruby",
                    "java": "java", "go": "go", "rs": "rust", "cpp": "cpp",
                    "c": "c", "sql": "sql",
                }
                lang   = lang_map.get(ext, "")
                header = f"📄 **{repo}/{file_name}**\n\n" if repo and file_name else ""
                return f"{header}```{lang}\n{content}\n```"
    except Exception:
        pass
    return ""


# ── Smart Reply ───────────────────────────────────────────────────────────────
def _smart_reply_from_tools(tools_done: list, user_input: str) -> str:
    import json
    from agent.tool_executor import get_last_tool_result

    _GH_TOOLS = {
        "list_repos", "create_repo", "search_repos", "list_repo_files",
        "read_file_from_repo", "list_issues", "create_issue", "read_issue",
        "list_pull_requests", "create_pull_request", "merge_pull_request",
        "list_branches", "create_branch", "list_projects", "create_project",
        "get_project_columns", "add_issue_to_project", "move_issue_to_column",
        "update_project_issue_fields", "list_project_issues",
        "update_project_issue_by_title", "create_project_issue",
    }

    # Check GitHub not connected
    for t_name in tools_done:
        if t_name in _GH_TOOLS:
            res = str(get_last_tool_result(t_name))
            if "No GitHub credentials found" in res:
                return "🐙 GitHub is not connected yet.\n\nPlease click **Connect GitHub** in the menu (☰) to link your account first!"

    tools_set = set(tools_done)

    def get_tool_error(t_name: str) -> str:
        raw = get_last_tool_result(t_name)
        if not raw:
            return ""
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                return ""
            msg = data.get("message", "Unknown error.")
            if "No commits between" in msg:
                return "No changes between these branches — pull request not needed."
            return msg
        except Exception:
            return ""

    # ── Upload/Download ───────────────────────────────────────────────────────
    if "upload_file" in tools_set:
        return "✅ File uploaded to **Google Drive** successfully! 📁"
    if "download_email_attachment" in tools_set:
        return "✅ Attachment downloaded to your Downloads folder successfully!"

    # ── create_project ────────────────────────────────────────────────────────
    if "create_project" in tools_set:
        raw = get_last_tool_result("create_project")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                title = data.get("title", "")
                url   = data.get("url", "")
                note  = data.get("note", "")
                lines = [f"✅ Project **{title}** created successfully! 📋"]
                if url:
                    lines.append(f"\n🔗 **URL:** {url}")
                if data.get("repo_linked"):
                    lines.append(f"📦 **Linked to:** {data['repo_linked']}")
                if note:
                    lines.append(f"\n💡 {note}")
                return "\n".join(lines)
            return f"❌ {data.get('message', 'Failed to create project.')}"
        except Exception:
            return "✅ Project created successfully! 📋"

    # ── create_project_issue ──────────────────────────────────────────────────
    if "create_project_issue" in tools_set:
        raw = get_last_tool_result("create_project_issue")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                num   = data.get("number", "")
                url   = data.get("url", "")
                col   = data.get("column", "Backlog")
                lines = [f"✅ Issue **#{num}** created and added to project board in **{col}**! 🐛"]
                if url:
                    lines.append(f"\n🔗 **URL:** {url}")
                if data.get("assignee"):
                    lines.append(f"👤 **Assigned to:** @{data['assignee']}")
                if data.get("start_date"):
                    lines.append(f"📅 **Start:** {data['start_date']}")
                if data.get("end_date"):
                    lines.append(f"📅 **End:** {data['end_date']}")
                return "\n".join(lines)
            return f"❌ {data.get('message', 'Unknown error.')}"
        except Exception:
            return "✅ Issue created and added to project board! 🐛"

    # ── update_project_issue_by_title ─────────────────────────────────────────
    if "update_project_issue_by_title" in tools_set:
        raw = get_last_tool_result("update_project_issue_by_title")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                num     = data.get("issue_num", "")
                updates = data.get("updates", [])
                if updates:
                    return f"✅ Issue #{num} updated: {', '.join(updates)} 📋"
                return f"✅ Issue #{num} updated successfully! 📋"
            err = data.get("message", "")
            if not err or "project_id" in err.lower() or "issue_title" in err.lower():
                return (
                    "⚠️ All AI providers are currently busy.\n\n"
                    "Please try again — tell me:\n"
                    "1. The **project board name**\n"
                    "2. The **issue title** to update\n"
                    "3. What to update (assignee, labels, dates, column)"
                )
            return f"❌ {err}"
        except Exception:
            pass

    # ── move_issue_to_column ──────────────────────────────────────────────────
    if "move_issue_to_column" in tools_set:
        raw = get_last_tool_result("move_issue_to_column")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                return f"✅ Issue moved to **{data.get('column', '')}** successfully! 📋"
            err = data.get("message", "")
            if not err or "item_id" in err.lower() or "project_id" in err.lower():
                return (
                    "⚠️ All AI providers are currently busy.\n\n"
                    "Please try again with:\n"
                    "**move the '[issue title]' issue in [project name] to [column]**\n\n"
                    "Example: *move the 'Implement user auth' issue in test project to Ready*"
                )
            return f"❌ {err}"
        except Exception:
            pass

    # ── update_project_issue_fields ───────────────────────────────────────────
    if "update_project_issue_fields" in tools_set:
        raw = get_last_tool_result("update_project_issue_fields")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                return f"✅ Project fields updated: {', '.join(data.get('updated', []))} 📅"
            err = data.get("message", "")
            if not err or "item_id" in err.lower():
                return "⚠️ All AI providers are busy. Please try again."
            return f"❌ {err}"
        except Exception:
            pass

    # ── list_project_issues ───────────────────────────────────────────────────
    if "list_project_issues" in tools_set:
        raw = get_last_tool_result("list_project_issues")
        fmt = _format_project_issues(raw)
        return fmt if fmt else "📋 Project issues retrieved — please try again."

    if "list_projects" in tools_set:
        raw = get_last_tool_result("list_projects")
        fmt = _format_projects(raw)
        return fmt if fmt else "📋 Projects retrieved — please try again."

    # ── merge_pull_request ────────────────────────────────────────────────────
    if "merge_pull_request" in tools_set:
        raw = get_last_tool_result("merge_pull_request")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                num    = data.get("issue_num", "")
                method = data.get("method", "merge")
                head   = data.get("head", "")
                base   = data.get("base", "")
                lines  = [f"✅ Pull request **#{num}** merged via **{method}**! 🔀"]
                if head and base:
                    lines.append(f"\n📌 **{head}** → **{base}**")
                return "\n".join(lines)
            return f"❌ {data.get('message', 'Merge failed.')}"
        except Exception:
            return "✅ Pull request merged successfully! 🔀"

    # ── list repos / issues / PRs ─────────────────────────────────────────────
    if "list_repos" in tools_set:
        raw = get_last_tool_result("list_repos")
        fmt = _format_repos(raw)
        return fmt if fmt else "🐙 Repos retrieved — please try again."

    if "list_issues" in tools_set:
        raw = get_last_tool_result("list_issues")
        fmt = _format_issues(raw)
        return fmt if fmt else "🐛 Issues retrieved — please try again."

    if "list_pull_requests" in tools_set:
        raw = get_last_tool_result("list_pull_requests")
        fmt = _format_prs(raw)
        return fmt if fmt else "🔀 PRs retrieved — please try again."

    if "search_repos" in tools_set:
        raw = get_last_tool_result("search_repos")
        fmt = _format_search_repos(raw)
        return fmt if fmt else "🔍 Results retrieved — please try again."

    if "list_branches" in tools_set and "create_pull_request" not in tools_set:
        raw = get_last_tool_result("list_branches")
        fmt = _format_branches(raw)
        pr_kw = ["pull request", "create pr", "open pr", "pull-request"]
        if any(kw in user_input.lower() for kw in pr_kw) and fmt:
            return (
                f"{fmt}\n\n"
                "Please provide:\n\n"
                "1. **Title** — PR title?\n"
                "2. **Head branch** — which branch has your changes?\n"
                "3. **Base branch** — merge into which? (default: main)\n"
                "4. **Description** — (or say 'skip')\n\n"
                "Reply with all details in one message! 🔀"
            )
        return fmt if fmt else "🌿 Branches retrieved — please try again."

    # ── read_file_from_repo — FIX: always return content in code block ────────
    if "read_file_from_repo" in tools_set:
        raw = get_last_tool_result("read_file_from_repo")
        fmt = _format_file_content(raw)
        return fmt if fmt else "✅ File retrieved — please try again to see the content."

    if "list_repo_files" in tools_set:
        raw   = get_last_tool_result("list_repo_files")
        items = _parse_tool_result(raw)
        if not items:
            return "📁 No files found."
        if len(items) == 1 and "message" in items[0]:
            return f"📁 {items[0]['message']}"
        NL    = "\n"
        lines = ["📁 Files in repository:" + NL]
        for i, item in enumerate(items, 1):
            icon = "📁" if item.get("type") == "dir" else "📄"
            size = f" ({item['size']} bytes)" if item.get("size") else ""
            lines.append(f"{i}. {icon} **{item.get('name', '')}**{size}")
        return NL.join(lines)

    if "read_issue" in tools_set:
        raw = get_last_tool_result("read_issue")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "error":
                return f"❌ {data.get('message', 'Could not read issue.')}"
            return (
                f"🐛 **Issue #{data.get('number', '')}: {data.get('title', '')}**\n\n"
                f"**Author:** {data.get('author', '')} | **State:** {data.get('state', '')} | "
                f"**Created:** {data.get('created', '')}\n\n"
                f"{data.get('body', '')}"
            )
        except Exception:
            pass

    # ── create tools ──────────────────────────────────────────────────────────
    def get_tool_err(t_name: str) -> str:
        raw = get_last_tool_result(t_name)
        if not raw:
            return ""
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                return ""
            msg = data.get("message", "Unknown error.")
            if "No commits between" in msg:
                return "No changes between these branches — pull request not needed."
            return msg
        except Exception:
            return ""

    if "create_repo" in tools_set:
        err = get_tool_err("create_repo")
        if not err:
            raw = get_last_tool_result("create_repo")
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    data = data[0]
                return (
                    f"✅ Repository **{data.get('name', '')}** created successfully! 🐙\n\n"
                    f"🔗 **URL:** {data.get('url', '')}\n"
                    f"🔒 **Visibility:** {'Private' if data.get('private') else 'Public'}\n"
                    f"📄 **README:** {'Yes' if data.get('auto_init') else 'No'}"
                )
            except Exception:
                return "✅ GitHub repository created successfully! 🐙"
        return f"❌ {err}"

    if "create_issue" in tools_set:
        err = get_tool_err("create_issue")
        if not err:
            raw = get_last_tool_result("create_issue")
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    data = data[0]
                lines = [f"✅ Issue **#{data.get('number', '')}** created successfully! 🐛"]
                if data.get("url"):
                    lines.append(f"\n🔗 **URL:** {data['url']}")
                if data.get("assignee"):
                    lines.append(f"👤 **Assigned to:** @{data['assignee']}")
                if data.get("start_date"):
                    lines.append(f"📅 **Start:** {data['start_date']}")
                if data.get("end_date"):
                    lines.append(f"📅 **End:** {data['end_date']}")
                return "\n".join(lines)
            except Exception:
                return "✅ GitHub issue created successfully! 🐛"
        return f"❌ {err}"

    if "create_pull_request" in tools_set:
        err = get_tool_err("create_pull_request")
        if not err:
            raw = get_last_tool_result("create_pull_request")
            try:
                data = json.loads(raw)
                if isinstance(data, list):
                    data = data[0]
                num  = data.get("number", "")
                repo = data.get("repo", "")
                return (
                    f"✅ Pull request **#{num}** created successfully! 🔀\n\n"
                    f"🔗 **URL:** {data.get('url', '')}\n"
                    f"📌 **{data.get('head', '')}** → **{data.get('base', '')}**\n\n"
                    f"Would you like to **merge** this PR? Say: *merge PR #{num} in {repo}*"
                )
            except Exception:
                return "✅ Pull request created successfully! 🔀"
        return f"❌ {err}"

    if "create_branch" in tools_set and "create_pull_request" not in tools_set:
        err = get_tool_err("create_branch")
        return "✅ Branch created successfully! 🌿" if not err else f"❌ {err}"

    # ── Email/Drive fallbacks ─────────────────────────────────────────────────
    last = tools_done[-1] if tools_done else ""
    if "send_email" in last:
        return "✅ Email sent successfully! 📧"
    if "download_file" in last:
        return "✅ File downloaded successfully!"
    if "search" in last or "list" in last:
        return ""
    return "✅ Done!"


# ── Exception Handler ─────────────────────────────────────────────────────────
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
    error_type = _classify_error(error_str)

    if error_type == "auth_error":
        log_error_fn(error_str, context=f"user_input={user_input!r}")
        return key_index, True, fallback_hint, "❌ Authentication failed. Please check your API keys."

    if error_type in ("rate_limit", "token_limit"):
        # If tools already ran this request — return smart reply immediately
        tools_done = get_tools_called_fn()
        if tools_done:
            log_llm_fallback_fn(key_index + 1 if key_index >= 0 else key_index, error_str)
            smart = _smart_reply_from_tools(tools_done, user_input)
            if smart:
                return key_index, True, fallback_hint, smart
        # No tools called — try next key
        if key_index >= 0:
            next_key = key_index + 1
            if next_key < len(groq_keys):
                log_llm_fallback_fn(key_index + 1, error_str)
                return next_key, False, fallback_hint, None
        # Groq exhausted — try Mistral
        if has_mistral and key_index != -2:
            log_llm_fallback_fn(key_index + 1 if key_index >= 0 else key_index, error_str)
            return -2, False, fallback_hint, None
        # Mistral exhausted — try Ollama
        if key_index != -1:
            return -1, False, fallback_hint, None
        log_error_fn(error_str, context=f"user_input={user_input!r}")
        return key_index, True, fallback_hint, "All AI providers are currently busy. Please try again in a minute."

    if key_index == -2:
        tools_done = get_tools_called_fn()
        if tools_done:
            smart = _smart_reply_from_tools(tools_done, user_input)
            if smart:
                return key_index, True, fallback_hint, smart
        log_llm_fallback_fn(-2, error_str)
        return -1, False, fallback_hint, None

    if key_index == -1:
        log_error_fn(error_str, context=f"user_input={user_input!r}")
        clean = error_str.split('|')[0].strip() if '|' in error_str else error_str
        return key_index, True, fallback_hint, f"❌ {clean}"

    if error_type in ("parse_fail", "groq_400") and key_index >= 0:
        log_llm_fallback_fn(key_index + 1, error_str)
        tools_done = get_tools_called_fn()
        if tools_done:
            fallback_hint = (
                f"[System: Previous attempt already called tools: {', '.join(tools_done)}. "
                "Do NOT call these tools again. Use results already obtained.]"
            )
        next_key = key_index + 1
        if next_key >= len(groq_keys):
            next_key = -2 if has_mistral else -1
        return next_key, False, fallback_hint, None

    if error_type != "unknown" and key_index >= 0:
        log_llm_fallback_fn(key_index + 1, error_str)
        next_key = key_index + 1
        if next_key >= len(groq_keys):
            next_key = -2 if has_mistral else -1
        return next_key, False, fallback_hint, None

    log_error_fn(error_str, context=f"user_input={user_input!r}")
    return key_index, True, fallback_hint, f"Something went wrong: {error_str[:200]}"


# ── Garbage Detection ─────────────────────────────────────────────────────────
_GARBAGE_MARKERS = [
    "type': 'reference'", "type': 'text'", "{'type':", '{"type":',
    "responded: get_email_attachments", "responded: search_emails",
    "responded: download", "responded: upload", "responded: list_emails",
    "responded: read_email",
    'search_emails": {', 'list_emails": {', 'get_email_attachments": {',
    'download_email_attachment": {', 'upload_file": {', 'download_file": {',
    'search_files": {', 'read_file": {', 'send_email": {',
    'list_repos": {', 'create_repo": {', 'search_repos": {',
    'list_repo_files": {', 'read_file_from_repo": {',
    'list_issues": {', 'create_issue": {', 'read_issue": {',
    'list_pull_requests": {', 'create_pull_request": {', 'merge_pull_request": {',
    'list_branches": {', 'list_projects": {', 'create_project": {',
    'get_project_columns": {', 'add_issue_to_project": {',
    'move_issue_to_column": {', 'update_project_issue_fields": {',
    'list_project_issues": {', 'update_project_issue_by_title": {',
    'create_project_issue": {',
    '"function":', '"args":',
    'the JSON object', 'the function call',
]


def _is_garbage_reply(reply: str) -> bool:
    if not reply:
        return False
    if "pydanticserializat" in reply.lower():
        return True
    if reply.strip().startswith("[") and "{'type'" in reply:
        return True
    if ">![]({"  in reply or '"file_path"' in reply:
        return True
    for marker in _GARBAGE_MARKERS:
        if marker in reply:
            return True
    if '"function"' in reply and '"args"' in reply:
        return True
    return False


# ── Standard Agent ────────────────────────────────────────────────────────────
def run_agent(
    user_input: str,
    conversation_history: list,
    start_key_index: int = 0,
    creds=None,
    github_token: dict = None,
) -> tuple:
    from agent.logger import log_request, log_response, log_error, log_llm_fallback
    from agent.tool_executor import reset_tools_called, get_tools_called

    user_input = _preprocess_input(user_input)
    log_request(user_input)
    reset_tools_called()
    start = time.time()

    sections        = _detect_sections(user_input)
    all_tools, tsm  = fetch_tools()
    lt              = build_langchain_tools(all_tools, tsm, creds=creds, github_token=github_token)
    groq_keys       = _get_groq_keys()
    has_mistral     = bool(os.getenv("MISTRAL_API_KEY"))
    key_index       = start_key_index
    last_working    = start_key_index
    reply           = None
    fallback_hint   = None
    retry_count     = 0

    while True:
        if retry_count >= 6:
            reply = "All AI providers are currently busy. Please try again in a minute."
            break
        retry_count += 1

        llm, key_index = get_llm(key_index)
        ae = _build_agent_executor(llm, lt, sections)
        ch = _build_chat_history(conversation_history, fallback_hint, sections)

        try:
            result = ae.invoke({"input": user_input, "chat_history": ch})
            reply  = result["output"]

            for t in get_tools_called():
                if "No GitHub credentials found" in str(get_last_tool_result(t)):
                    reply = "🐙 GitHub is not connected yet.\n\nPlease click **Connect GitHub** in the menu (☰) to link your account first!"
                    break

            if _is_garbage_reply(reply):
                try:
                    reply = ae.invoke({"input": user_input, "chat_history": ch})["output"]
                except Exception:
                    reply = "I ran into an issue. Please try again."

            last_working = key_index
            break

        except Exception as e:
            nk, sb, fallback_hint, er = _handle_exception(
                str(e), key_index, groq_keys, has_mistral,
                fallback_hint, get_tools_called, log_llm_fallback, log_error, user_input)
            if sb:
                reply = er
                break
            key_index = nk

    reply = reply or "I ran into an issue. Please try again."
    log_response(user_input, reply, time.time() - start, tools_called=get_tools_called())
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": reply})
    return reply, conversation_history, last_working


# ── Streaming Agent ───────────────────────────────────────────────────────────
def run_agent_stream(
    user_input: str,
    conversation_history: list,
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

    sections      = _detect_sections(user_input)
    all_tools, tsm = fetch_tools()
    lt             = build_langchain_tools(all_tools, tsm, creds=creds, github_token=github_token)
    groq_keys      = _get_groq_keys()
    has_mistral    = bool(os.getenv("MISTRAL_API_KEY"))
    key_index      = start_key_index
    last_working   = start_key_index
    reply          = None
    fallback_hint  = None
    retry_count    = 0

    while True:
        if retry_count >= 6:
            reply = "All AI providers are currently busy. Please try again in a minute."
            yield reply, None, None
            break
        retry_count += 1

        llm, key_index = get_llm(key_index)
        ae = _build_agent_executor(llm, lt, sections)
        ch = _build_chat_history(conversation_history, fallback_hint, sections)

        try:
            full = []
            for chunk in ae.stream({"input": user_input, "chat_history": ch}):
                if "output" in chunk and chunk["output"]:
                    full.append(chunk["output"])
                    yield chunk["output"], None, None

            reply = "".join(full)

            for t in get_tools_called():
                if "No GitHub credentials found" in str(get_last_tool_result(t)):
                    reply = "🐙 GitHub is not connected yet.\n\nPlease click **Connect GitHub** in the menu (☰) to link your account first!"
                    break

            if _is_garbage_reply(reply):
                full       = []
                tools_done = get_tools_called()
                if tools_done:
                    fallback_hint = (
                        f"[System: Previous attempt called tools: {', '.join(tools_done)}. "
                        "Do NOT call these again. Use results already obtained.]"
                    )
                if key_index >= 0:
                    nk = key_index + 1
                    if nk >= len(groq_keys):
                        nk = -2 if has_mistral else -1
                elif key_index == -2:
                    nk = -1
                else:
                    nk = key_index
                log_llm_fallback(key_index, "garbage reply")
                key_index   = nk
                retry_count += 1
                if retry_count >= 6:
                    reply = "I ran into an issue. Please try again."
                    yield reply, None, None
                    break
                continue

            reply        = reply or "I ran into an issue. Please try again."
            last_working = key_index
            break

        except Exception as e:
            import traceback
            traceback.print_exc()
            log_error(f"EXCEPTION [{type(e).__name__}]: {str(e)}")
            nk, sb, fallback_hint, er = _handle_exception(
                str(e), key_index, groq_keys, has_mistral,
                fallback_hint, get_tools_called, log_llm_fallback, log_error, user_input)
            if sb:
                reply = er or "I ran into an issue. Please try again."
                yield reply, None, None
                break
            key_index = nk

    reply = reply or "I ran into an issue. Please try again."
    log_response(user_input, reply, time.time() - start, tools_called=get_tools_called())
    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": reply})
    yield None, conversation_history, last_working


# ── CLI Mode ──────────────────────────────────────────────────────────────────
def start_agent():
    print("Multi MCP Agent ready! Type exit to quit.\n")
    history = []
    key = 0
    while True:
        ui = input("You: ").strip()
        if not ui:
            continue
        if ui.lower() == "exit":
            print("Goodbye!")
            break
        reply, history, key = run_agent(ui, history, key)
        print(f"\nAssistant: {reply}\n")