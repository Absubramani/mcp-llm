"""
orchestrator.py — Agent execution with dual-LLM architecture.

LLM 1 (Router):  llama-3.1-8b-instant  — uses GROQ_ROUTER_KEY (= GROQ_API_KEY_1)
LLM 2 (Agent):   llama-3.3-70b-versatile — uses GROQ_API_KEY_2, GROQ_API_KEY_3, Mistral, Ollama

Key isolation:
  GROQ_ROUTER_KEY  → router.py only
  GROQ_API_KEY_2   → agent, primary
  GROQ_API_KEY_3   → agent, fallback
  MISTRAL_API_KEY  → agent, fallback
  Ollama           → agent, last resort

Fixes applied:
  - CoercingModel: coerces all tool args to str before Pydantic validates them.
    Prevents "Input should be a valid string, got int" errors from Ollama/Groq.
  - Tool parse retry: on tool_use_failed (Groq bad tool call format), retries
    on next key with a hint injected into the prompt.
"""

import os
import time
from langchain_groq import ChatGroq


# ── RepairChatGroq — fixes Groq/Llama-3 tool-calling hallucinations ────────────
# Llama-3-70b on Groq intermittently generates:
#   1. XML-style tool calls (<function=list_emails{...}>)
#   2. Combined names (name='list_emails {"max_results": "3"}')
# Both causes Groq's API to reject the request with 400 'validation failed'.
class RepairChatGroq(ChatGroq):
    def bind(self, **kwargs):
        # parallel_tool_calls=True triggers the XML bug — disable it.
        kwargs.setdefault("parallel_tool_calls", False)
        return super().bind(**kwargs)

    def invoke(self, input, config=None, **kwargs):
        msg = super().invoke(input, config, **kwargs)
        return self._repair_message(msg)

    def _repair_message(self, msg):
        """Catch and fix 'combined' tool names (e.g. name='list_emails {"a": 1}')."""
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            import json
            repaired = []
            for tc in msg.tool_calls:
                name = tc.get("name", "")
                args = tc.get("args", {})
                
                # Case: 'list_emails {"max_results": "3"}'
                if " " in name or "{" in name:
                    parts = name.split("{", 1) if "{" in name else name.split(" ", 1)
                    clean_name = parts[0].strip()
                    json_str = "{" + parts[1] if "{" in name else parts[1]
                    
                    try:
                        # Try to parse the jammed JSON
                        hallucinated_args = json.loads(json_str)
                        if isinstance(hallucinated_args, dict):
                            args.update(hallucinated_args)
                        tc["name"] = clean_name
                        tc["args"] = args
                    except Exception:
                        # If parsing fails, just use the clean name and hope for the best
                        tc["name"] = clean_name

                repaired.append(tc)
            msg.tool_calls = repaired
        return msg

    def stream(self, input, config=None, **kwargs):
        # For streaming, we collect the full message if it has tool calls,
        # repair it, and yield the repaired message.
        full_msg = None
        for chunk in super().stream(input, config, **kwargs):
            if full_msg is None:
                full_msg = chunk
            else:
                full_msg += chunk
            
            # If it's not a tool call, we can yield immediately
            if not (hasattr(chunk, "tool_calls") and chunk.tool_calls):
                yield chunk
        
        # If the final message had tool calls, repair it and yield a single repaired chunk
        if full_msg and hasattr(full_msg, "tool_calls") and full_msg.tool_calls:
            yield self._repair_message(full_msg)


from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from typing import Any, Optional
from pydantic import BaseModel, Field, create_model, model_validator
from agent.tool_schema import fetch_tools
from agent.tool_executor import execute_tool, get_last_tool_result
from agent.prompt import get_prompt
from agent.logger import log_llm_selected
from agent.router import route
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

_GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


# ── CoercingModel — base for ALL tool arg schemas ─────────────────────────────
# Runs before Pydantic validates any field.
# Converts int/float/bool/None → str so LLM numeric outputs never crash.
# e.g. Ollama sends max_results=3 (int) → coerced to "3" (str) ✅
class CoercingModel(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def coerce_all_to_str(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        out = {}
        for k, v in values.items():
            if v is None:
                out[k] = ""
            elif isinstance(v, bool):
                out[k] = "true" if v else "false"
            elif isinstance(v, (int, float)):
                out[k] = str(v)
            elif isinstance(v, str) and v.strip().lower() == "null":
                # Groq LLMs sometimes emit the literal string "null" for optional
                # params — treat it the same as Python None so MCP servers never
                # receive a meaningless "null" string as an argument value.
                out[k] = ""
            else:
                out[k] = v
        return out


# ── Agent LLM — uses GROQ_API_KEY_2 and GROQ_API_KEY_3 only ──────────────────
def _get_agent_groq_keys() -> list:
    """
    Agent LLM uses KEY_2 and KEY_3 only.
    KEY_1 (GROQ_ROUTER_KEY) is reserved exclusively for the Router LLM.
    """
    return [k for k in [
        os.getenv("GROQ_API_KEY_2"),
        os.getenv("GROQ_API_KEY_3"),
    ] if k]


def get_llm(key_index: int = 0):
    provider = os.getenv("LLM_PROVIDER", "groq")

    if provider == "groq":
        agent_keys = _get_agent_groq_keys()
        if 0 <= key_index < len(agent_keys):
            log_llm_selected("groq", key_index + 2)  # +2 because we start from KEY_2
            return RepairChatGroq(
                api_key=agent_keys[key_index],
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
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    log_llm_selected("ollama")
    return ChatOllama(
        model="llama3.1:8b",
        base_url=ollama_url,
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
        "list_emails": "List inbox emails. max_results, query, page_token optional.",
        "read_email": "Read full email by id.",
        "send_email": "Send email. to required. subject/body empty triggers ask flow.",
        "reply_to_email": "Reply to email by id. body required.",
        "forward_email": "Forward email. email_id, to_email required.",
        "delete_email": "Move email to trash. confirmed='true' required.",
        "restore_email": "Restore email from trash. confirmed='true' required.",
        "search_emails": "Search emails by Gmail query.",
        "send_email_with_attachment": "Send email with local file attachment. file_path required.",
        "get_email_attachments": "List attachments in an email by id.",
        "download_email_attachment": "Download attachment. email_id, attachment_id, filename required.",
        "list_unread_emails": "List unread inbox emails.",
        "mark_as_read": "Mark emails as read. email_ids: exact id or comma-separated.",
        "mark_as_unread": "Mark emails as unread. email_ids: exact id or comma-separated.",
        "save_draft": "Save email as draft.",
        "list_drafts": "List all saved drafts.",
        "update_draft": "Update draft. Call list_drafts first to get draft_id.",
        "send_draft": "Send a draft. Call list_drafts first to get draft_id.",
        "delete_draft": "Delete a draft. confirmed='true' required.",
        "schedule_email": "Schedule email. to, send_at required.",
        "list_scheduled_emails": "List all scheduled emails.",
        "cancel_scheduled_email": "Cancel a scheduled email by job_id.",
        "get_user_timezone": "Get system timezone.",
        "create_folder": "Create folder or nested folders. path required.",
        "list_files": "List files in folder. path empty = root.",
        "create_text_file": "Create text file in Drive. path and content required.",
        "read_file": "Read file content from Drive. path required.",
        "delete_file": "Move file to trash. confirmed='true' required.",
        "restore_file": "Restore file from trash. confirmed='true' required.",
        "move_file": "Move file to another folder.",
        "rename_file": "Rename a file or folder.",
        "copy_file": "Copy file to another folder.",
        "search_files": "Search files by keyword. NEVER include extension in query.",
        "get_file_info": "Get file details.",
        "upload_file": "Upload local file to Drive. file_path required.",
        "download_file": "Download Drive file to local machine.",
        "list_recent_files": "List recently modified files.",
        "share_file": "Share file with email. role: reader/commenter/writer.",
        "list_repos": "List user GitHub repos. limit optional.",
        "create_repo": "Create GitHub repo. name required.",
        "search_repos": "Search GitHub repos. query required.",
        "list_repo_files": "List files in repo. repo required (owner/repo or short name).",
        "read_file_from_repo": "Read file from repo. repo and file_path required.",
        "list_issues": "List issues in repo. repo required.",
        "create_issue": "Create issue. repo and title required.",
        "read_issue": "Read specific issue. repo and issue_number required.",
        "list_branches": "List all branches. repo required.",
        "list_pull_requests": "List pull requests. repo required.",
        "create_pull_request": "Create pull request. repo, title, head required.",
        "merge_pull_request": "Merge a pull request. repo and pull_number required.",
        "create_branch": "Create new branch. repo, branch_name required.",
        "list_projects": "List GitHub Projects v2. repo optional.",
        "create_project": "Create a new GitHub Project v2. title required.",
        "get_project_columns": "Get project board columns. project_id required.",
        "add_issue_to_project": "Add issue to project board. project_id and issue_url required.",
        "move_issue_to_column": "Move issue to column. project_id, item_id, column_name required.",
        "update_project_issue_fields": "Update start_date/end_date on project item.",
        "list_project_issues": "List all issues on project board with item_id.",
        "update_project_issue_by_title": "Update/move project issue by title. project_id, issue_title required.",
        "create_project_issue": "Create issue + add to project Backlog. repo, project_id, title required.",
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
        func_info  = tool["function"]
        tool_name  = func_info["name"]
        tool_desc  = _SHORT_DESC.get(tool_name, func_info["description"][:120])
        server_name = tool_server_map.get(tool_name, "drive")
        properties = func_info.get("parameters", {}).get("properties", {})

        fields = {}
        for prop_name, prop_info in properties.items():
            fields[prop_name] = (Optional[str], Field(
                default="",
                description=prop_info.get("description", "")
            ))
        if not fields:
            fields["dummy"] = (Optional[str], Field(default=""))

        # Use CoercingModel as base — coerces int/float/bool/None to str
        # before Pydantic validates, preventing validation errors from any LLM
        ArgsModel = create_model(f"{tool_name}Args", __base__=CoercingModel, **fields)

        def make_tool_fn(t_name, s_name, user_creds, gh_token, is_gh):
            def tool_fn(**kwargs) -> str:
                kwargs.pop("dummy", None)
                if not is_gh:
                    kwargs = {k: v for k, v in kwargs.items() if v not in ("", None)}
                return execute_tool(s_name, t_name, kwargs, creds=user_creds, github_token=gh_token)
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


# ── Section-based Tool Filter ─────────────────────────────────────────────────
def _filter_tools_by_sections(
    langchain_tools: list,
    tool_server_map: dict,
    sections: list,
) -> list:
    """
    Return only the LangChain tools whose MCP server is in `sections`.

    This is the critical piece that makes the router work end-to-end:
      Router  → sections=['gmail']
      Filter  → keeps only gmail tools in the agent's tool list
      Result  → LLM can only call gmail tools → no 'not in request.tools' errors
                AND Groq receives fewer tool schemas → less chance of token overflow

    Falls back to all tools if sections covers every service or is empty.
    """
    if not sections or set(sections) >= {"gmail", "drive", "github"}:
        return langchain_tools  # no filtering needed — full tool set

    allowed_servers = set(sections)
    filtered = [
        t for t in langchain_tools
        if tool_server_map.get(t.name, "drive") in allowed_servers
    ]
    # Safety: if filtering produced an empty list fall back to full set
    return filtered if filtered else langchain_tools


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
    limit = 6  # Kept high so it remembers past lists and IDs
    chat_history = []
    
    for msg in conversation_history[-limit:]:
        content = msg.get("content", "")
        if not content:
            continue
            
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=content))
        elif msg["role"] == "assistant":
            # Smart Truncation: prevent massive email/file bodies from crashing Groq's 12K TPM
            # Keep first 1500 and last 1500 chars (safe for IDs and context)
            if len(content) > 3000:
                content = content[:1500] + "\n\n...[content truncated for context length]...\n\n" + content[-1500:]
            chat_history.append(AIMessage(content=content))
            
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
    if any(x in lower for x in ["invalid_api_key", "invalid api key", "401", "unauthorized"]):
        return "auth_error"
    if any(x in lower for x in ["tool_use_failed", "failed_generation", "pydantic", "serializ"]):
        return "parse_fail"
    if "400" in error_str and "failed to call a function" in lower:
        return "groq_400"
    return "unknown"


# ── Result Formatters ─────────────────────────────────────────────────────────
def _format_emails(raw: str) -> str:
    from agent.orchestrator import _parse_tool_result
    items = _parse_tool_result(raw)
    if not items:
        return ""
    NL = "\n"
    # Gmail tools sometimes return {emails: [...], ...} instead of a straight list
    emails = items
    if isinstance(items, list) and items and "emails" in items[0]:
        emails = items[0]["emails"]
    elif isinstance(items, dict) and "emails" in items:
        emails = items["emails"]

    if not emails or not isinstance(emails, list):
        if isinstance(items, list) and items and "message" in items[0]:
            return f"📧 {items[0]['message']}"
        return ""

    lines = ["📧 **Your latest emails**:" + NL]
    for i, e in enumerate(emails, 1):
        email_id = e.get('id', '')
        lines.append(f"{i}. **From:** {e.get('from', 'Unknown')} | **Subject:** {e.get('subject', '(no subject)')} <!-- id:{email_id} -->")
        lines.append(f"   📅 {e.get('date', '')}")
        snippet = e.get('snippet', '')
        if snippet:
            lines.append(f"   > {snippet[:120]}...")
        lines.append("")

    lines.append("Reply with a number to read, summarize, or reply!")
    return NL.join(lines)

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
        lines.append(f"{i}. **{name}**{priv} | ⭐ {stars} | {lang} <!-- name:{name} -->")
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
        number = iss.get('number')
        lines.append(f"{i}. **#{number}** {iss.get('title')} | by {iss.get('author', '')} <!-- number:{number} -->")
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
        number = pr.get('number')
        lines.append(f"{i}. **#{number}** {pr.get('title')} | {pr.get('head')} → {pr.get('base')} <!-- number:{number} -->")
    return NL.join(lines)


def _format_branches(raw: str) -> str:
    branches = _parse_tool_result(raw)
    if not branches:
        return ""
    NL = "\n"
    lines = ["🌿 Branches:" + NL]
    for i, b in enumerate(branches, 1):
        name = b.get('name', '')
        default = " (default)" if b.get("default") else ""
        lines.append(f"{i}. **{name}**{default} <!-- name:{name} -->")
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
        pid = p.get('project_id') or p.get('id', '')
        num = p.get('number')
        lines.append(f"{i}. **{p.get('title')}** (Project #{num}) <!-- project_id:{pid} -->")
    lines.append(NL + "Reply with a project name to view its board.")
    return NL.join(lines)


def _format_file_content(raw: str) -> str:
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
                ext = file_name.rsplit(".", 1)[-1].lower() if "." in file_name else ""
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

    _GH_TOOLS = {
        "list_repos", "create_repo", "search_repos", "list_repo_files",
        "read_file_from_repo", "list_issues", "create_issue", "read_issue",
        "list_pull_requests", "create_pull_request", "merge_pull_request",
        "list_branches", "create_branch", "list_projects", "create_project",
        "get_project_columns", "add_issue_to_project", "move_issue_to_column",
        "update_project_issue_fields", "list_project_issues",
        "update_project_issue_by_title", "create_project_issue",
    }

    for t_name in tools_done:
        if t_name in _GH_TOOLS:
            res = str(get_last_tool_result(t_name))
            if "No GitHub credentials found" in res:
                return "🐙 GitHub is not connected yet.\n\nPlease click **Connect GitHub** in the menu (☰) to link your account first!"

    tools_set = set(tools_done)

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

    if "upload_file" in tools_set:
        return "✅ File uploaded to **Google Drive** successfully! 📁"
    if "download_email_attachment" in tools_set:
        return "✅ Attachment downloaded to your Downloads folder successfully!"

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

    if "create_project_issue" in tools_set:
        raw = get_last_tool_result("create_project_issue")
        try:
            data = json.loads(raw)
            if isinstance(data, list):
                data = data[0]
            if data.get("status") == "success":
                num = data.get("number", "")
                url = data.get("url", "")
                col = data.get("column", "Backlog")
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
                    "**move the '[issue title]' issue in [project name] to [column]**"
                )
            return f"❌ {err}"
        except Exception:
            pass

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

    if "list_project_issues" in tools_set:
        raw = get_last_tool_result("list_project_issues")
        fmt = _format_project_issues(raw)
        return fmt if fmt else "📋 Project issues retrieved — please try again."

    if "list_projects" in tools_set:
        raw = get_last_tool_result("list_projects")
        fmt = _format_projects(raw)
        return fmt if fmt else "📋 Projects retrieved — please try again."

    if "list_emails" in tools_set or "search_emails" in tools_set:
        raw = get_last_tool_result("list_emails") or get_last_tool_result("search_emails")
        fmt = _format_emails(raw)
        return fmt if fmt else "📧 Emails retrieved — please try again."

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
        return fmt if fmt else "🌿 Branches retrieved — please try again."

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
    error_str, key_index, agent_keys, has_mistral,
    fallback_hint, get_tools_called_fn, log_llm_fallback_fn, log_error_fn, user_input,
):
    error_type = _classify_error(error_str)

    if error_type == "auth_error":
        log_error_fn(error_str, context=f"user_input={user_input!r}")
        return key_index, True, fallback_hint, "❌ Authentication failed. Please check your API keys."

    if error_type in ("rate_limit", "token_limit"):
        tools_done = get_tools_called_fn()
        if tools_done:
            log_llm_fallback_fn(key_index + 2 if key_index >= 0 else key_index, error_str)
            smart = _smart_reply_from_tools(tools_done, user_input)
            if smart:
                return key_index, True, fallback_hint, smart
        if key_index >= 0:
            next_key = key_index + 1
            if next_key < len(agent_keys):
                log_llm_fallback_fn(key_index + 2, error_str)
                return next_key, False, fallback_hint, None
        if has_mistral and key_index != -2:
            log_llm_fallback_fn(key_index + 2 if key_index >= 0 else key_index, error_str)
            return -2, False, fallback_hint, None
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
        log_llm_fallback_fn(key_index + 2, error_str)
        tools_done = get_tools_called_fn()
        if tools_done:
            fallback_hint = (
                f"[System: Previous attempt already called tools: {', '.join(tools_done)}. "
                "Do NOT call these tools again. Use results already obtained.]"
            )
        else:
            # Inject tool format hint so next key avoids the same parse error
            fallback_hint = (
                "[System: Previous attempt failed with a tool call format error. "
                "You MUST call tools using proper JSON function call format only. "
                "Do NOT use <function=...> XML-style format.]"
            )
        next_key = key_index + 1
        if next_key >= len(agent_keys):
            next_key = -2 if has_mistral else -1
        return next_key, False, fallback_hint, None

    if error_type != "unknown" and key_index >= 0:
        log_llm_fallback_fn(key_index + 2, error_str)
        next_key = key_index + 1
        if next_key >= len(agent_keys):
            next_key = -2 if has_mistral else -1
        return next_key, False, fallback_hint, None

    log_error_fn(error_str, context=f"user_input={user_input!r}")
    return key_index, True, fallback_hint, f"Something went wrong: {error_str[:200]}"


# ── Garbage Detection ─────────────────────────────────────────────────────────
_GARBAGE_MARKERS = [
    "type': 'reference'", "type': 'text'", "{'type':", '{"type":',
    'search_emails": {', 'list_emails": {', 'list_repos": {', 'create_repo": {',
    '"function":', '"args":', '{"name":', '{"parameters":', '{"call":',
]


def _is_garbage_reply(reply: str) -> bool:
    if not reply:
        return False
    # If it's a very short response check if it's just raw JSON
    if reply.strip().startswith("{") and "}" in reply:
        # Check if it looks like a tool call or schema instead of a sentence
        if any(marker in reply for marker in _GARBAGE_MARKERS):
            return True
    if "pydanticserializat" in reply.lower():
        return True
    if reply.strip().startswith("[") and "{'type'" in reply:
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

    log_request(user_input)
    reset_tools_called()
    start = time.time()

    # ── Router LLM ────────────────────────────────────────────────────────────
    router_result = route(user_input)

    if not router_result.in_scope:
        reply = router_result.out_of_scope_reply or "❌ I can only help with Gmail, Google Drive, and GitHub tasks."
        conversation_history.append({"role": "user",      "content": user_input})
        conversation_history.append({"role": "assistant", "content": reply})
        return reply, conversation_history, start_key_index

    cleaned_input = router_result.cleaned_input or user_input
    sections      = router_result.sections or ["gmail", "drive", "github"]

    all_tools, tsm = fetch_tools()
    lt             = build_langchain_tools(all_tools, tsm, creds=creds, github_token=github_token)
    lt             = _filter_tools_by_sections(lt, tsm, sections)  # ← router-driven filter
    agent_keys     = _get_agent_groq_keys()
    has_mistral    = bool(os.getenv("MISTRAL_API_KEY"))
    key_index      = start_key_index
    last_working   = start_key_index
    reply          = None
    fallback_hint  = None
    retry_count    = 0

    while True:
        if retry_count >= 6:
            reply = "All AI providers are currently busy. Please try again in a minute."
            break
        retry_count += 1

        llm, key_index = get_llm(key_index)
        ae = _build_agent_executor(llm, lt, sections)
        ch = _build_chat_history(conversation_history, fallback_hint, sections)

        try:
            result = ae.invoke({"input": cleaned_input, "chat_history": ch})
            reply  = result["output"]

            for t in get_tools_called():
                if "No GitHub credentials found" in str(get_last_tool_result(t)):
                    reply = "🐙 GitHub is not connected yet.\n\nPlease click **Connect GitHub** in the menu (☰) to link your account first!"
                    break

            if _is_garbage_reply(reply):
                try:
                    reply = ae.invoke({"input": cleaned_input, "chat_history": ch})["output"]
                except Exception:
                    reply = "I ran into an issue. Please try again."

            last_working = key_index
            break

        except Exception as e:
            nk, sb, fallback_hint, er = _handle_exception(
                str(e), key_index, agent_keys, has_mistral,
                fallback_hint, get_tools_called, log_llm_fallback, log_error, cleaned_input)
            if sb:
                reply = er
                break
            key_index = nk

    reply = reply or "I ran into an issue. Please try again."
    log_response(cleaned_input, reply, time.time() - start, tools_called=get_tools_called())
    conversation_history.append({"role": "user",      "content": user_input})
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

    log_request(user_input)
    reset_tools_called()
    start = time.time()

    # ── Router LLM ────────────────────────────────────────────────────────────
    router_result = route(user_input)

    if not router_result.in_scope:
        reply = router_result.out_of_scope_reply or "❌ I can only help with Gmail, Google Drive, and GitHub tasks."
        conversation_history.append({"role": "user",      "content": user_input})
        conversation_history.append({"role": "assistant", "content": reply})
        yield reply, conversation_history, start_key_index
        return

    cleaned_input = router_result.cleaned_input or user_input
    sections      = router_result.sections or ["gmail", "drive", "github"]

    all_tools, tsm = fetch_tools()
    lt             = build_langchain_tools(all_tools, tsm, creds=creds, github_token=github_token)
    lt             = _filter_tools_by_sections(lt, tsm, sections)  # ← router-driven filter
    agent_keys     = _get_agent_groq_keys()
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
            full   = []
            buffer = ""
            garbage_detected = False

            for chunk in ae.stream({"input": cleaned_input, "chat_history": ch}):
                if "output" in chunk and chunk["output"]:
                    text = chunk["output"]
                    full.append(text)

                    # ── Garbage Buffering — prevent leaking raw JSON to UI ────
                    # Buffer the first 50 chars to check if LLM is flaking
                    if not garbage_detected and len(buffer) < 50:
                        buffer += text
                        if len(buffer) >= 20: # Enough to check for JSON markers
                            if _is_garbage_reply(buffer):
                                garbage_detected = True
                                break # Stop streaming immediately
                        
                        # Only yield if we've buffered enough to be sure it's not JSON
                        # OR if we've already started yielding
                        if len(buffer) >= 50:
                            yield buffer, None, None
                        continue

                    if not garbage_detected:
                        yield text, None, None

            if garbage_detected:
                # Trigger fallback logic below by setting reply=garbage
                reply = "".join(full)
            else:
                reply = "".join(full)

            for t in get_tools_called():
                if "No GitHub credentials found" in str(get_last_tool_result(t)):
                    reply = "🐙 GitHub is not connected yet.\n\nPlease click **Connect GitHub** in the menu (☰) to link your account first!"
                    break

            if _is_garbage_reply(reply):
                full      = []
                tools_done = get_tools_called()
                if tools_done:
                    fallback_hint = (
                        f"[System: Previous attempt called tools: {', '.join(tools_done)}. "
                        "Do NOT call these again. Use results already obtained.]"
                    )
                if key_index >= 0:
                    nk = key_index + 1
                    if nk >= len(agent_keys):
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
                str(e), key_index, agent_keys, has_mistral,
                fallback_hint, get_tools_called, log_llm_fallback, log_error, cleaned_input)
            if sb:
                reply = er or "I ran into an issue. Please try again."
                yield reply, None, None
                break
            key_index = nk

    reply = reply or "I ran into an issue. Please try again."
    log_response(cleaned_input, reply, time.time() - start, tools_called=get_tools_called())
    conversation_history.append({"role": "user",      "content": user_input})
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