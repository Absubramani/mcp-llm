import os
import re
import time
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import Field, create_model
from agent.tool_schema import fetch_tools
from agent.tool_executor import execute_tool
from agent.prompt import get_prompt
from agent.logger import log_llm_selected
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


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
def build_langchain_tools(all_tools: list, tool_server_map: dict, creds=None) -> list:
    langchain_tools = []

    for tool in all_tools:
        func_info = tool["function"]
        tool_name = func_info["name"]
        tool_desc = func_info["description"]
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

        def make_tool_fn(t_name, s_name, user_creds):
            def tool_fn(**kwargs) -> str:
                kwargs.pop("dummy", None)
                kwargs = {k: v for k, v in kwargs.items() if v != ""}
                return execute_tool(s_name, t_name, kwargs, creds=user_creds)
            tool_fn.__name__ = t_name
            tool_fn.__doc__ = tool_desc
            return tool_fn

        langchain_tool = StructuredTool.from_function(
            func=make_tool_fn(tool_name, server_name, creds),
            name=tool_name,
            description=tool_desc,
            args_schema=ArgsModel,
        )
        langchain_tools.append(langchain_tool)

    return langchain_tools


# ── Shared Helpers ────────────────────────────────────────────────────────────
def _build_agent_executor(llm, langchain_tools):
    agent = create_openai_tools_agent(llm, langchain_tools, get_prompt())
    return AgentExecutor(
        agent=agent,
        tools=langchain_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=8,
        early_stopping_method="generate",
    )


def _build_chat_history(conversation_history, fallback_hint=None):
    chat_history = []
    for msg in conversation_history[-6:]:
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
    if re.search(r'(search_emails|list_emails|get_email|download_|upload_file|send_email|search_files|read_file)\s*[א-׿؀-ۿﬀ-﷿]?\s*\{', reply):
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
) -> tuple[str, list[dict], int]:

    from agent.logger import log_request, log_response, log_error, log_llm_fallback
    from agent.tool_executor import reset_tools_called, get_tools_called

    user_input = _preprocess_input(user_input)
    log_request(user_input)
    reset_tools_called()
    start = time.time()

    all_tools, tool_server_map = fetch_tools()
    langchain_tools = build_langchain_tools(all_tools, tool_server_map, creds=creds)
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
        agent_executor = _build_agent_executor(llm, langchain_tools)
        chat_history = _build_chat_history(conversation_history, fallback_hint)

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
):
    from agent.logger import log_request, log_response, log_error, log_llm_fallback
    from agent.tool_executor import reset_tools_called, get_tools_called

    user_input = _preprocess_input(user_input)
    log_request(user_input)
    reset_tools_called()
    start = time.time()

    all_tools, tool_server_map = fetch_tools()
    langchain_tools = build_langchain_tools(all_tools, tool_server_map, creds=creds)
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
        agent_executor = _build_agent_executor(llm, langchain_tools)
        chat_history = _build_chat_history(conversation_history, fallback_hint)

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