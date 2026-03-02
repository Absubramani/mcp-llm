import os
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.tools import StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import Field, create_model
from agent.tool_schema import fetch_tools
from agent.tool_executor import execute_tool
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm():
    return ChatOllama(
        model="llama3.1:8b",
        base_url="http://localhost:11434",
        temperature=0,
    )

# ── Prompt ────────────────────────────────────────────────────────────────────
def get_prompt():
    return ChatPromptTemplate.from_messages([
        ("system",
            "You are a smart, friendly and patient AI assistant with access to Google Drive and Gmail tools.\n\n"

            "CRITICAL: When calling any tool — ALWAYS use real values from previous tool results. NEVER use placeholder text like 'insert_id_here' or 'email_id_from_search_result'.\n"

            "AVAILABLE TOOLS:\n"
            "DRIVE: create_folder, list_files, create_text_file, read_file, delete_file, move_file, rename_file, copy_file, search_files, get_file_info\n"
            "GMAIL: list_emails, read_email, send_email, reply_to_email, delete_email, search_emails\n\n"

            "INPUT UNDERSTANDING:\n"
            "1. Understand spelling mistakes — 'emal', 'mial', 'emil' all mean 'email'. 'delet' means 'delete'. 'creat' means 'create'.\n"
            "2. Understand short forms — 'lst' means 'list', 'snd' means 'send', 'rd' means 'read', 'del' means 'delete'.\n"
            "3. Understand grammar mistakes — 'show me my mails', 'i want to saw my emails' all mean 'list emails'.\n"
            "4. Understand informal language — 'gimme my mails', 'check my inbox' all mean 'list emails'.\n"
            "5. If the request is unclear, make a reasonable guess and proceed — never ask the user to rephrase.\n\n"

            "FOLLOW-UP CONTEXT:\n"
            "1. Always remember the last email or file mentioned in the conversation.\n"
            "2. 'read that' or 'open it' — read the last mentioned email or file.\n"
            "3. 'delete it' or 'remove it' — delete the last mentioned email or file.\n"
            "4. 'reply to him' or 'reply to that' — reply to the last mentioned email.\n"
            "5. 'summarize it' or 'summarize that' — summarize the last mentioned email.\n"
            "6. Use chat history to understand what 'it', 'that', 'him', 'this' refers to.\n\n"

            "EMAIL RULES:\n"
            "1. When user asks to READ an email — call list_emails first, then IMMEDIATELY call read_email with the id. Show COMPLETE content — From, To, Subject, Date, and full Body. Never truncate.\n"
            "2. When user asks to SUMMARIZE an email:\n"
            "   MANDATORY STEP 1 — call list_emails or search_emails to find the email id.\n"
            "   MANDATORY STEP 2 — call read_email with that exact id to get the FULL body.\n"
            "   MANDATORY STEP 3 — only after read_email returns real content, write a 3-5 line summary covering: who sent it, what it is about, and what action is needed.\n"
            "   WARNING: NEVER summarize from snippet or subject alone — always call read_email first.\n"
            "   WARNING: NEVER skip read_email — summary without read_email is forbidden.\n"
            "3. When user asks to DELETE an email — call search_emails with keyword from the subject only. Search by the exact subject word the user mentioned. Pick the FIRST result with matching subject. Never pick emails where the keyword only appears in the body.\n"
            "4. When user asks to REPLY — find the email id first, then call reply_to_email.\n"
            "5. NEVER hallucinate — only use real data from tools.\n\n"

            "ERROR HANDLING:\n"
            "1. NEVER show raw technical errors to the user.\n"
            "2. ALWAYS read the tool result carefully — if it says 'status: success', tell the user it was successful.\n"
            "3. If tool result says 'status: success' — always confirm with a friendly message like 'Done! Email deleted successfully.'\n"
            "4. If tool result says 'status: error' — give a friendly error message.\n"
            "5. If email not found — say 'I couldn't find any email matching that.'\n"
            "6. If sending fails — say 'I was unable to send the email. Please check the address and try again.'\n"
            "7. If Drive operation fails — say 'I had trouble with that Drive operation. Please try again.'\n\n"

            "RESPONSE STYLE:\n"
            "1. Be professional, friendly and concise.\n"
            "2. Keep responses short and clear — avoid over-explaining.\n"
            "3. Always confirm completed actions clearly — 'Done! Your email has been sent.'\n"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])


# ── Tools Builder ─────────────────────────────────────────────────────────────
def build_langchain_tools(all_tools: list, tool_server_map: dict) -> list:
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
                fields[prop_name] = (str, Field(...))
            else:
                fields[prop_name] = (str, Field(default=""))

        ArgsModel = create_model(f"{tool_name}Args", **fields)

        def make_tool_fn(t_name, s_name):
            def tool_fn(**kwargs) -> str:
                return execute_tool(s_name, t_name, kwargs)
            tool_fn.__name__ = t_name
            tool_fn.__doc__ = tool_desc
            return tool_fn

        langchain_tool = StructuredTool.from_function(
            func=make_tool_fn(tool_name, server_name),
            name=tool_name,
            description=tool_desc,
            args_schema=ArgsModel,
        )
        langchain_tools.append(langchain_tool)

    return langchain_tools


# ── Agent ─────────────────────────────────────────────────────────────────────
def run_agent(user_input: str, conversation_history: list[dict]) -> tuple[str, list[dict]]:
    from agent.logger import log_request, log_response, log_error, log_rate_limit
    import time

    log_request(user_input)
    start = time.time()

    all_tools, tool_server_map = fetch_tools()
    langchain_tools = build_langchain_tools(all_tools, tool_server_map)

    llm = get_llm()
    prompt = get_prompt()

    agent = create_openai_tools_agent(llm, langchain_tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent,
        tools=langchain_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10,
        early_stopping_method="generate",
    )

    chat_history = []
    for msg in conversation_history[-4:]:
        if msg["role"] == "user":
            chat_history.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant" and msg.get("content"):
            chat_history.append(AIMessage(content=msg["content"]))

    try:
        result = agent_executor.invoke({
            "input": user_input,
            "chat_history": chat_history,
        })
        reply = result["output"]

    except Exception as e:
        import traceback
        error_str = str(e)
        log_error(error_str, context=f"user_input={user_input!r}")

        # Handle rate limit
        if "429" in error_str or "rate_limit_exceeded" in error_str:
            import re
            wait_match = re.search(r'try again in (\d+)m', error_str)
            wait_mins = int(wait_match.group(1)) if wait_match else 0
            log_rate_limit(wait_mins * 60, "llama-3.3-70b-versatile")
            reply = f"I've hit the rate limit. Please try again in {wait_mins} minutes."

        # Handle token limit
        elif "413" in error_str or "too large" in error_str.lower():
            reply = "That request was too large. Please try with a shorter message."

        # Handle tool use failed - retry
        elif "tool_use_failed" in error_str or "Failed to call a function" in error_str:
            try:
                result = agent_executor.invoke({
                    "input": user_input + " (use simple search query)",
                    "chat_history": [],
                })
                reply = result["output"]
            except Exception as e2:
                log_error(str(e2), context="retry_failed")
                reply = "I had trouble with that request. Please try again."

        else:
            reply = "I ran into an issue processing your request. Please try again."

    duration = time.time() - start
    log_response(user_input, reply, duration)

    conversation_history.append({"role": "user", "content": user_input})
    conversation_history.append({"role": "assistant", "content": reply})

    return reply, conversation_history


def start_agent():
    print("Multi MCP Agent ready! Type exit to quit.\n")
    conversation_history = []

    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Goodbye!")
            break

        reply, conversation_history = run_agent(user_input, conversation_history)
        print(f"\nAssistant: {reply}\n") 