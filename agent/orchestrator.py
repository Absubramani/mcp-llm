import os
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_core.tools import StructuredTool
from langchain_core.messages import HumanMessage, AIMessage
from langchain.agents import AgentExecutor, create_openai_tools_agent
from pydantic import Field, create_model
from agent.tool_schema import fetch_tools
from agent.tool_executor import execute_tool
from agent.prompt import get_prompt          # ← import from new file
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)


# ── LLM ───────────────────────────────────────────────────────────────────────
def get_llm():
    provider = os.getenv("LLM_PROVIDER", "ollama")
    if provider == "groq":
        return ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model="llama-3.3-70b-versatile",
            temperature=0,
        )
    else:
        return ChatOllama(
            model="llama3.1:8b",
            base_url="http://localhost:11434",
            temperature=0,
        )


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
    from agent.logger import log_request, log_response, log_error
    from agent.tool_executor import reset_tools_called, get_tools_called
    import time

    log_request(user_input)
    reset_tools_called()
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
    )

    followup_keywords = [
        "read that", "summarize it", "summarize that", "delete it",
        "reply to that", "reply to him", "open it", "that one",
        "second one", "first one", "third one", "the same", "it"
    ]
    new_action_keywords = [
        "list", "lst", "show", "gimme", "get", "create",
        "send", "snd", "search", "find", "make"
    ]

    is_followup = any(kw in user_input.lower() for kw in followup_keywords)
    is_new_action = any(kw in user_input.lower() for kw in new_action_keywords)

    chat_history = []
    if is_followup or not is_new_action:
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
        error_str = str(e)
        log_error(error_str, context=f"user_input={user_input!r}")
        reply = "I ran into an issue processing your request. Please try again."

    duration = time.time() - start
    log_response(user_input, reply, duration, tools_called=get_tools_called())

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
