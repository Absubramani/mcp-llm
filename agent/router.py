"""
router.py — Router LLM (llama-3.1-8b-instant on dedicated Groq key)

Runs BEFORE the agent LLM on every request.
Uses GROQ_ROUTER_KEY (= GROQ_API_KEY_1) exclusively — never touches agent keys.

Responsibilities:
  1. Scope check      — is this request in-scope?
  2. Intent check     — conversational greeting vs action?
  3. Input cleanup    — fix spelling, normalize phrasing
  4. Section routing  — which of gmail / drive / github are needed?

Returns a RouterResult dataclass.
Falls back gracefully on any error — never crashes the main flow.
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("mcp_llm")

_ROUTER_MODEL = "llama-3.3-70b-versatile"

_ROUTER_SYSTEM = """You are a routing classifier for an AI assistant that manages Gmail, Google Drive, and GitHub.

Your job is to analyze a user message and return a JSON object with these exact fields:

{
  "in_scope": true or false,
  "is_conversational": true or false,
  "sections": ["gmail", "drive", "github"],
  "cleaned_input": "normalized version of user input",
  "out_of_scope_reply": ""
}

RULES:

1. in_scope = false ONLY for:
   - General knowledge questions (who is X, what is Y, explain concept Z)
   - Weather, news, math, jokes, coding help unrelated to the user's own files/repos
   - Anything not about the user's Gmail, Drive, or GitHub account
   - If in_scope is false, set out_of_scope_reply to a friendly 1-line message like:
     "❌ I can only help with Gmail, Google Drive, and GitHub tasks."
   - CRITICAL: "read the second one", "summarize it", "reply to that", "delete it", "forward that", "yes", "no" ARE ALWAYS IN-SCOPE!

2. is_conversational = true ONLY for:
   - Greetings: "hi", "hello", "hey", "what can you do", "help"
   - These should NOT trigger any tool calls
   - If true, sections can be empty []

3. sections = list of which services are relevant:
   - Include "gmail" if message involves email, inbox, send, reply, draft, schedule, attachment
   - Include "drive" if message involves files, folders, upload, download, document, spreadsheet
   - Include "github" if message involves repo, issue, PR, branch, project board, code
   - If unclear which service, include all three: ["gmail", "drive", "github"]
   - NEVER return empty sections for in-scope, non-conversational requests

4. cleaned_input = normalized user message:
   - Fix spelling: "lst" → "list", "emal" → "email", "delet" → "delete", "snd" → "send"
   - Expand informal: "gimme" → "show me", "repos" → "repositories"
   - Remove file extensions from search terms: "notes.txt" → "notes" (keep for create/read actions)
   - Keep original intent intact — do not rewrite meaning
   - If input is already clean, return it as-is

5. out_of_scope_reply = "" unless in_scope is false

Return ONLY valid JSON. No explanation. No markdown. No extra text."""


def _get_router_llm():
    """
    Get Groq LLM for routing.
    Uses GROQ_ROUTER_KEY exclusively — this is GROQ_API_KEY_1.
    Never touches GROQ_API_KEY_2 or GROQ_API_KEY_3 (reserved for Agent LLM).
    """
    from langchain_groq import ChatGroq

    router_key = os.getenv("GROQ_ROUTER_KEY")
    if not router_key:
        raise RuntimeError("GROQ_ROUTER_KEY not set in .env")

    return ChatGroq(
        api_key=router_key,
        model=_ROUTER_MODEL,
        temperature=0,
        max_retries=1,
        max_tokens=256,
    )


@dataclass
class RouterResult:
    in_scope: bool = True
    is_conversational: bool = False
    sections: list = field(default_factory=lambda: ["gmail", "drive", "github"])
    cleaned_input: str = ""
    out_of_scope_reply: str = ""
    error: Optional[str] = None


_FALLBACK = RouterResult(
    in_scope=True,
    is_conversational=False,
    sections=["gmail", "drive", "github"],
    cleaned_input="",
    error="router_fallback",
)


def route(user_input: str) -> RouterResult:
    """
    Run the Router LLM on user_input.
    Returns RouterResult. Never raises — falls back to all-sections on error.
    """
    if not user_input or not user_input.strip():
        return _FALLBACK

    try:
        llm = _get_router_llm()
        from langchain_core.messages import SystemMessage, HumanMessage
        response = llm.invoke([
            SystemMessage(content=_ROUTER_SYSTEM),
            HumanMessage(content=user_input.strip()),
        ])
        raw = response.content.strip()

        # Strip markdown fences if model wraps in ```json
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        data = json.loads(raw)

        sections = data.get("sections", ["gmail", "drive", "github"])
        if not isinstance(sections, list) or not sections:
            sections = ["gmail", "drive", "github"]
        valid_sections = {"gmail", "drive", "github"}
        sections = [s for s in sections if s in valid_sections]
        if not sections:
            sections = ["gmail", "drive", "github"]

        cleaned = data.get("cleaned_input", "").strip() or user_input.strip()
        in_scope = bool(data.get("in_scope", True))
        is_conv  = bool(data.get("is_conversational", False))

        # ── Router result log ─────────────────────────────────────────────────
        log.info(
            f"ROUTER  | in_scope={in_scope} | conversational={is_conv} | "
            f"sections={sections} | cleaned={cleaned!r}"
        )

        return RouterResult(
            in_scope=in_scope,
            is_conversational=is_conv,
            sections=sections,
            cleaned_input=cleaned,
            out_of_scope_reply=data.get("out_of_scope_reply", ""),
        )

    except Exception as e:
        err = str(e)
        # Give a clear actionable warning if the key is simply missing
        if "GROQ_ROUTER_KEY not set" in err:
            log.warning(
                "ROUTER  | GROQ_ROUTER_KEY missing in .env — falling back to all sections. "
                "⚠️  This loads the FULL prompt (~13K tokens) and WILL hit Groq's 12K TPM limit. "
                "Set GROQ_ROUTER_KEY in .env to fix this."
            )
        else:
            log.warning(f"ROUTER  | fallback triggered: {err}")

        return RouterResult(
            in_scope=True,
            is_conversational=False,
            sections=["gmail", "drive", "github"],
            cleaned_input=user_input.strip(),
            error=err,
        )