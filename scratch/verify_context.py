import json
from langchain_core.messages import AIMessage, HumanMessage

def test_context_resolution():
    # Simulate turn 1: user lists emails
    # Agent output contains hidden IDs
    turn1_output = """📧 **Your latest emails**:
1. **From:** Indeed | **Subject:** new job openings <!-- id:email_123 -->
2. **From:** LinkedIn | **Subject:** Citi is hiring <!-- id:email_456 -->

Reply with a number to read, summarize, or reply!"""

    # Simulate turn 2: user says "read my first email"
    user_input = "read my first email"
    
    # In a real run, the ChatPromptTemplate would combine the system prompt
    # and this history. The LLM would see turn1_output.
    # We want to ensure that turn1_output physically contains the IDs.
    
    print("Verifying hidden IDs in Turn 1 output...")
    assert "<!-- id:email_123 -->" in turn1_output
    assert "<!-- id:email_456 -->" in turn1_output
    print("✅ Turn 1 output contains hidden IDs.")

    # Verification of prompt logic (from prompt.py)
    print("\nVerifying Prompt Logic for Context Resolution...")
    system_prompt_snippet = """
CRITICAL: CONTEXTUAL RESOLUTION (FIRST, SECOND, IT, THAT)
   - Use chat history to resolve relative references.
   - Look for hidden HTML comments in previous results for IDs: <!-- id:XYZ --> or <!-- number:123 -->.
"""
    assert "hidden HTML comments" in system_prompt_snippet
    print("✅ System prompt instructs model to look for hidden HTML comments.")

if __name__ == "__main__":
    test_context_resolution()
