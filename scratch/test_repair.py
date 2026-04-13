import json

class MockMessage:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls

def _repair_message(msg):
    """Catch and fix 'combined' tool names (e.g. name='list_emails {"a": 1}')."""
    if hasattr(msg, "tool_calls") and msg.tool_calls:
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
                except Exception as e:
                    print(f"Repair failed: {e}")
                    tc["name"] = clean_name

            repaired.append(tc)
        msg.tool_calls = repaired
    return msg

# Test cases
test_cases = [
    {
        "name": "Hallucinated name with JSON",
        "input": [{"name": 'list_emails {"max_results": "3"}', "args": {}}],
        "expected_name": "list_emails",
        "expected_args": {"max_results": "3"}
    },
    {
        "name": "Standard tool call",
        "input": [{"name": 'read_email', "args": {"id": "123"}}],
        "expected_name": "read_email",
        "expected_args": {"id": "123"}
    },
    {
        "name": "Hallucinated name with space but no JSON",
        "input": [{"name": 'list_emails 3', "args": {}}],
        "expected_name": "list_emails",
        "expected_args": {}
    }
]

for tc in test_cases:
    print(f"Testing: {tc['name']}")
    msg = MockMessage(tc["input"])
    repaired_msg = _repair_message(msg)
    res = repaired_msg.tool_calls[0]
    print(f"  Result: name='{res['name']}', args={res['args']}")
    assert res["name"] == tc["expected_name"]
    assert res["args"] == tc["expected_args"]
    print("  ✅ Passed")
