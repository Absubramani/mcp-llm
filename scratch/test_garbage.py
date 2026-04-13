import json

_GARBAGE_MARKERS = [
    "type': 'reference'", "type': 'text'", "{'type':", '{"type":',
    'search_emails": {', 'list_emails": {', 'list_repos": {', 'create_repo": {',
    '"function":', '"args":', '{"name":', '{"parameters":', '{"call":',
]

def _is_garbage_reply(reply: str) -> bool:
    if not reply:
        return False
    if reply.strip().startswith("{") and "}" in reply:
        if any(marker in reply for marker in _GARBAGE_MARKERS):
            return True
    if "pydanticserializat" in reply.lower():
        return True
    for marker in _GARBAGE_MARKERS:
        if marker in reply:
            return True
    return False

# Test cases
test_cases = [
    '{"type": "function", "name": "list_emails", "parameters": {"max_results": {"type": "string", "value": "3"}}}',
    '{"name": "list_emails", "parameters": {...}}',
    'Hello, here are your emails:',
    'Here is the result: {"type": "function"}'
]

for tc in test_cases:
    print(f"Input: {tc[:50]}... -> Garbage: {_is_garbage_reply(tc)}")
