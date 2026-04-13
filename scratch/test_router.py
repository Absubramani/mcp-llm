from agent.router import route

test_input = "read the second one"
result = route(test_input)
print(f"Input: {test_input}")
print(f"In Scope: {result.in_scope}")
print(f"Sections: {result.sections}")
