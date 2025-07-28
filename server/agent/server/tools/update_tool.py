def update_tool(tool_name, new_code):
    """Updates an existing tool's code by overwriting its file."""
    with open(f'./server/tools/{tool_name}.py', 'w') as f:
        f.write(new_code)