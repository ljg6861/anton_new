"""
Tool for searching and understanding the agent's own code.
"""


class CodeSearchTool:
    """
    A tool that allows the agent to search through its own codebase.
    """

    function = {
        "type": "function",
        "function": {
            "name": "search_codebase",
            "description": "Search through the agent's codebase for relevant files or code snippets.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query related to code structure, functionality, or implementation details."
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of code snippets to return. Defaults to 3."
                    }
                },
                "required": ["query"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Execute the code search functionality."""
        from server.agent.rag_manager import rag_manager

        query = arguments.get('query')
        if not query:
            return "❌ Error: A search query is required."

        max_results = arguments.get('max_results', 3)

        try:
            # Search the knowledge base for relevant code snippets
            results = rag_manager.retrieve_knowledge(query=query, top_k=max_results)

            if not results:
                return "No relevant code found in the codebase for your query."

            # Format the results
            response = ["Here are the relevant code snippets:"]

            for i, result in enumerate(results, 1):
                source = result.get("source", "Unknown source")
                text = result.get("text", "").strip()

                # Limit very long snippets
                if len(text) > 1500:
                    text = text[:1500] + "...\n[truncated for brevity]"

                response.append(f"\n## Snippet {i}: {source}")
                response.append(f"```python\n{text}\n```")

            return "\n".join(response)

        except Exception as e:
            return f"❌ Error searching codebase: {str(e)}"