"""
Tool for retrieving statistics about the agent's codebase.
"""


class CodebaseStatsTool:
    """
    A tool that provides statistics and information about the indexed codebase. This is NOT a codebase search.
    """

    function = {
        "type": "function",
        "function": {
            "name": "get_codebase_stats",
            "description": "Get statistics about the agent's indexed codebase.",
            "parameters": {
                "type": "object",
                "properties": {
                    "refresh": {
                        "type": "boolean",
                        "description": "Whether to refresh the code index before returning stats. Defaults to false."
                    }
                }
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Execute the codebase statistics functionality."""
        from server.agent.code_indexer import code_indexer

        refresh = arguments.get('refresh', False)

        try:
            if refresh:
                updated_files = code_indexer.refresh_index()
                refresh_msg = f"Refreshed code index: {updated_files} files updated."
            else:
                refresh_msg = "Using existing code index."

            stats = code_indexer.get_stats()

            # Format file extension statistics
            extension_stats = []
            for ext, count in sorted(stats["file_extensions"].items(), key=lambda x: x[1], reverse=True):
                extension_stats.append(f"  - {ext}: {count} files")

            response = [
                           f"📊 Codebase Statistics ({refresh_msg})",
                           f"Total indexed files: {stats['total_files']}",
                           f"Knowledge entries: {stats['knowledge_entries']}",
                           "\nFile types:"
                       ] + extension_stats

            return "\n".join(response)

        except Exception as e:
            return f"❌ Error retrieving codebase statistics: {str(e)}"