"""
Tool for completely resetting and rebuilding the code index.
"""
import os
import logging
import time

logger = logging.getLogger(__name__)


class RebuildCodeIndexTool:
    """
    A tool that completely resets and rebuilds the code index.
    """

    function = {
        "type": "function",
        "function": {
            "name": "rebuild_code_index",
            "description": "Completely reset and rebuild the code index. This deletes the existing index first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "confirm": {
                        "type": "boolean",
                        "description": "Set to true to confirm the destructive rebuild operation."
                    }
                },
                "required": ["confirm"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        """Execute the code index rebuild."""
        from server.agent.code_indexer import code_indexer
        from server.agent.rag_manager import rag_manager

        confirm = arguments.get('confirm', False)

        if not confirm:
            return "⚠️ Rebuild not performed. You must set 'confirm' to true to rebuild the index."

        try:
            # 1. Delete existing index files
            index_path = rag_manager.index_path
            doc_store_path = rag_manager.doc_store_path

            files_deleted = 0

            if os.path.exists(index_path):
                os.remove(index_path)
                files_deleted += 1

            if os.path.exists(doc_store_path):
                os.remove(doc_store_path)
                files_deleted += 1

            # 2. Reset the in-memory structures
            rag_manager._initialize_empty_stores()
            code_indexer.indexed_files_meta.clear()
            code_indexer.source_to_ids.clear()

            # 3. Rebuild the index from scratch
            start_time = time.time()
            files_indexed = code_indexer.index_directory()
            indexing_time = time.time() - start_time

            # 4. Save the new index
            rag_manager.save()

            return (
                f"✅ Successfully rebuilt code index.\n"
                f"- {files_deleted} old index files deleted\n"
                f"- {files_indexed} files newly indexed\n"
                f"- {rag_manager.index.ntotal} total knowledge entries created\n"
                f"- Completed in {indexing_time:.1f} seconds"
            )

        except Exception as e:
            logger.error(f"Error during index rebuild: {e}", exc_info=True)
            return f"❌ Error rebuilding index: {str(e)}"