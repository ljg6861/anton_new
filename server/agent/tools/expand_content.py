"""
Tool for expanding compressed content and examples when the agent needs more detail.
"""
import json
from typing import Dict, Any, Optional
from pathlib import Path

from server.agent.tools.base_tool import BaseTool
from server.agent.concept_graph import load_pack


class ExpandContentTool(BaseTool):
    """Tool for expanding compressed observations and examples"""
    
    name = "expand_content"
    description = "Expand compressed content, retrieve raw data, or get detailed examples when summaries aren't sufficient"
    
    parameters = {
        "type": "object",
        "properties": {
            "content_type": {
                "type": "string",
                "enum": ["raw_content", "example", "observation"],
                "description": "Type of content to expand"
            },
            "identifier": {
                "type": "string", 
                "description": "Hash for raw content/observation, or node_id#example_id for examples"
            },
            "max_lines": {
                "type": "integer",
                "default": 50,
                "description": "Maximum lines to return for large content"
            }
        },
        "required": ["content_type", "identifier"]
    }
    
    def __init__(self):
        super().__init__()
        self.knowledge_store = None  # Will be set by the agent
    
    def set_knowledge_store(self, knowledge_store):
        """Set the knowledge store dependency"""
        self.knowledge_store = knowledge_store
    
    async def _execute(self, content_type: str, identifier: str, max_lines: int = 50, **kwargs) -> Dict[str, Any]:
        """Execute the content expansion"""
        
        try:
            if content_type == "raw_content":
                # Expand raw content from hash
                if not self.knowledge_store:
                    return {"error": "Knowledge store not available"}
                    
                raw_content = self.knowledge_store.retrieve_raw_content(identifier)
                if not raw_content:
                    return {"error": f"Raw content not found for hash: {identifier}"}
                
                # Return truncated version for large content
                lines = raw_content.split('\n')
                if len(lines) <= max_lines:
                    return {"content": raw_content, "truncated": False}
                else:
                    # Show beginning, middle indicator, and end
                    start_lines = lines[:max_lines//2]
                    end_lines = lines[-(max_lines//2):]
                    truncated_content = '\n'.join(start_lines) + f"\n... ({len(lines) - max_lines} lines omitted) ...\n" + '\n'.join(end_lines)
                    return {"content": truncated_content, "truncated": True, "total_lines": len(lines)}
            
            elif content_type == "observation":
                # Expand compressed observation
                if not self.knowledge_store:
                    return {"error": "Knowledge store not available"}
                    
                expanded = self.knowledge_store.expand_compressed_content(identifier, max_lines)
                return {"content": expanded}
            
            elif content_type == "example":
                # Expand example from node_id#example_id format
                if '#' not in identifier:
                    return {"error": "Example identifier must be in format: node_id#example_id"}
                
                node_id, example_id = identifier.split('#', 1)
                
                # Try to find the example in loaded packs
                # This is a simplified version - you might want to make this more robust
                example_content = self._find_example_in_packs(node_id, example_id)
                if example_content:
                    return {"content": example_content}
                else:
                    return {"error": f"Example not found: {identifier}"}
            
            else:
                return {"error": f"Unknown content type: {content_type}"}
                
        except Exception as e:
            return {"error": f"Failed to expand content: {str(e)}"}
    
    def _find_example_in_packs(self, node_id: str, example_id: str) -> Optional[str]:
        """Find example in available knowledge packs"""
        try:
            # Look for pack directories (this is simplified - you might want to make this configurable)
            pack_dirs = [
                "learning/packs/calc.v1",
                # Add other pack directories as needed
            ]
            
            for pack_dir in pack_dirs:
                pack_path = Path(pack_dir)
                if not pack_path.exists():
                    continue
                    
                try:
                    adj, nodes_by_id = load_pack(str(pack_path))
                    node = nodes_by_id.get(node_id)
                    
                    if node:
                        examples = node.get("examples", [])
                        # Parse example_id (e.g., "ex1" -> index 0, "ex2" -> index 1)
                        if example_id.startswith("ex"):
                            try:
                                ex_index = int(example_id[2:]) - 1  # ex1 -> 0, ex2 -> 1
                                if 0 <= ex_index < len(examples):
                                    example = examples[ex_index]
                                    if isinstance(example, dict):
                                        inp = example.get("input", "")
                                        out = example.get("output", "")
                                        return f"Input: {inp}\nOutput: {out}"
                            except ValueError:
                                continue
                                
                except Exception:
                    continue
                    
            return None
            
        except Exception:
            return None


# Register the tool for automatic discovery
expand_content_tool = ExpandContentTool()
