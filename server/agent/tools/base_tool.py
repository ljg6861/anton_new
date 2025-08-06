"""
Base tool interface for the dynamic tool management system.
Provides standardized metadata, versioning, and capability definitions.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class ToolCapability(Enum):
    """Enumeration of tool capabilities for better discovery and categorization."""
    FILE_SYSTEM = "file_system"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
    GIT_OPERATIONS = "git_operations"
    TOOL_CREATION = "tool_creation"
    DATA_PROCESSING = "data_processing"
    EXTERNAL_API = "external_api"


@dataclass
class ToolMetadata:
    """Metadata container for tool information."""
    name: str
    version: str
    description: str
    capabilities: List[ToolCapability]
    author: Optional[str] = None
    dependencies: Optional[List[str]] = None
    tags: Optional[List[str]] = None


class BaseTool(ABC):
    """
    Abstract base class for all tools in the system.
    Provides standardized interface with metadata and versioning support.
    """
    
    def __init__(self):
        """Initialize the tool with its metadata."""
        self._metadata = self.get_metadata()
        self._function_schema = self.get_function_schema()
    
    @property
    def metadata(self) -> ToolMetadata:
        """Get tool metadata."""
        return self._metadata
    
    @property
    def function(self) -> Dict[str, Any]:
        """Get the tool's function schema for LLM consumption."""
        return self._function_schema
    
    @abstractmethod
    def get_metadata(self) -> ToolMetadata:
        """
        Return metadata about this tool.
        Must be implemented by each tool.
        """
        pass
    
    @abstractmethod
    def get_function_schema(self) -> Dict[str, Any]:
        """
        Return the OpenAI function schema for this tool.
        Must be implemented by each tool.
        """
        pass
    
    @abstractmethod
    def run(self, arguments: Dict[str, Any]) -> str:
        """
        Execute the tool with the given arguments.
        Must be implemented by each tool.
        
        Args:
            arguments: Dictionary of arguments for the tool
            
        Returns:
            String result of the tool execution
        """
        pass
    
    def validate_arguments(self, arguments: Dict[str, Any]) -> bool:
        """
        Validate arguments against the function schema.
        Override this method for custom validation logic.
        
        Args:
            arguments: Arguments to validate
            
        Returns:
            True if arguments are valid, False otherwise
        """
        required = self._function_schema.get("function", {}).get("parameters", {}).get("required", [])
        return all(arg in arguments for arg in required)
    
    def get_full_name(self) -> str:
        """Get the full name including version for conflict resolution."""
        return f"{self._metadata.name}_v{self._metadata.version.replace('.', '_')}"
    
    def is_compatible_version(self, required_version: str) -> bool:
        """
        Check if this tool version is compatible with the required version.
        Uses simple semantic versioning rules.
        
        Args:
            required_version: Required version string (e.g., "1.0.0")
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            current = [int(x) for x in self._metadata.version.split('.')]
            required = [int(x) for x in required_version.split('.')]
            
            # Major version must match exactly
            if current[0] != required[0]:
                return False
            
            # Minor version must be >= required
            if len(current) > 1 and len(required) > 1:
                if current[1] < required[1]:
                    return False
            
            return True
        except (ValueError, IndexError):
            return False