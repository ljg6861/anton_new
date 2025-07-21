# agent/models.py

"""
Contains Pydantic models for data structures used throughout the agent.
"""
from pydantic import BaseModel


class AgentAction(BaseModel):
    """
    A model representing a specific action to be taken by the agent.

    Note: This model was in the original code but is not actively used
    in the agent loop. It is preserved here for potential future use.
    """
    action: str
    data: list | str