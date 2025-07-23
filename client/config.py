# config.py

"""
Central configuration for the Anton client application.
"""

# The base URL for the Anton agent server API.
# Centralizing this makes it easy to change for different environments (e.g., development, production).
API_BASE_URL = "http://localhost:8000"

# Default timeout for API requests in seconds.
DEFAULT_TIMEOUT = 300.0

# Default temperature for the agent's chat completions.
DEFAULT_TEMPERATURE = 0.6
