# config.py

"""
Central configuration for the Anton client application.
"""

import os

# The base URL for the Anton agent server API.
# Centralizing this makes it easy to change for different environments (e.g., development, production).
AGENT_PORT = os.getenv("AGENT_PORT", "8001")

# Determine the host based on environment
ANTON_ENV = os.getenv("ANTON_ENV", "dev")
if ANTON_ENV == "prod":
    # Production uses the network IP for external access
    AGENT_HOST = os.getenv("AGENT_HOST", "192.168.1.250")
else:
    # Development can use localhost
    AGENT_HOST = os.getenv("AGENT_HOST", "localhost")

API_BASE_URL = f"http://{AGENT_HOST}:{AGENT_PORT}"

# Default timeout for API requests in seconds.
DEFAULT_TIMEOUT = 300.0

# Default temperature for the agent's chat completions.
DEFAULT_TEMPERATURE = 0.6
