from typing import Any, Dict
import requests
from pydantic import BaseModel

class FetchWebpage(BaseModel):
    name: str = "fetch_webpage"
    description: str = "Fetches the content of a web page given a URL. Returns truncated output to prevent token overflow."
    
    def get_metadata(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "format": "uri",
                        "description": "The URL to fetch (must include http:// or https://)"
                    }
                },
                "required": ["url"]
            }
        }
    
    def get_function_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.get_metadata()["parameters"]
        }
    
    def run(self, url: str) -> str:
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            content = response.text
            # Truncate output to prevent token overflow
            return content[:2048]  # Limit to first 2KB
        except Exception as e:
            return f"Error fetching web page: {str(e)}"