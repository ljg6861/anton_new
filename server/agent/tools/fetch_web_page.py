import requests
import json
from bs4 import BeautifulSoup
import re

class FetchWebPageTool:
    function = {
        "type": "function",
        "function": {
            "name": "fetch_web_page",
            "description": "Fetch and extract clean text content from a public URL, removing HTML markup and formatting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL to fetch (must start with http:// or https://)."
                    },
                    "timeout": {
                        "type": "integer",
                        "description": "Request timeout in seconds. Defaults to 10 if not specified."
                    }
                },
                "required": ["url"]
            }
        }
    }

    def _extract_clean_text(self, html_content: str, max_length: int = 10000) -> str:
        """Extract clean text from HTML content, removing markup and excess whitespace."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            # Remove excessive newlines and spaces
            text = re.sub(r'\n\s*\n', '\n\n', text)
            text = re.sub(r' +', ' ', text)
            
            # Truncate if too long to prevent context overload
            if len(text) > max_length:
                text = text[:max_length] + "... [Content truncated to prevent context overload]"
            
            return text.strip()
        except Exception as e:
            return f"Error extracting text: {str(e)}"

    def run(self, arguments: dict) -> str:
        url = arguments.get('url')
        timeout = arguments.get('timeout', 10)

        if not url or not isinstance(url, str):
            return "❌ Error: Invalid or missing URL"

        if not url.lower().startswith("http"):
            return "❌ Error: Only HTTP/HTTPS URLs are supported"

        # Check robots.txt (simplified for now; could be expanded)
        if 'robots.txt' in url:
            return "❌ Error: Access denied by robots.txt"

        # Apply rate limiting (simulated)
        import time
        if hasattr(self, '_last_request_time'):
            elapsed = time.time() - self._last_request_time
            if elapsed < 1.0:  # 1 second minimum between requests
                time.sleep(1.0 - elapsed)
        self._last_request_time = time.time()

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Connection': 'keep-alive'
        }

        try:
            response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
            
            if not response.ok:
                return f"❌ Error: HTTP {response.status_code} - {response.reason}"
            
            content_type = response.headers.get('Content-Type', '').lower()
            
            # Check if it's HTML content
            if 'text/html' not in content_type:
                return f"❌ Error: URL does not return HTML content (Content-Type: {content_type})"
            
            # Extract clean text from HTML
            clean_text = self._extract_clean_text(response.text)
            
            if not clean_text or clean_text.startswith("Error extracting text"):
                return f"❌ {clean_text}"
            
            return f"✅ Content from {url}:\n\n{clean_text}"
        
        except requests.exceptions.RequestException as e:
            return f"❌ Network error: {str(e)}"

        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"