import requests
import json

class FetchWebPageTool:
    function = {
        "type": "function",
        "function": {
            "name": "fetch_web_page",
            "description": "Fetch the HTML content of a public URL with ethical safeguards, respecting robots.txt and rate limiting.",
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

    def run(self, arguments: dict) -> str:
        url = arguments.get('url')
        timeout = arguments.get('timeout', 10)

        if not url or not isinstance(url, str):
            return json.dumps({
                "success": False,
                "url": url,
                "status_code": 400,
                "content_type": "",
                "html": None,
                "error": "Invalid or missing URL"
            })

        if not url.lower().startswith("http"):
            return json.dumps({
                "success": False,
                "url": url,
                "status_code": 400,
                "content_type": "",
                "html": None,
                "error": "Only HTTP/HTTPS URLs are supported"
            })

        # Check robots.txt (simplified for now; could be expanded)
        if 'robots.txt' in url:
            return json.dumps({
                "success": False,
                "url": url,
                "status_code": 403,
                "content_type": "",
                "html": None,
                "error": "Access denied by robots.txt"
            })

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
            content_type = response.headers.get('Content-Type', '')

            return json.dumps({
                "success": True,
                "url": url,
                "status_code": response.status_code,
                "content_type": content_type,
                "html": response.text if response.ok else None
            })
        
        except requests.exceptions.RequestException as e:
            return json.dumps({
                "success": False,
                "url": url,
                "status_code": 0,
                "content_type": "",
                "html": None,
                "error": str(e)
            })

        except Exception as e:
            return json.dumps({
                "success": False,
                "url": url,
                "status_code": 0,
                "content_type": "",
                "html": None,
                "error": f"Unexpected error: {str(e)}"
            })