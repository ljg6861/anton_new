import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import json  # Import json for a better return format


class WebSearchTool:
    function = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web and extracts useful data from the top links.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "num_results": {"type": "integer", "description": "Number of top results to return."}
                },
                "required": ["query"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        query = arguments.get('query')
        # Use a default of 3 results if not specified
        num_results = arguments.get('num_results', 3)

        # Use DuckDuckGo Search API for reliable results
        search_results = []
        with DDGS() as ddgs:
            search_results = list(ddgs.text(
                keywords=query,
                region='us-en',
                safesearch='off',
                max_results=num_results
            ))

        if not search_results:
            return json.dumps([{"error": "No search results found.", "query": query}])

        # Extract data from each link
        results = []
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'}

        for result in search_results:
            link = result.get('href')
            if not link:
                continue

            try:
                # Set a timeout to prevent hanging on slow sites
                response = requests.get(link, headers=headers, timeout=10)
                response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)

                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract title and meta description
                title = soup.title.string.strip() if soup.title else 'No title found'

                meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
                description = meta_desc_tag[
                    'content'].strip() if meta_desc_tag and 'content' in meta_desc_tag.attrs else 'No description found'

                results.append({
                    'title': title,
                    'url': link,
                    'description': description
                })
            # More specific exception handling
            except requests.exceptions.RequestException as e:
                results.append({
                    'error': f'Failed to fetch URL: {str(e)}',
                    'url': link
                })
            except Exception as e:
                results.append({
                    'error': f'An unexpected error occurred while processing URL: {str(e)}',
                    'url': link
                })

        # Return a JSON string for easy parsing
        return json.dumps(results, indent=2)