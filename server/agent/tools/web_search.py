import requests
from bs4 import BeautifulSoup
import ddgs
import json  # Import json for a better return format


class WebSearchTool:
    function = {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Searches the web and extracts useful data from the top links. Automatically fetches content from the most relevant results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query."},
                    "num_results": {"type": "integer", "description": "Number of top results to return."},
                    "fetch_content": {"type": "boolean", "description": "Whether to fetch full content from the most relevant results. Defaults to True."}
                },
                "required": ["query"]
            }
        }
    }

    def run(self, arguments: dict) -> str:
        query = arguments.get('query')
        # Use a default of 3 results if not specified
        num_results = arguments.get('num_results', 3)
        # Default to fetching content unless explicitly disabled
        fetch_content = arguments.get('fetch_content', True)

        # Use DuckDuckGo Search API for reliable results
        search_results = []
        search_results = list(ddgs.DDGS().text(
                query=query,
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

        for i, result in enumerate(search_results):
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

                result_entry = {
                    'title': title,
                    'url': link,
                    'description': description
                }
                
                # Fetch content from the first 2 most relevant results if requested
                if fetch_content and i < 2:
                    try:
                        # Extract main content from the page
                        content = self._extract_main_content(soup)
                        if content and len(content.strip()) > 100:  # Only include if we got substantial content
                            result_entry['content'] = content[:3000]  # Limit to 3000 chars
                            result_entry['content_preview'] = content[:500] + "..." if len(content) > 500 else content
                    except Exception as content_error:
                        result_entry['content_error'] = f"Failed to extract content: {str(content_error)}"

                results.append(result_entry)
                
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
    
    def _extract_main_content(self, soup):
        """Extract main content from BeautifulSoup object, removing navigation, ads, etc."""
        # Remove elements that typically don't contain main content
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "iframe", "noscript"]):
            element.decompose()
            
        # Remove elements with common ad/navigation class names
        for element in soup.find_all(class_=lambda x: x and any(
            keyword in x.lower() for keyword in ['ad', 'advertisement', 'sidebar', 'navigation', 'menu', 'footer', 'header']
        )):
            element.decompose()
            
        # Try to find main content areas
        main_content = None
        
        # Look for semantic HTML5 elements first
        for tag in ['main', 'article']:
            element = soup.find(tag)
            if element:
                main_content = element
                break
                
        # If no semantic elements, look for content divs
        if not main_content:
            for selector in ['.content', '.main-content', '.article-content', '.post-content', '#content', '#main']:
                element = soup.select_one(selector)
                if element:
                    main_content = element
                    break
        
        # Fallback: use the body if nothing else found
        if not main_content:
            main_content = soup.find('body')
            
        if main_content:
            text = main_content.get_text(separator=' ', strip=True)
            # Clean up excessive whitespace
            import re
            text = re.sub(r'\s+', ' ', text)
            return text.strip()
            
        return ""