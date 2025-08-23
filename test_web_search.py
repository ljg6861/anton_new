#!/usr/bin/env python3
"""
Test the web search tool to see why it's returning irrelevant results.
"""
import sys
sys.path.append('/home/lucas/anton_new')

from server.agent.tools.web_search import WebSearchTool
import json

def test_web_search():
    print("🔍 Testing Web Search Tool...")
    
    tool = WebSearchTool()
    
    # Test the exact query that failed
    test_queries = [
        "Crazy Train Ozzy Osbourne song details",
        '"Crazy Train" Ozzy Osbourne music analysis',
        "Randy Rhoads Crazy Train guitar analysis"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n--- Test {i}: '{query}' ---")
        try:
            result = tool.run({"query": query, "num_results": 3, "fetch_content": False})
            parsed_result = json.loads(result)
            
            print(f"Found {len(parsed_result)} results:")
            for j, item in enumerate(parsed_result[:3], 1):
                if 'error' in item:
                    print(f"  {j}. ERROR: {item['error']}")
                else:
                    print(f"  {j}. {item.get('title', 'No title')}")
                    print(f"     URL: {item.get('url', 'No URL')}")
                    print(f"     Desc: {item.get('description', 'No desc')[:100]}...")
                    
        except Exception as e:
            print(f"  ERROR: {str(e)}")
    
    return True

if __name__ == "__main__":
    test_web_search()
