# /multi_agent_project/main.py

import logging
from typing import List, Dict

from orchestrator import MultiStepAgentOrchestrator

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    """
    Main function to run the chat application.
    """
    orchestrator = MultiStepAgentOrchestrator()
    chat_history: List[Dict[str, str]] = []

    print("Welcome to the Multi-Agent Chat! Type 'exit' to quit.")

    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break

        print("\n--- Agent Output Stream ---")
        full_response = ""
        # The generator yields parts of the process as they happen.
        for chunk in orchestrator.stream(user_input=user_query, chat_history=chat_history):
            print(chunk, end="", flush=True)
            full_response += chunk

        # Update chat history
        chat_history.append({"role": "user", "content": user_query})
        # Extract only the final answer part for the history
        final_answer = full_response.split("--- Final Answer ---")[-1].strip()
        chat_history.append({"role": "assistant", "content": final_answer})
        print("\n--- End of Response ---")


if __name__ == "__main__":
    main()