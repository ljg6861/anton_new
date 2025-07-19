import logging
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import uuid
from typing import Dict, List

# Assume orchestrator.py and other agent files are in the same directory
# or a discoverable path.
from orchestrator import MultiStepAgentOrchestrator

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

# --- In-Memory Session Management ---
# In a production environment, you would use a more robust session store
# like Redis, a database, or server-side session cookies.
SESSIONS: Dict[str, List[Dict[str, str]]] = {}
ORCHESTRATOR = MultiStepAgentOrchestrator()

@app.route('/start', methods=['GET'])
def start_session():
    """
    Starts a new chat session and returns a unique session ID.
    """
    session_id = str(uuid.uuid4())
    SESSIONS[session_id] = []  # Initialize empty chat history
    logger.info(f"New session started: {session_id}")
    return jsonify({"session_id": session_id})

@app.route('/chat', methods=['POST'])
def chat():
    """
    Handles a chat message from the user, streams the agent's response.
    """
    data = request.json
    session_id = data.get('session_id')
    user_message = data.get('message')

    if not session_id or session_id not in SESSIONS:
        logger.error(f"Invalid or missing session_id: {session_id}")
        return jsonify({"error": "Invalid or missing session ID"}), 400

    if not user_message:
        logger.error("Empty message received.")
        return jsonify({"error": "Message cannot be empty"}), 400

    logger.info(f"Received message for session {session_id}: {user_message}")

    # Get the chat history for the current session
    chat_history = SESSIONS[session_id]

    def generate_response():
        """
        A generator function that yields chunks of the agent's response.
        """
        try:
            full_response = ""
            # The orchestrator's stream method is a generator. We yield from it.
            for chunk in ORCHESTRATOR.stream(user_input=user_message, chat_history=chat_history):
                full_response += chunk
                yield chunk

            # Once the stream is complete, update the session's chat history
            chat_history.append({"role": "user", "content": user_message})
            # Extract only the final answer for a clean history
            final_answer = full_response.split("--- Final Answer ---")[-1].strip()
            chat_history.append({"role": "assistant", "content": final_answer})
            SESSIONS[session_id] = chat_history # Update the history
            logger.info(f"Session {session_id} history updated.")

        except Exception as e:
            logger.error(f"An error occurred during response generation: {e}", exc_info=True)
            yield "Sorry, an error occurred while processing your request."

    # Return a streaming response
    return Response(generate_response(), mimetype='text/plain')

if __name__ == '__main__':
    # Note: This is a development server. For production, use a proper WSGI server like Gunicorn.
    app.run(host='0.0.0.0', port=5001, debug=True)
