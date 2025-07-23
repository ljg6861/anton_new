import subprocess


def get_git_diff() -> str:
    """
    Retrieves the diff of staged and unstaged changes in the repository.
    """
    try:
        # Get diff for staged files
        staged_diff = subprocess.run(
            ['git', 'diff', '--staged'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        ).stdout.strip()

        # Get diff for unstaged files
        unstaged_diff = subprocess.run(
            ['git', 'diff'],
            capture_output=True, text=True, check=True, encoding='utf-8'
        ).stdout.strip()

        if not staged_diff and not unstaged_diff:
            return "No pending code changes."

        full_diff = ""
        if staged_diff:
            full_diff += "--- Staged Changes ---\n" + staged_diff + "\n"
        if unstaged_diff:
            full_diff += "--- Unstaged Changes ---\n" + unstaged_diff + "\n"

        return full_diff.strip()

    except Exception as e:
        # This can happen if not in a git repo or git is not installed
        print(f"Could not get git diff: {e}")
        return "Could not retrieve git diff information."


def get_relevant_memories(user_prompt: str, memory_manager, top_k: int = 3) -> str:
    """
    Finds the most relevant past conversations from the vector store.
    """
    try:
        # 1. Create an embedding of the user's current question
        query_embedding = memory_manager.embedding_model.encode(user_prompt).tolist()

        # 2. Search the ChromaDB collection
        results = memory_manager.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        documents = results.get('documents', [[]])[0]
        if not documents:
            return "No relevant memories found."

        # 3. Format the retrieved memories into a string
        context_str = "\n## Relevant Memories (from past conversations):\n"
        for doc in documents:
            context_str += f"- {doc}\n"

        return context_str
    except Exception as e:
        return "Could not retrieve memories."