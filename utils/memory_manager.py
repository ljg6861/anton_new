# FILE: utils/memory_manager.py

import chromadb
import logging
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

# This can be initialized once and imported
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device="cpu")

class MemoryManager:
    """
    Manages storing and retrieving pre-evaluated memories using a vector database.
    """
    def __init__(self):
        self.embedding_model = embedding_model
        logger.info("Initializing Memory Manager with ChromaDB on CPU...")
        self.chroma_client = chromadb.PersistentClient(path="./chroma_db")
        self.collection = self.chroma_client.get_or_create_collection(name="anton_memories")
        logger.info("✅ MemoryManager is ready.")

    def store_memory(self, memory_text: str):
        """
        Creates an embedding for a given piece of text and stores it.
        """
        if not memory_text:
            logger.warning("Attempted to store an empty memory.")
            return

        try:
            embedding = self.embedding_model.encode(memory_text).tolist()
            # Use a hash of the content as a unique ID to prevent duplicates
            memory_id = str(hash(memory_text))

            # Upsert handles both creation and updates gracefully
            self.collection.upsert(
                embeddings=[embedding],
                documents=[memory_text],
                metadatas=[{"source": "conversation_learning"}],
                ids=[memory_id]
            )
            logger.info(f"✅ Memory stored: '{memory_text}'")
        except Exception as e:
            logger.error(f"Failed to create embedding or store memory: {e}")

    def query_memories(self, query_text: str, k: int = 3) -> list:
        """
        Searches for the k most relevant memories based on a query string.
        """
        if not query_text:
            return []
        query_embedding = self.embedding_model.encode(query_text).tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results.get('documents', [[]])[0]