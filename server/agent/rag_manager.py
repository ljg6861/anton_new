"""
Manages the Retrieval-Augmented Generation (RAG) knowledge base.

Handles embedding, storing, and retrieving knowledge snippets using
a FAISS vector store and a sentence-transformer model.
"""
import os
import pickle
import logging
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure a basic logger
logger = logging.getLogger(__name__)

class RAGManager:
    """
    A singleton class to manage the FAISS vector store and document store for the agent's knowledge.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(RAGManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'knowledge.index', doc_store_path: str = 'documents.pkl'):
        # This check prevents re-initialization on subsequent calls
        if hasattr(self, 'initialized'):
            return

        logger.info("Initializing RAGManager...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index_path = index_path
        self.doc_store_path = doc_store_path

        self._load()
        self.initialized = True
        logger.info(f"RAGManager initialized. Knowledge base contains {self.index.ntotal} entries.")

    def _load(self):
        """Loads the FAISS index and document store from disk if they exist."""
        if os.path.exists(self.index_path) and os.path.exists(self.doc_store_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.doc_store_path, 'rb') as f:
                    self.doc_store = pickle.load(f)
                logger.info("Loaded existing knowledge base from disk.")
            except Exception as e:
                logger.error(f"Error loading knowledge base, initializing a new one: {e}")
                self._initialize_empty_stores()
        else:
            self._initialize_empty_stores()

    def _initialize_empty_stores(self):
        """Initializes a new, empty FAISS index and document store."""
        logger.info("Creating a new, empty knowledge base.")
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.doc_store = {}  # A simple dictionary mapping index ID to document content

    def _save(self):
        """Saves the current state of the index and document store to disk."""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.doc_store_path, 'wb') as f:
                pickle.dump(self.doc_store, f)
            logger.debug("Knowledge base saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def add_knowledge(self, text: str, source: str):
        try:
            embedding = self.model.encode([text])
            self.index.add(np.array(embedding, dtype=np.float32))

            # The ID for the new entry is its position in the index
            new_id = self.index.ntotal - 1
            self.doc_store[new_id] = {'text': text, 'source': source}

            self._save()
            logger.info(f"Added new knowledge from source '{source}'.")
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)

    def retrieve_knowledge(self, query: str, top_k: int = 3) -> list[dict]:
        if self.index.ntotal == 0:
            return []

        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)

            # Retrieve the documents corresponding to the top indices
            results = [self.doc_store[i] for i in indices[0] if i in self.doc_store]
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}", exc_info=True)
            return []

# Create a singleton instance to be imported by other modules
rag_manager = RAGManager()