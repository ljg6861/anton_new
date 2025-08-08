"""
Manages the Retrieval-Augmented Generation (RAG) knowledge base.

Handles embedding, storing, and retrieving knowledge snippets using
a FAISS vector store and a sentence-transformer model.
"""
import os
import pickle
import logging
from typing import List, Dict, Any, Optional

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Configure a basic logger
logger = logging.getLogger(__name__)

class RAGManager:
    """
    A singleton class to manage the FAISS vector store and document store for the agent's knowledge.

    ### CHANGED ###
    - Refactored the singleton pattern to be more robust.
    - Decoupled adding knowledge from saving to disk for huge performance gains on batch additions.
    - Added comprehensive type hints.
    """
    _instance: Optional['RAGManager'] = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            logger.info("Creating a new RAGManager instance.")
            cls._instance = super(RAGManager, cls).__new__(cls)
            # Initialize on creation only
            cls._instance._init_rag(*args, **kwargs)
        return cls._instance

    def _init_rag(self, model_name: str = 'all-MiniLM-L6-v2', index_path: str = 'knowledge.index', doc_store_path: str = 'documents.pkl'):
        """Initializes the RAG components. This method is called only once."""
        logger.info("Initializing RAGManager components...")
        self.model: SentenceTransformer = SentenceTransformer(model_name)
        self.embedding_dim: int = self.model.get_sentence_embedding_dimension()
        self.index_path: str = index_path
        self.doc_store_path: str = doc_store_path

        self.index: Optional[faiss.Index] = None
        self.doc_store: Dict[int, Dict[str, Any]] = {}

        self._load()
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
        # IndexFlatL2 is a brute-force index. Good for up to ~1M vectors.
        self.index = faiss.IndexFlatL2(self.embedding_dim)

        ### NOTE: For very large datasets, consider a more advanced index for scalability.
        # This requires a "training" step on your data before adding.
        # nlist = 100  # Number of cells/clusters
        # quantizer = faiss.IndexFlatL2(self.embedding_dim)
        # self.index = faiss.IndexIVFPQ(quantizer, self.embedding_dim, nlist, 8, 8) # 8 bytes per vector, 8-bit precision

    def save(self):
        """
        ### CHANGED ###
        Saves the current state of the index and document store to disk.
        This is now an explicit public method.
        """
        if self.index is None:
            logger.error("Cannot save, index is not initialized.")
            return

        logger.info(f"Saving knowledge base with {self.index.ntotal} entries to disk...")
        try:
            faiss.write_index(self.index, self.index_path)
            # ### NOTE ###: Pickle is convenient but can be insecure and break between library versions.
            # For production, consider using a safer format like JSONL or a lightweight DB.
            with open(self.doc_store_path, 'wb') as f:
                pickle.dump(self.doc_store, f)
            logger.info("Knowledge base saved successfully.")
        except Exception as e:
            logger.error(f"Failed to save knowledge base: {e}")

    def add_knowledge(self, text: str, source: str):
        """
        Adds a text snippet and its source to the knowledge base.

        ### CHANGED ### - This method no longer saves to disk automatically.
        Call .save() explicitly after adding one or more documents.
        """
        try:
            # The model can encode a list of sentences, so we wrap `text` in a list.
            embedding = self.model.encode([text])
            self.index.add(np.array(embedding, dtype=np.float32))

            # The ID for the new entry is its position in the index (0-based).
            new_id = self.index.ntotal - 1
            self.doc_store[new_id] = {'text': text, 'source': source}

            logger.debug(f"Added new knowledge from source '{source}' to in-memory index.")
        except Exception as e:
            logger.error(f"Failed to add knowledge: {e}", exc_info=True)

    def retrieve_knowledge(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieves the top_k most relevant documents for a given query."""
        if self.index.ntotal == 0:
            logger.warning("Attempted to retrieve knowledge from an empty index.")
            return []

        try:
            query_embedding = self.model.encode([query])
            distances, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)

            # Retrieve the documents corresponding to the top indices, ensuring they exist.
            results = [self.doc_store[i] for i in indices[0] if i in self.doc_store and i != -1]
            return results
        except Exception as e:
            logger.error(f"Failed to retrieve knowledge: {e}", exc_info=True)
            return []

    def rebuild_index(self) -> int:
        """
        Rebuild the FAISS index from the current document store to remove orphaned vectors.
        This addresses the issue where deleted documents leave vectors in the FAISS index.
        
        Returns:
            Number of documents in the rebuilt index
        """
        if not self.doc_store:
            # If no documents, create an empty index
            self.index = faiss.IndexFlatL2(self.embedding_dim)
            logger.info("Rebuilt empty FAISS index")
            return 0
        
        try:
            logger.info(f"Rebuilding FAISS index from {len(self.doc_store)} documents...")
            
            # Create new index
            new_index = faiss.IndexFlatL2(self.embedding_dim)
            
            # Re-embed all current documents and add to new index
            texts = []
            doc_ids = []
            
            # Collect texts in order of document IDs
            for doc_id in sorted(self.doc_store.keys()):
                if doc_id in self.doc_store:
                    texts.append(self.doc_store[doc_id]['text'])
                    doc_ids.append(doc_id)
            
            if texts:
                # Encode all texts at once for efficiency
                embeddings = self.model.encode(texts)
                embeddings_array = np.array(embeddings, dtype=np.float32)
                
                # Add to the new index
                new_index.add(embeddings_array)
                logger.info(f"Added {len(embeddings)} embeddings to rebuilt index")
            
            # Replace the old index
            self.index = new_index
            
            # Create a new document store with sequential IDs
            new_doc_store = {}
            for i, old_doc_id in enumerate(doc_ids):
                new_doc_store[i] = self.doc_store[old_doc_id]
            
            self.doc_store = new_doc_store
            
            logger.info(f"Successfully rebuilt FAISS index with {self.index.ntotal} vectors")
            return self.index.ntotal
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}", exc_info=True)
            return 0

# Create a singleton instance to be imported by other modules
rag_manager = RAGManager()