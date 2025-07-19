import asyncio
import logging
import os
import torch  # Import torch to check for CUDA
import chainlit as cl
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

os.environ["HF_HOME"] = "./hf_cache"
logger = logging.getLogger(__name__)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# --- Conversational Memory System ---
class ConversationalMemorySystem:

    def __init__(self, embedding_model_name: str):
        """
        Initializes the memory system, dynamically selecting the best device for embeddings.
        """
        # Dynamically determine the best device for the embedding model
        device = "cpu"
        if torch.cuda.is_available():
            # If multiple GPUs are available, default to the second one for embeddings,
            # assuming the primary one might be busy with a larger LLM.
            if torch.cuda.device_count() > 1:
                device = "cuda:1"
            else:
                # Otherwise, use the only available GPU
                device = "cuda:0"

        logger.info(f"🧠 Initializing Conversational Memory on device: '{device}'")

        model_kwargs = {'device': device}
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs=model_kwargs
        )

        # Initialize an empty FAISS vector store with a dummy entry
        dummy_text = "system initialization"
        self.vector_store = FAISS.from_texts([dummy_text], self.embeddings)
        # Immediately remove the dummy entry to start with a clean slate
        if self.vector_store.index.ntotal > 0:
            self.vector_store.delete([self.vector_store.index_to_docstore_id[0]])

        self.memory_count = 0
        logger.info("✅ Conversational Memory System initialized successfully.")

    def retrieve(self, query: str, k: int = 2) -> str:
        """
        Retrieves the `k` most relevant memories for a given query.
        """
        if self.memory_count == 0:
            return "No relevant memories found in this session."

        # Ensure we don't try to retrieve more items than exist
        effective_k = min(k, self.memory_count)

        retrieved_docs = self.vector_store.similarity_search(query, k=effective_k)

        if not retrieved_docs:
            return "No relevant memories found for this query."

        context = "\n\n---\n\n".join([doc.page_content for doc in retrieved_docs])
        return context

    async def add_to_memory(self, text_chunk: str):
        """
        Asynchronously adds a new text chunk to the conversational memory.
        """
        # Langchain's add_texts is synchronous, so we use Chainlit's helper to await it
        add_texts_async = cl.make_async(self.vector_store.add_texts)

        await add_texts_async([text_chunk])
        self.memory_count += 1
        logger.info(f"Memory updated. Total memories: {self.memory_count}")