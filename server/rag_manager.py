import faiss
import numpy as np
import os
import json
from sentence_transformers import SentenceTransformer

class RAGManager:
    def __init__(self, model_name='all-MiniLM-L6-v2', index_path='knowledge.index', doc_store_path='documents.json'):
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        self.index_path = index_path
        self.doc_store_path = doc_store_path
        self.index = None
        self.doc_store = {}
        self._load()

    def _load(self):
        if os.path.exists(self.index_path) and os.path.exists(self.doc_store_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.doc_store_path, 'r') as f:
                    self.doc_store = json.load(f)
                print(f'Loaded existing knowledge base from disk. {self.index.ntotal} entries')
            except Exception as e:
                print(f'Error loading knowledge base: {e}')
                self.index = faiss.IndexFlatL2(self.embedding_dim)
        else:
            self.index = faiss.IndexFlatL2(self.embedding_dim)

    def add_document(self, text, metadata=None):
        embedding = self.model.encode([text])[0]
        embedding = np.array(embedding).reshape(1, -1)
        self.index.add(embedding)
        doc_id = str(len(self.doc_store))
        self.doc_store[doc_id] = {
            'text': text,
            'metadata': metadata or {}
        }
        self._save()

    def _save(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.doc_store_path, 'w') as f:
            json.dump(self.doc_store, f)

    def search(self, query, k=5):
        query_embedding = self.model.encode([query])[0]
        query_embedding = np.array(query_embedding).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        results = []
        for i, idx in enumerate(indices[0]):
            doc_id = str(idx)
            if doc_id in self.doc_store:
                results.append({
                    'text': self.doc_store[doc_id]['text'],
                    'metadata': self.doc_store[doc_id]['metadata'],
                    'distance': float(distances[0][i])
                })
        return results