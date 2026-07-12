import os
import json
import faiss
import numpy as np
from app.config import settings

class VectorStore:
    def __init__(self):
        os.makedirs(settings.data_dir, exist_ok=True)
        self.dim = settings.embedding_dim
        self.index = faiss.IndexFlatIP(self.dim)
        self.ids: list[str] = []
        self._load()

    def _load(self):
        if os.path.exists(settings.faiss_index_path) and os.path.exists(settings.faiss_ids_path):
            self.index = faiss.read_index(settings.faiss_index_path)
            with open(settings.faiss_ids_path) as f:
                self.ids = json.load(f)

    def save(self):
        faiss.write_index(self.index, settings.faiss_index_path)
        with open(settings.faiss_ids_path, "w") as f:
            json.dump(self.ids, f)

    def add(self, cache_id: str, vector: np.ndarray):
        self.index.add(np.expand_dims(vector, axis=0))
        self.ids.append(cache_id)
        self.save()

    def search(self, vector: np.ndarray, top_k: int = 5):
        if self.index.ntotal == 0:
            return []
        scores, positions = self.index.search(np.expand_dims(vector, axis=0), min(top_k, self.index.ntotal))
        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            results.append((self.ids[pos], float(score)))
        return results

vector_store = VectorStore()