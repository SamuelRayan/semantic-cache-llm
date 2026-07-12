from sentence_transformers import SentenceTransformer
from app.config import settings

class EmbeddingService:
    def __init__(self):
        self.model=SentenceTransformer(settings.embedding_model)

    def embed(self,text:str):
        vec=self.model.encode(text,normalize_embeddings=True)
        return vec.astype("float32")

embedding_service=EmbeddingService()