import os
from dataclasses import dataclass

@dataclass
class Settings:
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    similarity_threshold:float=0.92
    default_ttl_seconds: int=60*60*24
    short_ttl_seconds: int = 60*60
    data_dir: str="data"
    sqlite_path: str=os.path.join(data_dir,"cache.db")
    faiss_index_path: str=os.path.join(data_dir, "cache.index")
    faiss_ids_path: str=os.path.join(data_dir, "cache_ids.json")
    use_mock_llm: bool=os.getenv("USE-MOCK-LLM", "true").lower() == "true"
    openai_api_key : str=os.getenv("OPENAI_API_KEY", "")
    sync_scoring: bool=os.getenv("SYNC_SCORING", "false").lower() == "true"

settings=Settings()
