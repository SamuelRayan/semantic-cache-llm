import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.embeddings import embedding_service
import numpy as np

a = embedding_service.embed("What is the capital of France?")
b = embedding_service.embed("Can you tell me France's capital city?")

similarity = float(np.dot(a, b))  # vectors are already normalized, so dot product = cosine similarity
print(f"Similarity: {similarity:.4f}")