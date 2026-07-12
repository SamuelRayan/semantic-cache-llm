import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"..")))

from app.cache_engine import get_or_generate

prompts = [
    "What is the capital of France?",
    "Can you tell me France's capital city?",     # semantically similar -> should HIT
    "What's the weather like today in Paris?",     # unique + time-sensitive -> short TTL
    "What is the capital of France?",               # exact repeat -> should HIT
]

for p in prompts:
    result = get_or_generate(p)
    print(f"PROMPT: {p}")
    print(f"    STATUS: {result.cache_status} SIMILARITY: {result.similarity} LATENCY: {result.latency_ms: .1f}ms")
    print(f"    RESPONSE: {result.text}\n")