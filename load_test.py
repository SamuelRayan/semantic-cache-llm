import asyncio
import random
import httpx

BASE_URL = "http://localhost:8000"

TOPICS = [
    "the capital of France", "how photosynthesis works", "the plot of Romeo and Juliet",
    "how to bake sourdough bread", "the rules of chess", "quantum entanglement",
    "the history of the Roman empire", "how neural networks learn",
]

TEMPLATES = [
    "What is {t}?",
    "Can you explain {t}?",
    "Tell me about {t}.",
    "I'd like to understand {t}, can you help?",
    "Explain {t} to me like I'm new to this.",
]

async def send_one(client, prompt):
    r = await client.post(f"{BASE_URL}/v1/chat/completions", json={"prompt": prompt})
    return r.headers.get("X-Cache-Status"), r.elapsed.total_seconds()

async def main():
    async with httpx.AsyncClient(timeout=30) as client:
        results = {"HIT": 0, "MISS": 0}
        latencies = {"HIT": [], "MISS": []}
        for _ in range(300):
            topic = random.choice(TOPICS)
            template = random.choice(TEMPLATES)
            prompt = template.format(t=topic)
            status, latency = await send_one(client, prompt)
            results[status] += 1
            latencies[status].append(latency)

        total = results["HIT"] + results["MISS"]
        print(f"Total requests: {total}")
        print(f"Cache hit rate: {results['HIT'] / total:.1%}")
        if latencies["HIT"]:
            print(f"Avg latency HIT:  {sum(latencies['HIT']) / len(latencies['HIT']) * 1000:.1f}ms")
        if latencies["MISS"]:
            print(f"Avg latency MISS: {sum(latencies['MISS']) / len(latencies['MISS']) * 1000:.1f}ms")

if __name__ == "__main__":
    asyncio.run(main())