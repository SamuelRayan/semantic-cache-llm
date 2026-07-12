import hashlib
import time
from abc import ABC, abstractmethod
from app.config import settings

class LLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> dict:
        ...

class MockProvider(LLMProvider):
    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> dict:
        time.sleep(0.6)
        seed = hashlib.md5((system_prompt + prompt).encode()).hexdigest()[:8]
        fake_answer = f"[mock-{seed}] Simulated answer to: '{prompt[:60]}'"
        input_tokens = max(1, len(prompt.split()))
        output_tokens = max(1, len(fake_answer.split()))
        return {
            "text": fake_answer,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "model": model,
        }

class OpenAIProvider(LLMProvider):
    def __init__(self):
        from openai import OpenAI
        self.client = OpenAI(api_key=settings.openai_api_key)

    def generate(self, prompt: str, system_prompt: str, model: str, temperature: float) -> dict:
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
        return {
            "text": resp.choices[0].message.content,
            "input_tokens": resp.usage.prompt_tokens,
            "output_tokens": resp.usage.completion_tokens,
            "model": model,
        }

def get_provider() -> LLMProvider:
    if settings.use_mock_llm:
        return MockProvider()
    return OpenAIProvider()