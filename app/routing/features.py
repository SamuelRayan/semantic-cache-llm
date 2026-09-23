"""Handcrafted features for the baseline router — "the proposal as
written": cheap, interpretable signals a human would reach for first,
before reaching for an embedding at all."""

import re
import numpy as np

CODE_KEYWORDS = [
    "def ", "class ", "function", "import ", "```", "{", "}", ";",
    "return ", "for (", "while (", "console.log", "print(",
]

REASONING_KEYWORDS = [
    "prove", "analyze", "analyse", "compare", "step by step", "design",
    "debug", "explain why", "derive", "plan", "trade-off", "tradeoff",
]

CONSTRAINT_PATTERNS = [
    r"\bmust\b", r"\bshould\b", r"\bat least\b", r"\bno more than\b",
    r"\bonly\b", r"\bexactly\b", r"\brequirement", r"\bcannot\b", r"\bwithin\b",
]

FEATURE_NAMES = [
    "char_len", "token_len", "code_presence", "math_density",
    "question_count", "reasoning_keyword_count", "constraint_count",
    "context_len",
]

N_FEATURES = len(FEATURE_NAMES)


def extract_features(prompt: str, context_text: str = "") -> np.ndarray:
    text = prompt or ""
    lowered = text.lower()

    char_len = len(text)
    token_len = len(text.split())

    code_presence = 1.0 if ("```" in text or any(kw in lowered for kw in CODE_KEYWORDS)) else 0.0

    digits = sum(c.isdigit() for c in text)
    math_symbols = sum(c in "+-*/=<>^%" for c in text)
    math_density = (digits + math_symbols) / max(1, char_len)

    question_count = float(text.count("?"))

    reasoning_keyword_count = float(sum(1 for kw in REASONING_KEYWORDS if kw in lowered))

    constraint_count = float(sum(len(re.findall(p, lowered)) for p in CONSTRAINT_PATTERNS))

    context_len = float(len(context_text or ""))

    return np.array([
        char_len, token_len, code_presence, math_density,
        question_count, reasoning_keyword_count, constraint_count,
        context_len,
    ], dtype="float32")


def extract_features_batch(prompts, context_texts=None) -> np.ndarray:
    if context_texts is None:
        context_texts = [""] * len(prompts)
    return np.stack([extract_features(p, c) for p, c in zip(prompts, context_texts)])
