import json
import re
import numpy as np
from app.embeddings import embedding_service

# These tags are the "buckets" a conversation can fall into.
# Coarse on purpose — 6 buckets beats 600. The embedding does
# the fine-grained work; the tag is just a fast pre-filter.
CONTEXT_TAGS = [
    "cold_start",        # no prior context at all
    "debugging_session", # fixing errors, bugs, stack traces
    "planning_session",  # designing, architecting, planning
    "data_analysis",     # querying, analysing, charting data
    "writing_session",   # drafting, editing, summarising text
    "general_qa",        # factual questions with no project context
]

# Keywords that vote for each tag. Simple majority wins.
TAG_KEYWORDS = {
    "debugging_session": [
        "error", "bug", "fix", "traceback", "exception", "crash",
        "broken", "not working", "fails", "debug", "stack trace",
        "issue", "problem", "404", "500", "wrong", "doesn't work",
    ],
    "planning_session": [
        "design", "architect", "plan", "structure", "approach",
        "should i", "how would", "best way", "strategy", "outline",
    ],
    "data_analysis": [
        "query", "sql", "dataframe", "chart", "plot", "aggregate",
        "filter", "join", "average", "count", "sum", "analyse",
    ],
    "writing_session": [
        "write", "draft", "edit", "summarise", "summarize", "rewrite",
        "improve", "proofread", "tone", "paragraph", "article",
    ],
    "general_qa": [
        "what is", "explain", "define", "how does", "tell me about",
        "capital", "who is", "when was", "why is",
    ],
}

def _keyword_matches(keyword: str, text: str) -> bool:
    """Word-boundary match, not a bare substring check. A naive `kw in text`
    check lets short keywords like "sql" false-positive-match inside longer
    unrelated words — e.g. "SQLAlchemy" contains "sql" as a prefix, which
    used to make any Flask/SQLAlchemy debugging conversation get misfiled
    as data_analysis instead of debugging_session, since "sql" isn't a
    separate word there at all."""
    pattern = r"\b" + re.escape(keyword) + r"\b"
    return re.search(pattern, text) is not None


def classify_context(context_summary: str) -> str:
    """Keyword-vote classifier. Fast, zero ML cost, good enough for
    the pre-filter role it plays in the three-layer lookup."""
    if not context_summary or context_summary.strip() == "":
        return "cold_start"

    lowered = context_summary.lower()
    scores = {tag: 0 for tag in TAG_KEYWORDS}
    for tag, keywords in TAG_KEYWORDS.items():
        for kw in keywords:
            if _keyword_matches(kw, lowered):
                scores[tag] += 1

    best_tag = max(scores, key=scores.get)
    if scores[best_tag] == 0:
        return "general_qa"
    return best_tag


def build_context_summary(conversation_history: list[dict]) -> str:
    """Rolling compression: take the last N messages and join them into
    a short block of text. In a production version you'd call a cheap
    LLM to summarise; here we do extractive truncation which is free
    and sufficient for embedding + classification.

    conversation_history is a list of dicts:
        [{"role": "user"|"assistant", "content": "..."},  ...]
    """
    if not conversation_history:
        return ""

    # Take the last 5 messages, trim each to 200 chars to keep total short
    recent = conversation_history[-5:]
    parts = []
    for msg in recent:
        role = msg.get("role", "user")
        content = msg.get("content", "")[:200]
        parts.append(f"{role}: {content}")
    return " | ".join(parts)


def embed_context(context_summary: str) -> np.ndarray:
    """Embed the context summary the same way we embed prompts.
    Returns a normalised float32 vector, or a zero vector for
    empty context (cold_start) — cosine similarity against a zero
    vector is always 0.0, which is exactly the right behaviour:
    a cold-start entry never matches a context-rich query."""
    if not context_summary or context_summary.strip() == "":
        return np.zeros(384, dtype="float32")
    return embedding_service.embed(context_summary)


def context_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """Dot product of two normalised vectors = cosine similarity.
    Returns 0.0 if either vector is the zero vector (cold_start)."""
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(vec_a, vec_b))


def context_to_json(vec: np.ndarray) -> str:
    return json.dumps(vec.tolist())


def context_from_json(json_str: str) -> np.ndarray:
    return np.array(json.loads(json_str), dtype="float32")