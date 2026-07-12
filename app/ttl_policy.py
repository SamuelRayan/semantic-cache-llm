from app.config import settings

TIME_SENSITIVE_KEYWORDS = ["today", "now", "current", "latest", "this week", "currently", "right now"]

def assign_ttl(prompt: str) -> int:
    lowered = prompt.lower()
    if any(kw in lowered for kw in TIME_SENSITIVE_KEYWORDS):
        return settings.short_ttl_seconds
    return settings.default_ttl_seconds