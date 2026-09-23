"""
Template-slot caching for rapidly-changing values.

For a prompt like "What is AAPL's current stock price?" the STRUCTURE of
the answer never changes — only a handful of numbers inside it do. Instead
of treating the whole response as either fully cacheable (stale) or fully
non-cacheable (no savings), we extract the volatile pieces (prices,
percentages, dates, counts) into named `{{slot_N}}` placeholders, so the
template itself can be reused indefinitely and only the slot VALUES need
refreshing.
"""

import re
from dataclasses import dataclass, field


# ── Volatile-value detection ────────────────────────────────────────────────
# Order matters: more specific patterns (price, percentage, date) are tried
# before the generic number fallback, so e.g. "2026-09-23" is claimed whole
# by the date branch instead of being split into three separate "number"
# matches for 2026, 09, and 23.
_MONTH_NAMES = (
    "January|February|March|April|May|June|July|August|"
    "September|October|November|December"
)

TOKEN_RE = re.compile(
    r"(?P<price>\$(?P<price_num>\d[\d,]*(?:\.\d+)?))"
    r"|(?P<percentage>(?P<pct_num>\d+(?:\.\d+)?)%)"
    r"|(?P<date>(?:" + _MONTH_NAMES + r")\s+\d{1,2},?\s+\d{4}"
    r"|\d{4}-\d{2}-\d{2}"
    r"|\d{1,2}/\d{1,2}/\d{2,4})"
    r"|(?P<number>\d[\d,]*(?:\.\d+)?)"
)

# A bare integer this short (no decimal point) reads as a "count" (items,
# shares, users...); anything longer/decimal is a generic "number". This is
# a heuristic, not a semantic classifier — good enough for cache-slot typing.
_COUNT_MAX_DIGITS = 4


@dataclass
class TemplateEntry:
    template_text: str
    slot_values: dict
    slot_types: dict
    is_templatable: bool
    volatility_score: float


@dataclass
class DriftResult:
    tier: str                # "T0", "T1", or "T2"
    changed_slots: dict = field(default_factory=dict)  # slot -> (old, new), only for T1/T2
    reason: str = ""


def extract_template(prompt: str, response: str) -> TemplateEntry:
    """Scan `response` for volatile values and replace each with a named
    placeholder. `prompt` isn't currently used for extraction itself (the
    values are identified purely from the response text) but is kept in
    the signature since a smarter, LLM-based extractor would want it —
    e.g. to know that "AAPL" in the question makes a bare "$210.50" a
    stock price rather than, say, a product cost.
    """
    slot_values = {}
    slot_types = {}
    out_parts = []
    last_end = 0
    slot_index = 0
    volatile_chars = 0

    for m in TOKEN_RE.finditer(response):
        start, end = m.span()
        out_parts.append(response[last_end:start])

        if m.group("price") is not None:
            value = m.group("price_num").replace(",", "")
            placeholder = "$" + "{{" + f"slot_{slot_index}" + "}}"
            kind = "price"
        elif m.group("percentage") is not None:
            value = m.group("pct_num")
            placeholder = "{{" + f"slot_{slot_index}" + "}}" + "%"
            kind = "percentage"
        elif m.group("date") is not None:
            value = m.group("date")
            placeholder = "{{" + f"slot_{slot_index}" + "}}"
            kind = "date"
        else:
            raw = m.group("number")
            value = raw.replace(",", "")
            placeholder = "{{" + f"slot_{slot_index}" + "}}"
            kind = "count" if ("." not in raw and len(value) <= _COUNT_MAX_DIGITS) else "number"

        slot_name = f"slot_{slot_index}"
        slot_values[slot_name] = value
        slot_types[slot_name] = kind
        volatile_chars += (end - start)
        out_parts.append(placeholder)

        last_end = end
        slot_index += 1

    out_parts.append(response[last_end:])
    template_text = "".join(out_parts)

    is_templatable = slot_index > 0
    volatility_score = (volatile_chars / len(response)) if response else 0.0

    return TemplateEntry(
        template_text=template_text,
        slot_values=slot_values,
        slot_types=slot_types,
        is_templatable=is_templatable,
        volatility_score=volatility_score,
    )


def fill_template(template_text: str, new_slot_values: dict) -> str:
    """Substitute slot values back into a template. Plain string replace is
    safe here — `{{slot_1}}` and `{{slot_10}}` are distinct literal
    substrings (the closing `}}` immediately follows the exact slot name),
    so there's no risk of slot_1 accidentally matching inside slot_10."""
    result = template_text
    for slot_name, value in new_slot_values.items():
        result = result.replace("{{" + slot_name + "}}", str(value))
    return result


def _percent_change(old_val: float, new_val: float) -> float:
    if old_val == 0:
        return 0.0 if new_val == 0 else float("inf")
    return abs(new_val - old_val) / abs(old_val)


def check_slot_drift(old_values: dict, new_values: dict, slot_types: dict) -> DriftResult:
    """Classify how much each slot moved, and escalate to the worst tier
    seen across all slots:

      T0 — numbers within 5%, dates unchanged        -> substitute directly
      T1 — moved more than 5% but less than 50%       -> patch just the delta
      T2 — moved 50%+, or the value isn't comparable   -> full regeneration
    """
    tier_rank = {"T0": 0, "T1": 1, "T2": 2}
    worst_tier = "T0"
    changed = {}
    reasons = []

    for slot_name, old_val in old_values.items():
        if slot_name not in new_values:
            continue
        new_val = new_values[slot_name]
        slot_type = slot_types.get(slot_name, "number")

        if slot_type == "date":
            tier = "T0" if str(old_val) == str(new_val) else "T2"
        else:
            try:
                old_f = float(str(old_val).replace(",", ""))
                new_f = float(str(new_val).replace(",", ""))
            except ValueError:
                tier = "T0" if str(old_val) == str(new_val) else "T2"
            else:
                change = _percent_change(old_f, new_f)
                if change <= 0.05:
                    tier = "T0"
                elif change < 0.5:
                    tier = "T1"
                else:
                    tier = "T2"

        if tier != "T0":
            changed[slot_name] = (old_val, new_val)
            reasons.append(f"{slot_name}: {old_val} -> {new_val} ({tier})")

        if tier_rank[tier] > tier_rank[worst_tier]:
            worst_tier = tier

    return DriftResult(
        tier=worst_tier,
        changed_slots=changed,
        reason="; ".join(reasons) if reasons else "all slots within safe band",
    )


def build_patch_prompt(template_text: str, drift: DriftResult, slot_types: dict) -> str:
    """T1 helper: a small prompt asking only about the slots that actually
    moved, instead of regenerating the whole response."""
    lines = [
        f"- {name} ({slot_types.get(name, 'number')}): was {old}, now approximately {new}"
        for name, (old, new) in drift.changed_slots.items()
    ]
    return (
        "Only these values changed since the last answer; confirm or correct "
        "their current precise values as JSON (slot name -> value):\n\n"
        + "\n".join(lines)
        + '\n\nReturn ONLY JSON, e.g. {"slot_0": "215.00"}'
    )
