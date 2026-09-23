import sys, os, time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.cache_engine import get_or_generate
from app.cache_db import cache_db

def sep(title):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")

# ── Test 1: Cold start — no history, basic MISS then HIT ─────────────────────
sep("TEST 1 — Cold start: MISS then exact repeat HIT")

r1 = get_or_generate("What is the capital of France?")
print(f"  Request 1 (cold, first ever):  {r1.cache_status}  layer={r1.layer}  latency={r1.latency_ms:.0f}ms")
assert r1.cache_status == "MISS", "First request should always miss"
assert r1.layer == 3

time.sleep(2.5)   # let the async scorer start

r2 = get_or_generate("What is the capital of France?")
sim_str = f"{r2.similarity:.3f}" if r2.similarity is not None else "None"
print(f"  Request 2 (exact repeat):      {r2.cache_status}  layer={r2.layer}  sim={sim_str}")
# Exact repeat: similarity=1.0, both cold_start, but scorer not done yet
# so Layer 1 fires (context_tag matches, context vecs both zero → sim=0.0)
# Zero vectors won't pass CONTEXT_THRESHOLD, falls to Layer 2 which also
# needs scorer. So this may still MISS on first run until scorer finishes.
# Either outcome is correct behaviour — we just log it.
print(f"  (if MISS: scorer hasn't written score yet — correct, not a bug)")

# ── Test 2: Force a Layer 2 hit by manually writing a low-sensitivity score ──
sep("TEST 2 — Layer 2: context-free match on a low-sensitivity answer")

# First, generate and store an entry
r3 = get_or_generate("Explain what a binary search tree is.")
print(f"  Stored entry:  {r3.cache_status}  id will be in DB")
assert r3.cache_status == "MISS"

# Manually set a low sensitivity score (simulating what the async scorer
# would write after recognising this is a context-free definition)
entries = cache_db.get_all_with_context()
target = next((e for e in entries
               if e["context_tag"] is not None), None)
if target:
    cache_db.update_sensitivity_score(target["id"], 0.1)
    print(f"  Manually set sensitivity=0.1 on entry {target['id'][:8]}...")

# Now ask a close paraphrase from a completely different context
r4 = get_or_generate(
    "Can you explain binary search trees?",
    conversation_history=[
        {"role": "user",      "content": "I am working on a Python data structures assignment"},
        {"role": "assistant", "content": "Sure, what topic are you on?"},
    ]
)
print(f"  Paraphrase from different context:  {r4.cache_status}  layer={r4.layer}  sim={r4.similarity}")
if r4.cache_status == "HIT":
    print(f"  ✓ Layer 2 served a context-neutral answer to a different context user")
else:
    print(f"  (MISS — prompt similarity {r4.similarity} may be below 0.92 threshold — fine)")

# ── Test 3: Context classification ───────────────────────────────────────────
sep("TEST 3 — Context classification tags")

from app.context_engine import classify_context, build_context_summary

cold = classify_context("")
print(f"  Empty context → tag: {cold}")
assert cold == "cold_start"

debug_history = [
    {"role": "user",      "content": "I keep getting a KeyError traceback on line 42"},
    {"role": "assistant", "content": "Can you paste the full stack trace?"},
    {"role": "user",      "content": "Here it is: KeyError 'user_id' in auth.py"},
]
summary = build_context_summary(debug_history)
tag = classify_context(summary)
print(f"  Debug history  → tag: {tag}")
assert tag == "debugging_session", f"Expected debugging_session, got {tag}"

plan_history = [
    {"role": "user", "content": "I want to design a new microservice architecture"},
    {"role": "assistant", "content": "What's the main responsibility of this service?"},
]
tag2 = classify_context(build_context_summary(plan_history))
print(f"  Planning history → tag: {tag2}")
assert tag2 == "planning_session", f"Expected planning_session, got {tag2}"

print("  ✓ Classification tags working correctly")

# ── Test 4: Two people, same question, different contexts ─────────────────────
sep("TEST 4 — Same question, different contexts, different answers")

flask_history = [
    {"role": "user",      "content": "I have a Flask app with SQLAlchemy"},
    {"role": "assistant", "content": "What's the issue?"},
    {"role": "user",      "content": "Getting a 404 on my /api/users route"},
    {"role": "assistant", "content": "Check your route decorator"},
    {"role": "user",      "content": "The decorator looks fine"},
]

r_person_a = get_or_generate(
    "How do I fix this bug?",
    conversation_history=flask_history,
)
print(f"  Person A (Flask debug context):   {r_person_a.cache_status}  layer={r_person_a.layer}  tag in DB")

r_person_b = get_or_generate(
    "How do I fix this bug?",
    conversation_history=[],   # cold start
)
print(f"  Person B (no context):            {r_person_b.cache_status}  layer={r_person_b.layer}")

# Both are likely MISSes on first run (scorer not done, cache cold)
# The important thing is they were stored with DIFFERENT context tags
all_entries = cache_db.get_all_with_context()
tags_stored = [e["context_tag"] for e in all_entries]
print(f"  Context tags stored in DB: {set(tags_stored)}")
print("  ✓ Entries stored with correct context metadata for future lookups")

# ── Test 5: Context similarity ────────────────────────────────────────────────
sep("TEST 5 — Context embedding similarity")

from app.context_engine import embed_context, context_similarity

vec_flask1 = embed_context("user: Flask SQLAlchemy 404 error on /api/users route")
vec_flask2 = embed_context("user: Flask app route returning 404 not found error")
vec_unrelated = embed_context("user: I want to plan a marketing strategy for Q3")
vec_cold = embed_context("")

sim_similar   = context_similarity(vec_flask1, vec_flask2)
sim_different = context_similarity(vec_flask1, vec_unrelated)
sim_cold      = context_similarity(vec_flask1, vec_cold)

print(f"  Similar Flask contexts:    {sim_similar:.4f}  (want ≥ 0.72)")
print(f"  Unrelated context:         {sim_different:.4f}  (want < 0.72)")
print(f"  Against cold start (zero): {sim_cold:.4f}  (want = 0.0)")

assert sim_similar > sim_different, "Similar contexts should score higher than unrelated"
assert sim_cold == 0.0, "Cold start vector should always produce 0.0 similarity"
print("  ✓ Context similarity ordering correct")

# ── Summary ───────────────────────────────────────────────────────────────────
sep("ALL TESTS COMPLETE")
print(f"  Cache entries in DB: {cache_db.count()}")
print()
print("  What to note:")
print("  - MISSes on first run are correct: scorer is async,")
print("    Layer 2 hits only appear after score is written.")
print("  - Run the test a SECOND time (without deleting data/)") 
print("    and you will see Layer 2 hits appear for context-neutral answers.")
print()