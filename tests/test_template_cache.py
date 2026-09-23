"""
Feature 1 — Template-slot caching tests.

Tests extract_template / fill_template / check_slot_drift directly against
a crafted response string (rather than MockProvider's fixed fake output,
which contains no realistic volatile values) — this tests the cache logic
independently of the provider, per the project's own testing approach for
mock-provider limitations.
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.template_engine import extract_template, fill_template, check_slot_drift
from tests.mock_data_source import LIVE_DATA, get_live_value, simulate_price_change

_results = []


def check(condition, description, detail=""):
    status = "PASS" if condition else "FAIL"
    _results.append((status, description))
    line = f"  [{status}] {description}"
    if detail:
        line += f"   -- {detail}"
    print(line)
    return condition


def sep(title):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print(f"{'-'*60}")


# ── TEST 1 — extract_template identifies volatile slots ────────────────────
sep("TEST 1 - extract_template identifies volatile slots correctly")

prompt = "What is AAPL's current stock price?"
response = (
    f"AAPL is currently trading at ${get_live_value('aapl_price'):.2f}, "
    f"up 2.35% today. Data as of 2026-09-23. Trading volume: 1200 shares."
)
print(f"  prompt:   {prompt}")
print(f"  response: {response}")

entry = extract_template(prompt, response)
print(f"  template_text: {entry.template_text}")
print(f"  slot_values:   {entry.slot_values}")
print(f"  slot_types:    {entry.slot_types}")

check(entry.is_templatable, "is_templatable is True")
check(len(entry.slot_values) == 4, "exactly 4 volatile slots found",
      f"found {len(entry.slot_values)}")
check("price" in entry.slot_types.values(), "a 'price' slot was detected")
check("percentage" in entry.slot_types.values(), "a 'percentage' slot was detected")
check("date" in entry.slot_types.values(), "a 'date' slot was detected")
check(any(t in ("count", "number") for t in entry.slot_types.values()),
      "a 'count'/'number' slot was detected")
check(0.0 < entry.volatility_score <= 1.0, "volatility_score is in (0, 1]",
      f"score={entry.volatility_score:.3f}")

price_slot = next((n for n, t in entry.slot_types.items() if t == "price"), None)
check(
    price_slot is not None
    and entry.slot_values[price_slot] == f"{get_live_value('aapl_price'):.2f}",
    "price slot value matches the live price",
    f"slot={price_slot} value={entry.slot_values.get(price_slot)}"
)

date_slot = next((n for n, t in entry.slot_types.items() if t == "date"), None)
check(date_slot is not None and entry.slot_values[date_slot] == "2026-09-23",
      "date slot value extracted correctly", f"value={entry.slot_values.get(date_slot)}")


# ── TEST 2 — fill_template round-trip ───────────────────────────────────────
sep("TEST 2 - fill_template reconstructs the original response exactly")

filled = fill_template(entry.template_text, entry.slot_values)
check(filled == response, "filled template == original response",
      "" if filled == response else f"got: {filled!r}")


# ── TEST 3 — small drift (2%) -> T0 ─────────────────────────────────────────
sep("TEST 3 - small price move (2%) drifts to tier T0 (no LLM call)")

old_values = dict(entry.slot_values)
simulate_price_change("aapl_price", 2)
new_price = get_live_value("aapl_price")
new_values_t0 = dict(old_values)
new_values_t0[price_slot] = f"{new_price:.2f}"

print(f"  {price_slot}: {old_values[price_slot]} -> {new_values_t0[price_slot]}  (+2%)")

drift_small = check_slot_drift(old_values, new_values_t0, entry.slot_types)
check(drift_small.tier == "T0", "2% price move classified as tier T0",
      f"tier={drift_small.tier}  reason={drift_small.reason}")

refilled_t0 = fill_template(entry.template_text, new_values_t0)
check(f"${new_price:.2f}" in refilled_t0, "T0 refill correctly substitutes the new price",
      refilled_t0)


# ── TEST 4 — large drift (60%) -> T2 ────────────────────────────────────────
sep("TEST 4 - large price move (60%) drifts to tier T2 (full regeneration)")

# Reset the simulated feed back to the ORIGINAL price before applying a
# clean 60% jump from the same baseline used in Test 1.
LIVE_DATA["aapl_price"] = float(old_values[price_slot])
simulate_price_change("aapl_price", 60)
big_price = get_live_value("aapl_price")
new_values_t2 = dict(old_values)
new_values_t2[price_slot] = f"{big_price:.2f}"

print(f"  {price_slot}: {old_values[price_slot]} -> {new_values_t2[price_slot]}  (+60%)")

drift_big = check_slot_drift(old_values, new_values_t2, entry.slot_types)
check(drift_big.tier == "T2", "60% price move classified as tier T2",
      f"tier={drift_big.tier}  reason={drift_big.reason}")


# ── TEST 5 — date drift: same day -> T0, different day -> T2 ───────────────
sep("TEST 5 - date slot drift: unchanged date is T0, changed date is T2")

date_values_old = {date_slot: "2026-09-23"}
date_values_same = {date_slot: "2026-09-23"}
date_values_diff = {date_slot: "2026-09-24"}
date_types = {date_slot: "date"}

drift_same_day = check_slot_drift(date_values_old, date_values_same, date_types)
check(drift_same_day.tier == "T0", "unchanged date classified as T0",
      f"tier={drift_same_day.tier}")

drift_diff_day = check_slot_drift(date_values_old, date_values_diff, date_types)
check(drift_diff_day.tier == "T2", "changed date classified as T2",
      f"tier={drift_diff_day.tier}")


# ── SUMMARY ──────────────────────────────────────────────────────────────────
sep("SUMMARY")
passed = sum(1 for s, _ in _results if s == "PASS")
failed = sum(1 for s, _ in _results if s == "FAIL")
print(f"  {passed} passed, {failed} failed, {len(_results)} total checks")
if failed:
    print("  Failed checks:")
    for s, d in _results:
        if s == "FAIL":
            print(f"    - {d}")
