"""A fake "live" data source that changes between calls, used to simulate
rapidly-changing values (stock prices, crypto prices, exchange rates) for
testing template-slot drift detection without hitting a real API."""

LIVE_DATA = {
    "aapl_price": 210.50,
    "btc_price": 67000.00,
    "usd_inr": 83.50,
}


def get_live_value(key):
    return LIVE_DATA[key]


def simulate_price_change(key, pct):
    LIVE_DATA[key] *= (1 + pct / 100)
