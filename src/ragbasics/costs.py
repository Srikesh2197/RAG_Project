"""Prices and the cost ledger.

Prices are US dollars per million tokens, as listed by each vendor in September 2026.
Models not listed (the offline ones) cost nothing.
"""

import csv
from datetime import UTC, datetime
from pathlib import Path

EMBEDDING_PRICE = {
    "text-embedding-3-small": 0.02,
    "text-embedding-3-large": 0.13,
}
# (input, output)
GENERATION_PRICE = {
    "claude-sonnet-5-5": (2.00, 10.00),
}

LEDGER_FIELDS = ["time", "step", "model", "input_tokens", "output_tokens", "cost_usd"]


def embedding_cost(model: str, tokens: int) -> float:
    return EMBEDDING_PRICE.get(model, 0.0) * tokens / 1e6


def generation_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = GENERATION_PRICE.get(model, (0.0, 0.0))
    return (price_in * input_tokens + price_out * output_tokens) / 1e6


def append_ledger(
    path: Path, step: str, model: str, input_tokens: int, output_tokens: int, cost_usd: float
) -> None:
    """Add one row to the ledger, writing the header first if the file is new."""
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(LEDGER_FIELDS)
        time = datetime.now(UTC).isoformat(timespec="seconds")
        writer.writerow([time, step, model, input_tokens, output_tokens, f"{cost_usd:.6f}"])
