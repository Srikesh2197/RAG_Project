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
    # OpenAI, checked on the pricing page on 4 October 2026.
    "gpt-6-luna": (0.10, 0.50),
    "gpt-6.1-sol": (2.00, 10.00),
}

# Claude counts more tokens than tiktoken's cl100k_base for the same text. Measured on
# the Stage 1 calls: about 4,000 billed for prompts of about 2,750 cl100k tokens.
# Used only for estimates printed before a run; bills use the counts the API returns.
CLAUDE_TOKENS_PER_CL100K = 1.45

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
