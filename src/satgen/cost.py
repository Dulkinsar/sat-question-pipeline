"""Token counts and cost per stage. Prices are USD per million tokens."""

from __future__ import annotations

import threading
from collections import defaultdict
from dataclasses import dataclass, field

PRICES: dict[str, tuple[float, float]] = {
    # model id: (input $/M, output $/M)
    "gemini-2.5-flash-lite": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-haiku-5-5": (0.10, 0.50),  # prompts under 100k tokens; see platform.claude.com/docs pricing
    "claude-haiku-4-5-20251001": (1.00, 5.00),
    # The original 2025 pipeline reviewed with Claude 3.7 Sonnet (retired February 2026).
    "claude-3-7-sonnet-latest": (3.00, 15.00),
}


@dataclass
class CostTracker:
    prices: dict[str, tuple[float, float]] = field(default_factory=lambda: dict(PRICES))
    tokens_in: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    tokens_out: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    calls: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_stage: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def log(self, model: str, tokens_in: int, tokens_out: int, stage: str = "") -> None:
        with self._lock:
            self.tokens_in[model] += tokens_in
            self.tokens_out[model] += tokens_out
            self.calls[model] += 1
            if stage:
                self.by_stage[stage][model + ":in"] += tokens_in
                self.by_stage[stage][model + ":out"] += tokens_out

    def _price(self, model: str, t_in: int, t_out: int) -> float:
        p_in, p_out = self.prices.get(model, (0.0, 0.0))
        return (t_in * p_in + t_out * p_out) / 1_000_000

    def cost(self, model: str) -> float:
        return self._price(model, self.tokens_in[model], self.tokens_out[model])

    def stage_cost(self, stage: str) -> float:
        totals = self.by_stage.get(stage, {})
        models = {k.rsplit(":", 1)[0] for k in totals}
        return sum(self._price(m, totals.get(m + ":in", 0), totals.get(m + ":out", 0)) for m in models)

    def total(self) -> float:
        return sum(self.cost(m) for m in self.calls)

    def unpriced(self) -> list[str]:
        return sorted(m for m in self.calls if m not in self.prices)

    def summary(self, accepted: int) -> dict:
        per_model = {
            m: {
                "calls": self.calls[m],
                "tokens_in": self.tokens_in[m],
                "tokens_out": self.tokens_out[m],
                "usd": round(self.cost(m), 4),
            }
            for m in sorted(self.calls)
        }
        total = self.total()
        return {
            "per_model": per_model,
            "per_stage_usd": {s: round(self.stage_cost(s), 4) for s in sorted(self.by_stage)},
            "total_usd": round(total, 4),
            "accepted": accepted,
            "usd_per_accepted": round(total / accepted, 5) if accepted else None,
            "unpriced_models": self.unpriced(),
        }
