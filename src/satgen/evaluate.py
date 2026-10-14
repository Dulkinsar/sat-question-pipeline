"""Scores the gate and the reviewer on the labelled set. Each item is solved N times and
reviewed once, so every threshold and configuration comes from the same calls.
"""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .answers import normalize_spr
from .llm import LLM
from .models import SPR, Question
from .pipeline import Config, Pipeline, _top_vote


@dataclass
class ItemResult:
    id: str
    label: str
    true_answer: Optional[str]
    key: str
    votes: list[str]
    verdict: Optional[dict]
    recheck: list[str] = field(default_factory=list)
    cost: dict[str, float] = field(default_factory=dict)


def load_results(path: Path) -> list[ItemResult]:
    """Read back a saved <models>.items.jsonl so it can be scored again without new API calls."""
    return [ItemResult(**json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_items(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_items(items: list[dict], solver: LLM, reviewer: LLM, config: Config) -> list[ItemResult]:
    def one(item: dict) -> ItemResult:
        p = Pipeline(generator=solver, solver=solver, reviewer=reviewer, config=config)
        q = Question.from_dict(item["question"])
        if q.format == SPR:
            q.answer = normalize_spr(q.answer)
        votes = p.solve(solver, q, config.runs, "solve")
        agreement = sum(v == q.answer for v in votes)
        top, n = _top_vote(votes)
        salvage = agreement < config.threshold and top is not None and n >= config.threshold
        verdict, recheck = None, []
        try:
            v = p.review(q, votes, salvage)
            verdict = v.__dict__
            if v.accepted and v.corrected_answer not in (None, q.answer) and v.corrected_explanation:
                candidate = Question(**{**q.__dict__, "answer": v.corrected_answer})
                recheck = p.solve(reviewer, candidate, config.recheck_runs, "recheck")
        except ValueError:
            pass  # an unparseable review counts as a rejection
        cost = {s: p.tracker.stage_cost(s) for s in ("solve", "review", "recheck")}
        return ItemResult(item["id"], item["label"], item.get("true_answer"), q.answer, votes, verdict, recheck, cost)

    with ThreadPoolExecutor(max_workers=max(1, config.concurrency)) as pool:
        return list(pool.map(one, items))


def _review_decision(r: ItemResult, config: Config, salvaged: bool) -> tuple[bool, Optional[str]]:
    """(ship?, final key) from the review stage alone."""
    v = r.verdict
    if v is None or not v["accepted"]:
        return False, None
    fixed = v["corrected_answer"]
    if fixed is None or fixed == r.key:
        return (not salvaged), r.key
    if not v["corrected_explanation"]:
        return False, None
    confirmed = sum(x == fixed for x in r.recheck) >= config.recheck_needed
    return confirmed, fixed if confirmed else None


def decide(r: ItemResult, mode: str, k: int, config: Config) -> tuple[bool, Optional[str], float]:
    """Return (shipped, final key, cost spent on this item) under one configuration."""
    agree = sum(v == r.key for v in r.votes)
    c = r.cost
    if mode == "gate_only":
        return agree >= k, r.key, c["solve"]
    if mode == "review_only":
        ship, key = _review_decision(r, config, salvaged=False)
        # the reviewer is shown the solver votes, so the solves are part of its cost
        return ship, key, c["solve"] + c["review"] + c["recheck"]
    if agree >= k:
        ship, key = _review_decision(r, config, salvaged=False)
        return ship, key, c["solve"] + c["review"] + c["recheck"]
    if mode == "full":
        top, n = _top_vote(r.votes)
        if top is not None and n >= k:
            ship, key = _review_decision(r, config, salvaged=True)
            return ship, key, c["solve"] + c["review"] + c["recheck"]
    return False, None, c["solve"]


def score(results: list[ItemResult], mode: str, k: int, config: Config) -> dict:
    by_label = Counter(r.label for r in results)
    shipped = wrong = good_ok = bad_caught = amb_caught = 0
    cost = 0.0
    for r in results:
        ship, key, spent = decide(r, mode, k, config)
        cost += spent
        correct = ship and r.label != "ambiguous" and key == r.true_answer
        shipped += ship
        wrong += ship and not correct
        good_ok += r.label == "good" and correct
        bad_caught += r.label == "bad_key" and (not ship or correct)
        amb_caught += r.label == "ambiguous" and not ship
    ok = shipped - wrong
    return {
        "mode": mode,
        "k": k if mode != "review_only" else None,
        "shipped": shipped,
        "error_rate_shipped": round(wrong / shipped, 3) if shipped else None,
        "good_yield": round(good_ok / by_label["good"], 3) if by_label["good"] else None,
        "bad_key_caught": round(bad_caught / by_label["bad_key"], 3) if by_label["bad_key"] else None,
        "ambiguous_caught": round(amb_caught / by_label["ambiguous"], 3) if by_label["ambiguous"] else None,
        "usd": round(cost, 4),
        "usd_per_correct_shipped": round(cost / ok, 5) if ok else None,
    }


def report(results: list[ItemResult], config: Config) -> dict:
    rows = [score(results, "review_only", 0, config)]
    for mode in ("gate_only", "gate+review", "full"):
        rows += [score(results, mode, k, config) for k in range(1, config.runs + 1)]
    proposed = [r for r in results if r.verdict and r.verdict["corrected_answer"] not in (None, r.key)]
    return {
        "items": len(results),
        "labels": dict(Counter(r.label for r in results)),
        "solver_accuracy_on_good": _solver_accuracy(results),
        "reviewer": {
            "accepted_good": _share(results, "good", lambda r: bool(r.verdict and r.verdict["accepted"])),
            "rejected_ambiguous": _share(results, "ambiguous", lambda r: not (r.verdict and r.verdict["accepted"])),
            "corrections_proposed": len(proposed),
            "corrections_right": sum(r.verdict["corrected_answer"] == r.true_answer for r in proposed),
        },
        "configurations": rows,
    }


def _share(results, label, pred) -> Optional[float]:
    pool = [r for r in results if r.label == label]
    return round(sum(map(pred, pool)) / len(pool), 3) if pool else None


def _solver_accuracy(results) -> Optional[float]:
    votes = [(v, r.true_answer) for r in results if r.label == "good" for v in r.votes]
    return round(sum(v == t for v, t in votes) / len(votes), 3) if votes else None


def to_markdown(rep: dict) -> str:
    lines = [
        f"Items: {rep['items']} {rep['labels']}  ",
        f"Solver accuracy on good items: {rep['solver_accuracy_on_good']}  ",
        f"Reviewer: {rep['reviewer']}",
        "",
        "| " + " | ".join(head for head, _ in _COLUMNS) + " |",
        "|" + "---|" * len(_COLUMNS),
    ]
    for r in rep["configurations"]:
        cells = ["–" if r[key] is None else str(r[key]) for _, key in _COLUMNS]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


_COLUMNS = [
    ("configuration", "mode"),
    ("k", "k"),
    ("shipped", "shipped"),
    ("error rate (shipped)", "error_rate_shipped"),
    ("good yield", "good_yield"),
    ("bad keys caught", "bad_key_caught"),
    ("ambiguous caught", "ambiguous_caught"),
    ("USD", "usd"),
    ("USD per correct", "usd_per_correct_shipped"),
]


class OracleSimClient:
    """Smoke-test stand-in for `satgen eval --simulate`: it reads the labels, so its scores are meaningless."""

    def __init__(self, items: list[dict], model: str = "oracle-sim", accuracy: float = 0.85):
        import random
        import threading

        self.model = model
        self._acc = accuracy
        self._rng = random.Random(3)
        self._lock = threading.Lock()
        self._truth = {}
        for it in items:
            q = Question.from_dict(it["question"])
            self._truth[q.to_prompt()] = it

    def _find(self, prompt: str) -> dict:
        return next(it for p, it in self._truth.items() if p in prompt)

    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        it = self._find(prompt)
        with self._lock:
            roll, pick = self._rng.random(), self._rng.random()
        truth = it["true_answer"] or it["question"]["answer"]
        if "Return JSON with keys:\n  accepted" in prompt:
            key = prompt.split("Answer key: ")[1].split("\n")[0].strip()
            if it["label"] == "ambiguous" and roll < 0.7:
                return (
                    json.dumps({"accepted": False, "corrected_answer": None, "notes": "two defensible answers"}),
                    500,
                    60,
                )
            if key != truth:
                return (
                    json.dumps(
                        {"accepted": True, "corrected_answer": truth, "corrected_explanation": "fixed", "notes": "key"}
                    ),
                    500,
                    60,
                )
            return json.dumps({"accepted": True, "corrected_answer": None, "notes": "ok"}), 500, 60
        if roll < self._acc:
            return truth, 300, 2
        others = [k for k in "ABCD" if k != truth] if it["question"]["format"] == "mcq" else ["0"]
        return others[int(pick * len(others))], 300, 2
