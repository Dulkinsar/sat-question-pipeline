from pathlib import Path

from satgen import evaluate
from satgen.evaluate import ItemResult, decide
from satgen.pipeline import Config

ROOT = Path(__file__).resolve().parents[1]
CFG = Config(concurrency=4)
OK = {"accepted": True, "corrected_answer": None, "corrected_explanation": None, "notes": ""}
COST = {"solve": 0.01, "review": 0.1, "recheck": 0.0}


def item(label, key, truth, votes, verdict=OK, recheck=()):
    return ItemResult("x", label, truth, key, list(votes), verdict, list(recheck), dict(COST))


def test_eval_set_is_balanced_and_well_formed():
    items = evaluate.load_items(ROOT / "data/eval/eval_set.jsonl")
    labels = {i["label"] for i in items}
    assert labels == {"good", "bad_key", "ambiguous"}
    for it in items:
        q = it["question"]
        if it["label"] == "good":
            assert q["answer"] == it["true_answer"]
        if it["label"] == "bad_key":
            assert q["answer"] != it["true_answer"]


def test_full_mode_salvages_and_fixes_a_bad_key():
    fix = {**OK, "corrected_answer": "A", "corrected_explanation": "A is right"}
    r = item("bad_key", "B", "A", "AAAAB", verdict=fix, recheck="AAA")
    assert decide(r, "gate+review", 3, CFG)[:2] == (False, None)
    assert decide(r, "full", 3, CFG)[:2] == (True, "A")


def test_review_only_cost_includes_the_solver_calls():
    from satgen.evaluate import ItemResult, decide
    from satgen.pipeline import Config

    verdict = {"accepted": True, "corrected_answer": None, "corrected_explanation": None, "notes": ""}
    r = ItemResult("x", "good", "A", "A", ["A"] * 5, verdict, [], {"solve": 0.5, "review": 1.0, "recheck": 0.0})
    assert decide(r, "review_only", 0, Config())[2] == 1.5
