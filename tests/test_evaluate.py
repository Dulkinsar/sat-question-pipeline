from pathlib import Path

from satgen import evaluate
from satgen.evaluate import ItemResult, decide
from satgen.pipeline import Config

ROOT = Path(__file__).resolve().parents[1]
CFG = Config(concurrency=4)
OK = {"accepted": True, "corrected_answer": None, "corrected_explanation": None, "notes": ""}
COST = {"solve": 0.01, "review": 0.1, "recheck_review": 0.0, "blind": 0.1, "recheck_blind": 0.0}


def item(label, key, truth, votes, verdict=OK, recheck=(), blind=OK, blind_recheck=(), id_="x-good"):
    return ItemResult(
        id_, label, truth, key, list(votes), verdict, list(recheck), dict(COST), blind, list(blind_recheck)
    )


def test_eval_set_pairs_every_bad_key_with_its_good_twin():
    items = evaluate.load_items(ROOT / "data/eval/eval_set.jsonl")
    by_label = {lab: {i["base"]: i for i in items if i["label"] == lab} for lab in ("good", "bad_key", "ambiguous")}
    assert by_label["bad_key"].keys() == by_label["good"].keys()
    assert by_label["ambiguous"].keys() <= by_label["good"].keys()
    for base, bad in by_label["bad_key"].items():
        assert bad["question"]["stem"] == by_label["good"][base]["question"]["stem"]
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


def test_review_only_is_blind_and_costs_only_the_review():
    r = item("good", "A", "A", "BBBBB", verdict=None, blind=OK)
    ship, key, usd = decide(r, "review_only", 0, CFG)
    assert (ship, key, usd) == (True, "A", 0.1)  # the bad votes never reach the blind reviewer


def test_anchoring_counts_pairs_judged_ambiguous_only_under_the_wrong_key():
    rejected = {**OK, "accepted": False}
    good = item("good", "A", "A", "AAAAA", blind=OK, id_="q1-good")
    bad = item("bad_key", "B", "A", "AAAAA", blind=rejected, id_="q1-badkey")
    stats = evaluate._anchoring([good, bad], lambda r: r.blind_verdict)
    assert stats == {"pairs": 1, "called_ambiguous_only_with_wrong_key": 1, "rate": 1.0}
