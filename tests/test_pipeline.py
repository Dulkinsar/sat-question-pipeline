import json
from collections import Counter

import pytest

from satgen import pipeline
from satgen.llm import SimulatedClient, call_with_retry
from satgen.models import Question, Record
from satgen.pipeline import Config, Pipeline


class Scripted:
    """Returns canned replies in order; an Exception instance in the script is raised instead."""

    def __init__(self, replies, model="scripted"):
        self.model = model
        self.replies = list(replies)
        self.prompts = []

    def complete(self, prompt, system="", temperature=0.7, json_mode=False):
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply, 100, 10


def gen_json(answer="A", choices=None, **extra):
    choices = choices or {"A": "4", "B": "3", "C": "5", "D": "6"}
    return json.dumps({"qtype": "t", "stem": "What is 2 + 2?", "choices": choices, "answer": answer, **extra})


def make(gen, solver=None, reviewer=None, **cfg):
    cfg.setdefault("concurrency", 1)
    cfg.setdefault("base_delay", 0)
    return Pipeline(gen, solver or gen, reviewer or Scripted([]), config=Config(**cfg))


def unshuffled(p: Pipeline):
    p.shuffle = lambda q, i: None  # keep choice order fixed so scripted votes line up
    return p


def test_gate_blocks_low_agreement_without_review():
    gen = Scripted([gen_json("A"), "A", "B", "C", "B", "D"])  # 1 of 5 agree; no other answer reaches 3
    reviewer = Scripted([])
    rec = unshuffled(make(gen, reviewer=reviewer)).run(["t"], 1).records[0]
    assert rec.status == "screened_out"
    assert reviewer.prompts == []


def review(solved, key_correct=True, unambiguous=True, explanation=None):
    return json.dumps(
        {
            "solved_answer": solved,
            "key_correct": key_correct,
            "unambiguous": unambiguous,
            "explanation": explanation,
            "notes": "",
        }
    )


def test_gate_pass_then_review_accepts():
    gen = Scripted([gen_json("A"), "A", "A", "A", "B", "A"])
    reviewer = Scripted([review("A")])
    rec = unshuffled(make(gen, reviewer=reviewer)).run(["t"], 1).records[0]
    assert (rec.status, rec.agreement, rec.question.answer) == ("accepted", 4, "A")


def test_correction_applied_only_after_blind_recheck_confirms():
    # Solvers agree on C, the key says A: the question is salvaged for review.
    gen = Scripted([gen_json("A"), "C", "C", "C", "C", "A"])
    reviewer = Scripted(
        [
            review("C", key_correct=False, explanation="C is right"),
            "C",
            "C",
            "B",  # blind re-solve by the review model: 2 of 3 confirm
        ]
    )
    rec = unshuffled(make(gen, reviewer=reviewer)).run(["t"], 1).records[0]
    assert rec.status == "accepted"
    assert rec.question.answer == "C" and rec.original_answer == "A"
    assert rec.question.explanation == "C is right"
    assert "Answer key" not in reviewer.prompts[1]  # the re-check never sees a key


def test_correction_that_fails_recheck_goes_to_a_human():
    gen = Scripted([gen_json("A"), "A", "A", "A", "A", "A"])
    reviewer = Scripted(
        [
            review("D", key_correct=False, explanation="D"),
            "A",
            "A",
            "D",
        ]
    )
    rec = unshuffled(make(gen, reviewer=reviewer)).run(["t"], 1).records[0]
    assert rec.status == "needs_human"
    assert rec.question.answer == "A"  # never shipped with the unconfirmed key


def test_wrong_key_with_clear_wording_reaches_the_recheck():
    """The split verdict: a wrong key is a correction, never a rejection on its own."""
    gen = Scripted([gen_json("A"), "B", "B", "B", "A", "B"])
    reviewer = Scripted([review("B", key_correct=False, unambiguous=True, explanation="B fits"), "B", "B", "B"])
    rec = unshuffled(make(gen, reviewer=reviewer)).run(["t"], 1).records[0]
    assert rec.verdict.accepted and rec.verdict.corrected_answer == "B"
    assert (rec.status, rec.question.answer, len(rec.recheck_votes)) == ("accepted", "B", 3)


def test_key_flag_is_overruled_by_the_reviewers_own_answer():
    gen = Scripted([gen_json("A"), "A", "A", "A", "A", "A"])
    reviewer = Scripted([review("C", key_correct=True, explanation=None)])
    rec = unshuffled(make(gen, reviewer=reviewer)).run(["t"], 1).records[0]
    assert rec.verdict.key_correct is False and rec.verdict.corrected_answer == "C"
    assert rec.status == "needs_human"  # a correction without an explanation is not shipped


def test_ambiguous_wording_is_rejected_whatever_the_key():
    gen = Scripted([gen_json("A"), "A", "A", "A", "A", "A"])
    reviewer = Scripted([review("A", unambiguous=False)])
    assert unshuffled(make(gen, reviewer=reviewer)).run(["t"], 1).records[0].status == "rejected"


def test_retry_recovers_from_transient_failures():
    client = Scripted([TimeoutError("slow"), ConnectionError("reset"), "B"])
    delays = []
    text, _, _ = call_with_retry(client, "q", max_retries=3, base_delay=1, sleep=delays.append)
    assert text == "B" and len(delays) == 2


def test_resume_skips_finished_questions(tmp_path):
    ck = tmp_path / "ck.jsonl"
    sim = SimulatedClient()
    first = Pipeline(sim, sim, SimulatedClient("rev"), config=Config(concurrency=4)).run(
        ["words_in_context"], 10, checkpoint=ck
    )
    calls_before = sum(sim._seen.values())
    second = Pipeline(sim, sim, SimulatedClient("rev"), config=Config(concurrency=4)).run(
        ["words_in_context"], 10, checkpoint=ck
    )
    assert sum(sim._seen.values()) == calls_before  # nothing was regenerated
    assert [r.status for r in first.records] == [r.status for r in second.records]
    assert second.processed == set() and second.tracker.total() == 0


def test_shuffle_moves_the_key_with_its_text():
    p = make(Scripted([]))
    positions = Counter()
    for i in range(200):
        q = Question("t", "s", {"A": "right", "B": "w1", "C": "w2", "D": "w3"}, "A")
        p.shuffle(q, i)
        assert q.choices[q.answer] == "right"
        positions[q.answer] += 1
    assert all(30 < positions[k] < 70 for k in "ABCD")


def test_spr_questions_compare_votes_numerically():
    gen = Scripted(
        [
            json.dumps({"qtype": "spr_x", "stem": "If 2x = 3, what is x?", "choices": {}, "answer": "1.5"}),
            "3/2",
            "1.5",
            "x = 3/2",
            "6/4",
            "2",
        ]
    )
    reviewer = Scripted([review("1.5")])
    rec = make(gen, reviewer=reviewer).run(["spr_x"], 1).records[0]
    assert rec.question.format == "spr" and rec.question.answer == "3/2"
    assert rec.agreement == 4 and rec.status == "accepted"


def test_dedupe_against_run_and_bank():
    def acc(i, stem):
        return Record(
            index=i,
            qtype="t",
            status="accepted",
            question=Question(
                "t",
                stem,
                {"A": "1", "B": "2", "C": "3", "D": "4"},
                "A",
                passage="A farmer measured rainfall on twelve plots across two dry seasons in a row.",
            ),
        )

    a, b = acc(0, "Which choice best states the main idea?"), acc(1, "Which choice best states the main idea?")
    c = acc(2, "Which choice best describes the main idea of the text?")  # reworded: Jaccard 0.64
    bank = [
        Question(
            "t",
            "What does the passage suggest about the river?",
            {},
            "A",
            passage="The river flooded the valley each spring, carrying silt that enriched the fields.",
        )
    ]
    d = acc(3, "What does the passage suggest about the river?")
    d.question.passage = bank[0].passage
    flagged = make(Scripted([])).dedupe([a, b, c, d], bank)
    assert (a.status, b.status, d.status) == ("accepted", "duplicate", "duplicate")
    assert d.duplicate_of == "bank:0"
    assert any(rec is c for rec, _, _ in flagged)


def test_offline_end_to_end_ships_only_correct_keys(tmp_path):
    import re
    from fractions import Fraction

    from satgen.llm import _WORD_INDEX

    sim = SimulatedClient("gen")
    p = Pipeline(sim, sim, SimulatedClient("rev"), config=Config(concurrency=8))
    result = p.run(["words_in_context", "spr_linear_equations"], 60)
    out = tmp_path / "q.json"
    n = pipeline.export(result.records, out)
    rows = json.loads(out.read_text())
    assert n == len(rows) > 10
    for q in rows:  # the simulator plants wrong keys; none may reach the export
        if q["format"] == "mcq":
            word = re.search(r'"(\w+)"', q["stem"]).group(1)
            assert q["choices"][q["answer"]] == _WORD_INDEX[word][2]
        else:
            a, b, c = map(int, re.search(r"(\d+)x \+ (\d+) = (\d+)", q["stem"]).groups())
            assert Fraction(q["answer"]) == Fraction(c - b, a)
    stats = pipeline.stats(result.records)
    assert stats["attempted"] == 60 and stats["status_counts"]["duplicate"] > 0


def test_checkpoint_from_a_different_run_is_refused(tmp_path):
    ck = tmp_path / "ck.jsonl"
    sim = SimulatedClient()
    Pipeline(sim, sim, SimulatedClient("rev")).run(["words_in_context"], 3, checkpoint=ck)
    with pytest.raises(pipeline.CheckpointMismatch):
        Pipeline(sim, sim, SimulatedClient("rev")).run(["spr_linear_equations"], 3, checkpoint=ck)
    with pytest.raises(pipeline.CheckpointMismatch):  # same types, different style context
        Pipeline(sim, sim, SimulatedClient("rev"), context="new style").run(["words_in_context"], 3, checkpoint=ck)
