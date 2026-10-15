"""One question end to end: generate, solve N times, gate, review, blind re-check, dedupe, export."""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

from .answers import CHOICES, is_nullish, normalize_spr, parse_bool, parse_letter
from .cost import CostTracker
from .dedupe import MinHashIndex
from .llm import LLM, call_with_retry
from .models import MCQ, SPR, Question, Record, Verdict

log = logging.getLogger("satgen")

GEN_SYSTEM = (
    "You write original Digital SAT questions. For multiple choice, exactly one choice is "
    "correct and the three distractors are plausible but clearly wrong on close reading."
)
GEN_PROMPT = """Write one Digital SAT question of type: {qtype}
Variation: {index}

Style context:
{context}

{format_rules}

Return JSON with keys: qtype, format, passage, stem, choices, answer, explanation."""
MCQ_RULES = 'Format "mcq": choices is an object with keys A, B, C, D; answer is one letter.'
SPR_RULES = (
    'Format "spr" (student-produced response): choices is an empty object; answer is a single '
    "number written as an integer, decimal or fraction such as 3/4."
)

SOLVE_MCQ = """Answer this SAT question. Reply with JSON only: {{"answer": "<A, B, C or D>"}}

{question}"""
SOLVE_SPR = """Answer this SAT question. Reply with JSON only: {{"answer": "<number: integer, decimal or fraction>"}}

{question}"""

REVIEW_SYSTEM = (
    "Reviewer for a Digital SAT question bank. Check that the question has exactly one correct "
    "answer, that the key is right, and that the wording is unambiguous. Be strict: reject "
    "anything a careful student could reasonably argue about."
)
# The verdict is split into separate judgements. With a single "accepted" flag, reviewers
# reject a wrong-keyed question outright even while naming the right answer, which
# stops salvage from ever reaching the blind re-check.
REVIEW_PROMPT = """Review this question.

{question}

Answer key: {answer}
{votes}
Solve it yourself first, then judge the key and the wording separately.
Return JSON with keys:
  solved_answer (the answer you reach)
  key_correct (true/false: does the key match the one correct answer?)
  unambiguous (true/false: exactly one defensible answer and clear wording, judged as if the key were fixed)
  explanation (a full explanation of the right answer when key_correct is false, else null)
  notes (string)"""
VOTES_LINE = "Solver votes ({runs} independent runs): {votes}\n"
# Key-hidden review: the reviewer never sees the key or the votes, so neither can anchor its
# judgement of the wording; the code compares its answer with the key.
REVIEW_KEYLESS_PROMPT = """Review this question. Solve it yourself, then judge the wording.

{question}

Return JSON with keys:
  solved_answer (the answer you reach)
  unambiguous (true/false: exactly one defensible answer and clear wording)
  explanation (a full explanation of the correct answer)
  notes (string)"""


def parse_json(text: str) -> dict:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object in model output")
    data = json.loads(text[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("model output is not a JSON object")
    return data


@dataclass
class Config:
    runs: int = 5
    threshold: int = 3
    salvage: bool = True
    recheck_runs: int = 3
    recheck_needed: int = 2
    review_mode: str = "keyless"  # "keyless": reviewer sees neither key nor votes; "key": it sees both
    concurrency: int = 8
    seed: int = 0
    max_retries: int = 4
    base_delay: float = 1.0
    duplicate_jaccard: float = 0.8
    flag_jaccard: float = 0.5


@dataclass
class RunResult:
    records: list[Record]
    flagged: list[tuple[Record, str, float]]
    tracker: CostTracker
    processed: set[int] = field(default_factory=set)  # indices worked on in this session (the tracker's spend)


@dataclass
class Pipeline:
    generator: LLM
    solver: LLM
    reviewer: LLM
    context: str = ""
    config: Config = field(default_factory=Config)
    tracker: CostTracker = field(default_factory=CostTracker)

    def _call(self, client: LLM, stage: str, prompt: str, **kw) -> str:
        text, t_in, t_out = call_with_retry(
            client, prompt, max_retries=self.config.max_retries, base_delay=self.config.base_delay, **kw
        )
        self.tracker.log(client.model, t_in, t_out, stage=stage)
        return text

    def generate(self, index: int, qtype: str) -> Question:
        fmt = SPR if qtype.startswith("spr_") else MCQ
        prompt = GEN_PROMPT.format(
            qtype=qtype,
            index=index,
            context=self.context,
            format_rules=SPR_RULES if fmt == SPR else MCQ_RULES,
        )
        text = self._call(self.generator, "generate", prompt, system=GEN_SYSTEM, temperature=0.9, json_mode=True)
        data = parse_json(text)
        q = Question(
            qtype=str(data.get("qtype") or qtype),
            stem=str(data["stem"]),
            choices={str(k).strip().upper(): str(v) for k, v in (data.get("choices") or {}).items()},
            answer=str(data["answer"]).strip(),
            explanation=str(data.get("explanation") or ""),
            passage=str(data.get("passage") or ""),
            format=fmt,
        )
        if fmt == MCQ:
            q.answer = q.answer.upper()
        else:
            q.answer = normalize_spr(q.answer)
        q.validate()
        return q

    def shuffle(self, q: Question, index: int) -> None:
        """Permute MCQ choices with a per-question seed and move the key with its text."""
        if q.format != MCQ:
            return
        rng = random.Random(f"{self.config.seed}:{index}")
        texts = [q.choices[k] for k in CHOICES]
        right = q.choices[q.answer]
        rng.shuffle(texts)
        q.choices = dict(zip(CHOICES, texts, strict=True))
        q.answer = CHOICES[texts.index(right)]

    def solve(self, client: LLM, q: Question, runs: int, stage: str) -> list[str]:
        template = SOLVE_SPR if q.format == SPR else SOLVE_MCQ
        prompt = template.format(question=q.to_prompt())
        votes = []
        for _ in range(runs):
            text = self._call(client, stage, prompt, temperature=0.7, json_mode=True)
            vote = normalize_spr(text) if q.format == SPR else (parse_letter(text) or "?")
            votes.append(vote)
        return votes

    def review(self, q: Question, votes: Optional[list[str]], stage: str = "review", show_key: bool = True) -> Verdict:
        """Ask the reviewer for a verdict. votes=None hides the solver votes; show_key=False hides
        the key as well, and the key check is done here by comparing answers."""
        if show_key:
            prompt = REVIEW_PROMPT.format(
                question=q.to_prompt(),
                answer=q.answer,
                votes=VOTES_LINE.format(runs=len(votes), votes=", ".join(votes)) if votes is not None else "",
            )
        else:
            prompt = REVIEW_KEYLESS_PROMPT.format(question=q.to_prompt())
        data = parse_json(self._call(self.reviewer, stage, prompt, system=REVIEW_SYSTEM, temperature=0.0))
        unambiguous = parse_bool(data.get("unambiguous"))
        solved = _reviewer_answer(q, data.get("solved_answer"))
        if solved is None:
            if unambiguous:
                raise ValueError("solved_answer is not a valid answer")
            # A reviewer that finds two defensible answers often names both; that is a rejection.
            return Verdict(accepted=False, notes=str(data.get("notes") or ""))
        key_correct = parse_bool(data.get("key_correct")) if show_key else solved == q.answer
        if key_correct and solved != q.answer:
            key_correct = False  # trust the worked answer over the flag
        explanation = data.get("explanation")
        return Verdict(
            accepted=unambiguous,
            corrected_answer=None if key_correct else solved,
            corrected_explanation=str(explanation) if explanation and not key_correct else None,
            notes=str(data.get("notes") or ""),
            solved_answer=solved,
            key_correct=key_correct,
        )

    def process(self, index: int, qtype: str) -> Record:
        rec = Record(index=index, qtype=qtype)
        try:
            try:
                q = self.generate(index, qtype)
            except (ValueError, KeyError, TypeError, AttributeError, json.JSONDecodeError) as exc:
                rec.error = f"generation: {exc}"
                return rec  # status stays generation_failed
            self.shuffle(q, index)
            rec.question, rec.original_answer = q, q.answer

            rec.solve_votes = self.solve(self.solver, q, self.config.runs, "solve")
            rec.agreement = sum(v == q.answer for v in rec.solve_votes)
            rec.passed_screen = rec.agreement >= self.config.threshold
            salvage = False
            if not rec.passed_screen:
                top, n = _top_vote(rec.solve_votes)
                salvage = self.config.salvage and top is not None and n >= self.config.threshold
                if not salvage:
                    rec.status = "screened_out"
                    return rec

            try:
                if self.config.review_mode == "keyless":
                    rec.verdict = self.review(q, None, show_key=False)
                else:
                    rec.verdict = self.review(q, rec.solve_votes)
            except (ValueError, json.JSONDecodeError) as exc:
                rec.status, rec.error = "rejected", f"review: {exc}"
                return rec
            v = rec.verdict
            if not v.accepted:
                rec.status = "rejected"
                return rec
            if v.corrected_answer is None or v.corrected_answer == q.answer:
                # A salvaged question whose key the reviewer upheld still failed the gate.
                rec.status = "needs_human" if salvage else "accepted"
                return rec
            if not v.corrected_explanation:
                rec.status, rec.error = "needs_human", "key corrected without a new explanation"
                return rec

            candidate = Question(**{**q.__dict__, "answer": v.corrected_answer})
            rec.recheck_votes = self.solve(self.reviewer, candidate, self.config.recheck_runs, "recheck")
            if sum(x == v.corrected_answer for x in rec.recheck_votes) >= self.config.recheck_needed:
                q.answer, q.explanation = v.corrected_answer, v.corrected_explanation
                rec.status = "accepted"
            else:
                rec.status = "needs_human"
            return rec
        except Exception as exc:  # API failure after all retries; resumable from the checkpoint
            rec.status, rec.error = "error", f"{type(exc).__name__}: {exc}"
            log.error("question %d failed: %s", index, rec.error)
            return rec

    def run(
        self,
        qtypes: Iterable[str],
        count: int,
        checkpoint: Optional[Path] = None,
        bank: Iterable[Question] = (),
    ) -> RunResult:
        qtypes = list(qtypes)
        meta = self.fingerprint(qtypes)
        done = load_checkpoint(checkpoint, meta) if checkpoint else {}
        # Errored questions are retried on resume; every other status is final.
        todo = [i for i in range(count) if i not in done or done[i].status == "error"]
        log.info("%d questions to process (%d already in checkpoint)", len(todo), count - len(todo))

        lock = threading.Lock()
        if checkpoint:
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
        if checkpoint and checkpoint.exists() and checkpoint.stat().st_size:
            with checkpoint.open("rb") as f:
                f.seek(-1, 2)
                torn = f.read(1) != b"\n"
            if torn:  # finish a line cut short by a crash so the next record starts cleanly
                with checkpoint.open("a", encoding="utf-8") as f:
                    f.write("\n")
        sink = checkpoint.open("a", encoding="utf-8") if checkpoint else None
        if sink and sink.tell() == 0:
            sink.write(json.dumps({"_meta": meta}) + "\n")
        try:
            with ThreadPoolExecutor(max_workers=max(1, self.config.concurrency)) as pool:
                futures = {pool.submit(self.process, i, qtypes[i % len(qtypes)]): i for i in todo}
                for fut in as_completed(futures):
                    rec = fut.result()
                    with lock:
                        done[rec.index] = rec
                        if sink:
                            sink.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                            sink.flush()
        finally:
            if sink:
                sink.close()

        records = [done[i] for i in sorted(done) if i < count]
        flagged = self.dedupe(records, bank)
        return RunResult(records, flagged, self.tracker, set(todo))

    def fingerprint(self, qtypes: list[str]) -> dict:
        """Everything that must match for a checkpoint to be resumed safely."""
        c = self.config
        prompts = "\n".join(
            [
                GEN_SYSTEM,
                GEN_PROMPT,
                SOLVE_MCQ,
                SOLVE_SPR,
                REVIEW_SYSTEM,
                REVIEW_PROMPT,
                REVIEW_KEYLESS_PROMPT,
                self.context,
            ]
        )
        return {
            "models": [self.generator.model, self.solver.model, self.reviewer.model],
            "types": qtypes,
            "runs": c.runs,
            "threshold": c.threshold,
            "salvage": c.salvage,
            "recheck": [c.recheck_runs, c.recheck_needed],
            "review_mode": c.review_mode,
            "seed": c.seed,
            "prompts_sha256": hashlib.sha256(prompts.encode()).hexdigest()[:16],
        }

    def dedupe(self, records: list[Record], bank: Iterable[Question] = ()) -> list[tuple[Record, str, float]]:
        """Mark near-identical questions as duplicates; flag looser matches for a human."""
        index = MinHashIndex()
        for n, q in enumerate(bank):
            index.add(f"bank:{n}", q.fingerprint())
        flagged: list[tuple[Record, str, float]] = []
        for rec in records:
            if rec.status != "accepted":
                continue
            matches = index.query(rec.question.fingerprint())
            if matches and matches[0][1] >= self.config.duplicate_jaccard:
                rec.status, rec.duplicate_of = "duplicate", matches[0][0]
                continue
            flagged.extend((rec, key, round(s, 3)) for key, s in matches if s >= self.config.flag_jaccard)
            index.add(f"run:{rec.index}", rec.question.fingerprint())
        return flagged


def _reviewer_answer(q: Question, raw: object) -> Optional[str]:
    """The reviewer's own answer as a letter or canonical number, or None if it gives no single one.

    Reviewers often name an MCQ answer by its value ("8") or with a gloss ("D (concentrated)").
    """
    if is_nullish(raw) or not isinstance(raw, (str, int, float)):
        return None
    text = str(raw).strip()
    if q.format == SPR:
        value = normalize_spr(text)
        return None if value == "?" else value
    letter = parse_letter(text)
    if letter:
        return letter
    lead = re.match(r"^\(?([A-D])\)?(?:[\s.:,)]|$)", text)
    if lead and not re.search(r"\b(or|though|but)\b", text, re.IGNORECASE):
        return lead.group(1)
    for k, choice in q.choices.items():
        if text.lower() == choice.strip().lower():
            return k
        value = normalize_spr(text)
        if value != "?" and value == normalize_spr(choice):
            return k
    return None


def _top_vote(votes: list[str]) -> tuple[Optional[str], int]:
    counts = Counter(v for v in votes if v != "?")
    if not counts:
        return None, 0
    (top, n), *rest = counts.most_common()
    if rest and rest[0][1] == n:
        return None, 0  # tie: no clear alternative answer
    return top, n


class CheckpointMismatch(RuntimeError):
    pass


def load_checkpoint(path: Path, meta: Optional[dict] = None) -> dict[int, Record]:
    """Load finished records. Refuses a checkpoint written with different models or settings."""
    done: dict[int, Record] = {}
    if not path.exists():
        return done
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            log.warning("skipping unreadable checkpoint line %d (likely a write cut short by a crash)", n + 1)
            continue
        if "_meta" in row:
            if meta is not None and row["_meta"] != meta:
                raise CheckpointMismatch(
                    f"{path} was written by a different run ({row['_meta']}); use --fresh or another --checkpoint"
                )
            continue
        rec = Record.from_dict(row)
        done[rec.index] = rec  # later lines win, so a retried error replaces the old one
    return done


def load_bank(path: Path) -> list[Question]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        Question(
            qtype=r.get("type", r.get("qtype", "")),
            stem=r["stem"],
            choices=r.get("choices") or {},
            answer=str(r["answer"]),
            passage=r.get("passage", ""),
            format=r.get("format", MCQ),
        )
        for r in rows
    ]


def stats(records: list[Record], runs: int = 5) -> dict:
    total = len(records)
    by_status = Counter(r.status for r in records)
    accepted = [r for r in records if r.status == "accepted"]
    reviewed = [r for r in records if r.verdict is not None]
    proposed = [
        r
        for r in reviewed
        if r.verdict.corrected_answer is not None and r.verdict.corrected_answer != r.original_answer
    ]
    per_type: dict[str, dict] = {}
    for r in records:
        t = per_type.setdefault(r.qtype, {"attempted": 0, "accepted": 0})
        t["attempted"] += 1
        t["accepted"] += r.status == "accepted"
    agreement = Counter(r.agreement for r in records if r.question is not None)
    return {
        "attempted": total,
        "status_counts": {s: by_status.get(s, 0) for s in sorted(by_status)},
        "passed_gate": sum(r.passed_screen for r in records),
        "gate_pass_rate": _rate(sum(r.passed_screen for r in records), sum(r.question is not None for r in records)),
        "agreement_histogram": {f"{k}/{runs}": agreement.get(k, 0) for k in range(runs + 1)},
        "reviewed": len(reviewed),
        "salvaged_for_review": sum(1 for r in reviewed if not r.passed_screen),
        "key_corrections_proposed": len(proposed),
        "key_corrections_confirmed": sum(r.key_corrected and r.status == "accepted" for r in records),
        "accepted": len(accepted),
        "acceptance_rate": _rate(len(accepted), total),
        "answer_position_distribution": dict(
            sorted(Counter(r.question.answer for r in accepted if r.question.format == MCQ).items())
        ),
        "per_type": per_type,
    }


def _rate(n: int, d: int) -> Optional[float]:
    return round(n / d, 3) if d else None


def export(records: list[Record], path: Path) -> int:
    """Write accepted questions as JSON ready for bulk upload."""
    rows = [
        {
            "type": r.question.qtype,
            "format": r.question.format,
            "passage": r.question.passage,
            "stem": r.question.stem,
            "choices": r.question.choices,
            "answer": r.question.answer,
            "explanation": r.question.explanation,
        }
        for r in records
        if r.status == "accepted"
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return len(rows)
