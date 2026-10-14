"""Data models shared across the pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional

from .answers import CHOICES, parse_number

MCQ = "mcq"
SPR = "spr"  # student-produced response: the student types a number, there are no choices


@dataclass
class Question:
    """A Digital SAT question, either multiple choice or student-produced response."""

    qtype: str
    stem: str
    choices: dict[str, str]
    answer: str
    explanation: str = ""
    passage: str = ""
    format: str = MCQ

    def fingerprint(self) -> str:
        """Normalized text used for duplicate detection."""
        return " ".join((self.passage + " " + self.stem).lower().split())

    def to_prompt(self) -> str:
        parts = []
        if self.passage:
            parts.append(self.passage)
        parts.append(self.stem)
        parts.extend(f"{k}) {v}" for k, v in sorted(self.choices.items()))
        return "\n\n".join(parts)

    def validate(self) -> None:
        if not self.stem.strip():
            raise ValueError("empty stem")
        if self.format == MCQ:
            if set(self.choices) != set(CHOICES):
                raise ValueError("choices must be exactly A, B, C, D")
            if self.answer not in CHOICES:
                raise ValueError(f"answer must be one of {CHOICES}")
            texts = [" ".join(v.lower().split()) for v in self.choices.values()]
            if len(set(texts)) != len(texts):
                raise ValueError("two choices have the same text")
        elif self.format == SPR:
            if self.choices:
                raise ValueError("student-produced response questions have no choices")
            if parse_number(self.answer) is None:
                raise ValueError(f"SPR answer {self.answer!r} is not a number or fraction")
        else:
            raise ValueError(f"unknown format {self.format!r}")

    @classmethod
    def from_dict(cls, d: dict) -> "Question":
        return cls(
            qtype=d["qtype"],
            stem=d["stem"],
            choices=dict(d.get("choices") or {}),
            answer=str(d["answer"]),
            explanation=d.get("explanation", ""),
            passage=d.get("passage", ""),
            format=d.get("format", MCQ),
        )


@dataclass
class Verdict:
    """Outcome of the review step."""

    accepted: bool
    corrected_answer: Optional[str] = None
    corrected_explanation: Optional[str] = None
    notes: str = ""


# Terminal states a record can end in. Only "accepted" is exported.
STATUSES = (
    "generation_failed",  # the generator returned something that does not parse or validate
    "error",  # an API call kept failing after retries
    "screened_out",  # fewer than `threshold` solver runs matched the key
    "rejected",  # the reviewer rejected the question
    "needs_human",  # reviewer and solvers disagree after a key correction
    "duplicate",  # near-identical to an accepted question or to the existing bank
    "accepted",
)


@dataclass
class Record:
    """One question attempt plus everything the pipeline learned about it."""

    index: int
    qtype: str
    question: Optional[Question] = None
    original_answer: Optional[str] = None
    solve_votes: list[str] = field(default_factory=list)
    agreement: int = 0
    passed_screen: bool = False
    verdict: Optional[Verdict] = None
    recheck_votes: list[str] = field(default_factory=list)
    status: str = "generation_failed"
    error: str = ""
    duplicate_of: str = ""

    @property
    def key_corrected(self) -> bool:
        return (
            self.question is not None
            and self.original_answer is not None
            and self.question.answer != self.original_answer
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Record":
        d = dict(d)
        q = d.pop("question", None)
        v = d.pop("verdict", None)
        return cls(
            question=Question.from_dict(q) if q else None,
            verdict=Verdict(**v) if v else None,
            **d,
        )
