"""Builds data/eval/eval_set.jsonl: each base item as good, bad_key (key moved to a distractor) and ambiguous (a second correct answer planted)."""

from __future__ import annotations

import json
import random
from fractions import Fraction
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "data" / "eval" / "eval_set.jsonl"
RNG = random.Random(2025)

# (sentence, target word, correct meaning, three distractors, second defensible meaning)
WORDS = [
    (
        "Although the report ran only six pages, its argument was dense, packing a decade of field data into a few tightly reasoned paragraphs.",
        "dense",
        "concentrated",
        ["heavy", "crowded", "dim"],
        "compact",
    ),
    (
        "The critic's review was measured: she praised the cast's energy but noted that the second act dragged.",
        "measured",
        "restrained",
        ["quantified", "rhythmic", "lengthy"],
        "moderate",
    ),
    (
        "Early settlers found the valley's soil so poor that only the hardiest crops could take hold.",
        "poor",
        "infertile",
        ["inexpensive", "pitiable", "humble"],
        "barren",
    ),
    (
        "The new manager's first decision was to table the expansion plan until the budget review was complete.",
        "table",
        "postpone",
        ["display", "propose", "furnish"],
        "shelve",
    ),
    (
        "Researchers hoped the drug would arrest the spread of the infection before it reached the lungs.",
        "arrest",
        "halt",
        ["detain", "capture", "attract"],
        "stop",
    ),
    (
        "The historian's account is sound: every claim is supported by letters and shipping records from the period.",
        "sound",
        "reliable",
        ["noisy", "healthy", "asleep"],
        "valid",
    ),
    (
        "Because the instructions were so involved, most volunteers needed an hour just to assemble the first kit.",
        "involved",
        "complicated",
        ["included", "engaged", "committed"],
        "intricate",
    ),
    (
        "The scientist was careful not to discount the outlier, since it might reveal a flaw in the model.",
        "discount",
        "dismiss",
        ["mark down", "subtract", "recount"],
        "disregard",
    ),
    (
        "Her reputation for economy meant that the committee trusted her to cut the budget without harming the program.",
        "economy",
        "thrift",
        ["the market", "the system", "commerce"],
        "frugality",
    ),
    (
        "The playwright's later works are marked by a spare style, with little scenery and even less dialogue.",
        "spare",
        "minimal",
        ["extra", "leftover", "generous"],
        "sparse",
    ),
    (
        "The council's proposal met with qualified approval: members supported it only if the costs were cut.",
        "qualified",
        "limited",
        ["certified", "skilled", "eligible"],
        "conditional",
    ),
    (
        "The biologist argued that the data could not bear the weight of such a sweeping conclusion.",
        "bear",
        "support",
        ["produce", "resemble", "remove"],
        "sustain",
    ),
    (
        "Visitors were struck by how the museum's modest entrance belied the size of the galleries inside.",
        "belied",
        "disguised",
        ["revealed", "exaggerated", "confirmed"],
        "concealed",
    ),
    (
        "The mayor's remarks were deliberately vague, allowing both sides to claim her support.",
        "vague",
        "imprecise",
        ["dreamy", "faint", "empty"],
        "unclear",
    ),
    (
        "Few readers had the patience for the novel's digressive structure, which wandered through dozens of minor stories.",
        "digressive",
        "rambling",
        ["aggressive", "descending", "concise"],
        "meandering",
    ),
    (
        "The treaty was meant to check the growing power of the northern states.",
        "check",
        "restrain",
        ["inspect", "mark", "verify"],
        "curb",
    ),
    (
        "The inventor's notebooks reveal a singular mind, one unlike any of his contemporaries'.",
        "singular",
        "unique",
        ["lonely", "numerical", "ordinary"],
        "extraordinary",
    ),
    (
        "The general's account of the battle was colored by his wish to appear blameless.",
        "colored",
        "influenced",
        ["painted", "decorated", "brightened"],
        "skewed",
    ),
    (
        "The ambassador's tone was cordial, though both sides knew the negotiations would be difficult.",
        "cordial",
        "friendly",
        ["sugary", "hostile", "indifferent"],
        "warm",
    ),
    (
        "Critics charged that the study's sample was too narrow to represent the national population.",
        "narrow",
        "limited",
        ["slender", "close", "thin"],
        "restricted",
    ),
    (
        "The firm's founder was known for her candor; she told investors exactly what she thought of every proposal.",
        "candor",
        "frankness",
        ["cleverness", "sweetness", "caution"],
        "honesty",
    ),
    (
        "The governor moved to curb spending after revenues fell short of forecasts.",
        "curb",
        "limit",
        ["edge", "sidewalk", "encourage"],
        "restrain",
    ),
    (
        "Although the theory was novel, it built on ideas that had circulated among physicists for decades.",
        "novel",
        "new",
        ["fictional", "literary", "lengthy"],
        "original",
    ),
    (
        "The witness gave a faithful account of events, omitting nothing that she had seen.",
        "faithful",
        "accurate",
        ["loyal", "devout", "hopeful"],
        "exact",
    ),
    (
        "The professor's lecture was so dry that several students struggled to stay awake.",
        "dry",
        "dull",
        ["arid", "thirsty", "waterless"],
        "boring",
    ),
    (
        "The engineer's solution was elegant: a single valve replaced an entire system of pumps.",
        "elegant",
        "neatly effective",
        ["fashionable", "graceful in movement", "expensive"],
        "ingenious",
    ),
    (
        "The committee will entertain proposals from any department until the end of the month.",
        "entertain",
        "consider",
        ["amuse", "host", "delight"],
        "weigh",
    ),
    (
        "The explorer's journals give a vivid picture of life on the frozen frontier, down to the smallest daily chores.",
        "vivid",
        "detailed",
        ["brightly colored", "imaginary", "noisy"],
        "realistic",
    ),
]

# (two-sentence passage with a blank, correct transition, three distractors, second defensible transition)
TRANSITIONS = [
    (
        "Bees are known for pollinating flowering crops. ____ some species of bats also pollinate plants, especially in tropical regions.",
        "Similarly,",
        ["Therefore,", "For instance,", "In contrast,"],
        "Likewise,",
    ),
    (
        "The museum expected about 2,000 visitors on opening day. ____ more than 5,000 people arrived, and the doors had to close by noon.",
        "Instead,",
        ["Similarly,", "Therefore,", "For example,"],
        "However,",
    ),
    (
        "The bridge's steel cables had begun to corrode. ____ engineers closed it to heavy trucks until repairs were complete.",
        "Consequently,",
        ["Nevertheless,", "Similarly,", "For example,"],
        "As a result,",
    ),
    (
        "Many desert plants store water in thick leaves. ____ the aloe plant can survive months without rain.",
        "For example,",
        ["However,", "In contrast,", "Nevertheless,"],
        "For instance,",
    ),
    (
        "The first draft of the novel was over 900 pages long. ____ the published version runs to just 350.",
        "By contrast,",
        ["Similarly,", "Therefore,", "For instance,"],
        "In contrast,",
    ),
    (
        "The study found no link between screen time and sleep quality in adults. ____ it found a strong link in teenagers.",
        "However,",
        ["Therefore,", "Likewise,", "For example,"],
        "In contrast,",
    ),
    (
        "Rainfall in the region has dropped by half since 1990. ____ local farmers have switched to drought-resistant crops.",
        "As a result,",
        ["Nevertheless,", "For example,", "Similarly,"],
        "Consequently,",
    ),
    (
        "The team had trained for months and was in top form. ____ it lost the opening match by a wide margin.",
        "Nevertheless,",
        ["Therefore,", "For example,", "Similarly,"],
        "Even so,",
    ),
    (
        "Octopuses can change color in a fraction of a second. ____ they can alter the texture of their skin to match rocks or coral.",
        "Moreover,",
        ["However,", "Therefore,", "Instead,"],
        "In addition,",
    ),
    (
        "The recipe calls for fresh basil. ____ dried basil can be used, though the flavor will be milder.",
        "Alternatively,",
        ["Therefore,", "Similarly,", "For example,"],
        "If necessary,",
    ),
]


def fmt(x: Fraction) -> str:
    return str(x.numerator) if x.denominator == 1 else f"{x.numerator}/{x.denominator}"


def equivalent_form(x: Fraction) -> str:
    """A second, differently written choice with the same value as x."""
    if x.denominator == 1:
        return f"{2 * x.numerator}/2"
    return str(float(x)) if (float(x) * 100).is_integer() else f"{2 * x.numerator}/{2 * x.denominator}"


def math_items() -> list[dict]:
    items = []
    while len(items) < 15:  # a x + b = c
        a, x, b = RNG.randint(2, 9), RNG.choice([v for v in range(-9, 13) if v]), RNG.randint(1, 20)
        c = a * x + b
        right = Fraction(x)
        wrong = {Fraction(c + b, a), Fraction(c - b), Fraction(c, a) - b}
        wrong.discard(right)
        if len(wrong) < 3:
            continue
        items.append(
            {
                "qtype": "linear_equations",
                "passage": "",
                "stem": f"If {a}x + {b} = {c}, what is the value of x?",
                "right": fmt(right),
                "wrong": [fmt(w) for w in sorted(wrong)][:3],
                "alt": equivalent_form(right),
            }
        )
    while len(items) < 30:  # fee + rate * months = total
        fee, rate, months = RNG.randint(2, 9) * 10, RNG.randint(15, 60), RNG.randint(3, 14)
        total = fee + rate * months
        right = Fraction(months)
        wrong = {Fraction(total, rate), Fraction(total + fee, rate), Fraction(total - fee)}
        wrong = {w for w in wrong if w != right}
        if len(wrong) < 3:
            continue
        items.append(
            {
                "qtype": "linear_equations",
                "passage": "",
                "stem": (
                    f"A gym charges a one-time sign-up fee of ${fee} plus ${rate} per month. "
                    f"Maya paid ${total} in total. For how many months did she pay?"
                ),
                "right": fmt(right),
                "wrong": [fmt(w) for w in sorted(wrong)][:3],
                "alt": equivalent_form(right),
            }
        )
    return items


def spr_items() -> list[dict]:
    items = []
    while len(items) < 15:  # p x - q = r x + s, often a fractional answer
        p, r = RNG.randint(3, 9), RNG.randint(1, 2)
        q, s = RNG.randint(1, 15), RNG.randint(1, 15)
        x = Fraction(s + q, p - r)
        items.append(
            {
                "qtype": "spr_linear_equations",
                "stem": f"If {p}x - {q} = {r}x + {s}, what is the value of x?",
                "right": fmt(x),
                "wrong_value": fmt(Fraction(s - q, p - r) if s != q else x + 1),
            }
        )
    return items


def mcq(qtype: str, passage: str, stem: str, right: str, wrong: list[str]) -> tuple[dict, str]:
    texts = [right, *wrong]
    RNG.shuffle(texts)
    choices = dict(zip("ABCD", texts, strict=True))
    return {"qtype": qtype, "format": "mcq", "passage": passage, "stem": stem, "choices": choices}, "ABCD"[
        texts.index(right)
    ]


def variants(base_id: str, qtype: str, passage: str, stem: str, right: str, wrong: list[str], alt: str) -> list[dict]:
    rows = []
    q, key = mcq(qtype, passage, stem, right, wrong)
    rows.append(
        {
            "id": f"{base_id}-good",
            "base": base_id,
            "label": "good",
            "true_answer": key,
            "question": {**q, "answer": key},
        }
    )
    bad = RNG.choice([k for k in "ABCD" if k != key])
    rows.append(
        {
            "id": f"{base_id}-badkey",
            "base": base_id,
            "label": "bad_key",
            "true_answer": key,
            "question": {**q, "answer": bad},
        }
    )
    q2, key2 = mcq(qtype, passage, stem, right, [alt, *wrong[1:]])
    rows.append(
        {
            "id": f"{base_id}-ambiguous",
            "base": base_id,
            "label": "ambiguous",
            "true_answer": None,
            "question": {**q2, "answer": key2},
        }
    )
    return rows


def main() -> None:
    rows: list[dict] = []
    for n, (sentence, word, right, wrong, alt) in enumerate(WORDS):
        rows += variants(
            f"wic{n:02d}",
            "words_in_context",
            sentence,
            f'As used in the text, what does the word "{word}" most nearly mean?',
            right,
            wrong,
            alt,
        )
    for n, (passage, right, wrong, alt) in enumerate(TRANSITIONS):
        rows += variants(
            f"tr{n:02d}",
            "transitions",
            passage,
            "Which choice completes the text with the most logical transition?",
            right,
            wrong,
            alt,
        )
    for n, it in enumerate(math_items()):
        rows += variants(f"math{n:02d}", it["qtype"], it["passage"], it["stem"], it["right"], it["wrong"], it["alt"])
    for n, it in enumerate(spr_items()):
        base = {"qtype": it["qtype"], "format": "spr", "passage": "", "stem": it["stem"], "choices": {}}
        rows.append(
            {
                "id": f"spr{n:02d}-good",
                "base": f"spr{n:02d}",
                "label": "good",
                "true_answer": it["right"],
                "question": {**base, "answer": it["right"]},
            }
        )
        rows.append(
            {
                "id": f"spr{n:02d}-badkey",
                "base": f"spr{n:02d}",
                "label": "bad_key",
                "true_answer": it["right"],
                "question": {**base, "answer": it["wrong_value"]},
            }
        )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    labels = {lab: sum(r["label"] == lab for r in rows) for lab in ("good", "bad_key", "ambiguous")}
    print(f"wrote {len(rows)} items to {OUT}: {labels}")


if __name__ == "__main__":
    main()
