"""Turn model replies into answers. Only the first line is read, because the reasoning
after it mentions other letters.
"""

from __future__ import annotations

import json
import re
import unicodedata
from fractions import Fraction
from typing import Optional

CHOICES = ("A", "B", "C", "D")
NULLISH = {"", "none", "null", "n/a", "na", "-"}

_BARE = re.compile(r"^[\W_]*([A-Da-d])[\W_]*$")
_LABELLED = re.compile(r"(?i:\b(?:final\s+)?(?:answer|choice|option)\b)\s*(?:is|:|=|-)?\s*[(*]*\s*([A-D])(?![\w/])")
_FINAL = re.compile(r"(?i:\bfinal\s+answer\b)\s*(?:is|:|=|-)?\s*[(*]*\s*([A-D])(?![\w/])")
_TOKEN = re.compile(r"(?<![\w/])([A-D])(?![\w/])")

# Characters models emit that look like ASCII but are not.
_TRANSLATE = str.maketrans({"−": "-", "–": "-", "—": "-", "⁄": "/", "∕": "/", "\u00a0": " "})


def _clean(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_TRANSLATE).strip()


def _json_answer(text: str) -> Optional[str]:
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    value = data.get("answer") if isinstance(data, dict) else None
    return None if value is None else str(value)


def _first_line(text: str) -> str:
    return next((line.strip() for line in text.splitlines() if line.strip()), "")


def parse_letter(text: str) -> Optional[str]:
    """Extract the chosen letter from a solver reply, or None if it cannot be read safely."""
    text = _clean(text)
    if not text:
        return None
    from_json = _json_answer(text)
    if from_json is not None:
        m = _BARE.match(from_json.strip())
        return m.group(1).upper() if m else None
    final = _FINAL.findall(text)
    if final:
        return final[-1]
    line = _first_line(text)
    m = _BARE.match(line)
    if m:
        return m.group(1).upper()
    labelled = _LABELLED.findall(line)
    if labelled:
        return labelled[0]
    tokens = set(_TOKEN.findall(line))
    return tokens.pop() if len(tokens) == 1 else None


_NUM = r"-?(?:\d+(?:\.\d+)?|\.\d+)"
_FRACTION = re.compile(rf"^({_NUM})\s*/\s*({_NUM})$")
_LATEX_FRAC = re.compile(r"\\[dt]?frac\{\s*(-?[\d.]+)\s*\}\{\s*(-?[\d.]+)\s*\}")
_LEADING = re.compile(rf"^(?:[a-z]\s*=\s*)?({_NUM}(?:\s*/\s*{_NUM})?)(?!\d|\.\d)", re.IGNORECASE)
_NUMBERS = re.compile(rf"{_NUM}(?:\s*/\s*{_NUM})?")


def _to_fraction(token: str) -> Optional[Fraction]:
    token = token.replace(" ", "")
    m = _FRACTION.match(token)
    try:
        if m:
            return Fraction(m.group(1)) / Fraction(m.group(2))
        return Fraction(token)
    except (ValueError, ZeroDivisionError):
        return None


def parse_number(text: str) -> Optional[Fraction]:
    """Parse an SPR answer ("3/4", "0.75", "−2", "x = 9/2", "\\frac{3}{4}") into an exact fraction."""
    text = _clean(text).replace(",", "")
    if not text:
        return None
    from_json = _json_answer(text)
    if from_json is not None:
        text = _clean(from_json).replace(",", "")
    text = _LATEX_FRAC.sub(r"\1/\2", text).replace("$", "")
    line = _first_line(text)
    whole = _to_fraction(line)
    if whole is not None:
        return whole
    m = _LEADING.match(line)
    if m:
        return _to_fraction(m.group(1))
    found = {_to_fraction(t) for t in _NUMBERS.findall(line)} - {None}
    return found.pop() if len(found) == 1 else None


def snap_decimal(value: Fraction, text: str) -> Fraction:
    """Treat a rounded decimal like .333 or .6667 as the simple fraction it approximates.

    The SAT accepts a decimal answer that fills the grid, so 0.333 and 1/3 are the same answer.
    """
    if "." not in text:
        return value
    digits = len(text.split(".")[-1].rstrip())
    if digits < 3:
        return value
    simple = value.limit_denominator(100)
    return simple if abs(simple - value) < Fraction(1, 10**digits) else value


def normalize_spr(text: str) -> str:
    """Canonical string for an SPR vote so votes can be compared and counted."""
    value = parse_number(text)
    if value is None:
        return "?"
    decimal = re.search(r"-?\d*\.\d+", _clean(text))
    if decimal:
        value = snap_decimal(value, decimal.group(0))
    return str(value.numerator) if value.denominator == 1 else f"{value.numerator}/{value.denominator}"


def parse_bool(value: object) -> bool:
    """Strict boolean parsing for reviewer JSON: bool("false") is True in Python."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        return value.strip().lower() == "true"
    raise ValueError(f"expected a boolean, got {value!r}")


def is_nullish(value: object) -> bool:
    return value is None or (isinstance(value, str) and value.strip().lower() in NULLISH)
