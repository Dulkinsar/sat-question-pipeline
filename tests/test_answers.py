from fractions import Fraction

import pytest

from satgen.answers import parse_letter, parse_number


@pytest.mark.parametrize(
    "reply, expected",
    [
        ("B", "B"),
        ("(C)", "C"),
        ("Answer: C", "C"),
        ("The answer is B", "B"),
        ("A is tempting, but the final answer is D", "D"),
        ("I think A or B", None),  # two candidates: ambiguous, counts as "?"
        ('{"answer": "B"}', "B"),
        ("Answer: C. Option A is a common trap.", "C"),
        ("N/A", None),
        ("a reasonable guess", None),  # the article "a" is not a vote
        ("", None),
    ],
)
def test_parse_letter(reply, expected):
    assert parse_letter(reply) == expected


@pytest.mark.parametrize(
    "text, value",
    [
        ("3/4", Fraction(3, 4)),
        ("0.75", Fraction(3, 4)),
        (".75", Fraction(3, 4)),
        ("-2", Fraction(-2)),
        ("x = 9/2", Fraction(9, 2)),
        ("1,200", Fraction(1200)),
        ("\u22123", Fraction(-3)),  # Unicode minus sign
        ("3 / 4", Fraction(3, 4)),
        ("\\frac{3}{4}", Fraction(3, 4)),
        ("x = 5. Check: 2(5)+3 = 13", Fraction(5)),
        ('{"answer": "9/2"}', Fraction(9, 2)),
        ("none", None),
    ],
)
def test_parse_number(text, value):
    assert parse_number(text) == value
