from fractions import Fraction

import pytest

from satgen.answers import normalize_spr, parse_letter, parse_number


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
        ("2,5", Fraction(5, 2)),  # a decimal comma, not twenty-five
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


def test_only_the_answer_decimal_is_snapped():
    # The first decimal in the reply (0.33) is working, not the answer; snap the answer itself.
    assert normalize_spr('{"work": "about 0.33", "answer": "0.3333"}') == "1/3"
    assert normalize_spr("x = 0.333") == "1/3"
