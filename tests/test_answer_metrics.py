import pytest

from ragbasics.eval.answer_metrics import fit_guesser, is_abstention, match_verdict, normalise
from ragbasics.types import Question


def test_normalise_ignores_case_punctuation_articles_and_spacing():
    assert normalise("  The Verge. ") == "verge"
    assert normalise("Yes.") == normalise("yes") == "yes"
    assert normalise("Sam  Bankman-Fried") == normalise("sam bankman-fried")


@pytest.mark.parametrize(
    ("answer", "reference", "expected"),
    [
        ("Yes.", "Yes", True),
        ("sam bankman-fried", "Sam Bankman-Fried", True),
        ("No", "Yes", False),  # both from the closed set and different
        ("Insufficient information.", "Insufficient information.", True),
        ("Insufficient information.", "Spotify", False),  # declined an answerable question
        ("Yes", "Insufficient information.", False),  # answered an unanswerable one
        # Not bare: the judge decides.
        ("The passages name him as Sam Bankman-Fried.", "Sam Bankman-Fried", None),
        ("SBF", "Sam Bankman-Fried", None),
        ("Yes, both articles say so.", "Yes", None),
        ("The passages do not say.", "Insufficient information.", None),
    ],
)
def test_match_verdict(answer, reference, expected):
    assert match_verdict(answer, reference) is expected


def test_a_three_sentence_answer_that_ends_by_declining_is_an_abstention():
    answer = "The passages cover the Spotify profit. They say nothing of AI songs. " \
        "Insufficient information."
    assert is_abstention(answer)
    assert match_verdict(answer, "Insufficient information.") is True
    assert match_verdict(answer, "Spotify") is False


def test_mentioning_missing_information_mid_answer_is_not_an_abstention():
    assert not is_abstention("Spotify, although there is insufficient information on the DJ.")
    assert not is_abstention("Spotify")


def test_guesser_gives_each_type_its_most_common_answer():
    questions = [
        Question("q1", "", "Yes", "comparison_query"),
        Question("q2", "", "Yes", "comparison_query"),
        Question("q3", "", "No", "comparison_query"),
        Question("q4", "", "Insufficient information.", "null_query"),
    ]
    assert fit_guesser(questions) == {
        "comparison_query": "Yes",
        "null_query": "Insufficient information.",
    }
