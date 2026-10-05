import pytest

from langgraph_evals.schemas import (
    ExtractedAnswer,
    JudgeResult,
    Judgment,
    QuestionType,
    Unit,
)
from langgraph_evals.scoring import score


def unit(text: str) -> Unit:
    return Unit(kind="fact", text=text)


def test_score_normalizes_judgment_order_and_finds_missing_units() -> None:
    reference = ExtractedAnswer(
        polarity="unknown",
        units=[unit("A"), unit("B"), unit("C")],
    )
    response = ExtractedAnswer(
        polarity="unknown",
        units=[unit("A"), unit("extra")],
    )
    judge = JudgeResult(
        judgments=[
            Judgment(
                candidate_index=1,
                verdict="not_in_reference",
                issue="other",
                matched_reference_indexes=[],
                rationale="Extra information.",
            ),
            Judgment(
                candidate_index=0,
                verdict="supported",
                issue="none",
                matched_reference_indexes=[0],
                rationale="Matches A.",
            ),
        ],
        rationale="A is covered; B and C are missing; one extra claim was added.",
    )

    result = score(
        "case-1",
        QuestionType(is_binary=False),
        reference,
        response,
        judge,
    )

    assert [j.candidate_index for j in result.judgments] == [0, 1]
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1 / 3)
    assert result.f1 == pytest.approx(0.4)
    assert [u.text for u in result.missing_reference_units] == ["B", "C"]
    assert result.polarity_match is None


def test_binary_polarity_is_reported_separately_from_unit_scores() -> None:
    reference = ExtractedAnswer(polarity="yes", units=[unit("A")])
    response = ExtractedAnswer(polarity="no", units=[unit("A")])
    judge = JudgeResult(
        judgments=[
            Judgment(
                candidate_index=0,
                verdict="supported",
                issue="none",
                matched_reference_indexes=[0],
                rationale="Matches A.",
            )
        ],
        rationale="The content matches.",
    )

    result = score(
        None,
        QuestionType(is_binary=True),
        reference,
        response,
        judge,
    )

    assert result.precision == 1.0
    assert result.recall == 1.0
    assert result.f1 == 1.0
    assert result.polarity_match is False


def test_score_rejects_invalid_reference_index() -> None:
    reference = ExtractedAnswer(polarity="unknown", units=[unit("A")])
    response = ExtractedAnswer(polarity="unknown", units=[unit("A")])
    judge = JudgeResult(
        judgments=[
            Judgment(
                candidate_index=0,
                verdict="supported",
                issue="none",
                matched_reference_indexes=[1],
                rationale="Bad model index.",
            )
        ],
        rationale="Invalid index.",
    )

    with pytest.raises(ValueError, match="invalid reference index"):
        score(
            None,
            QuestionType(is_binary=False),
            reference,
            response,
            judge,
        )
