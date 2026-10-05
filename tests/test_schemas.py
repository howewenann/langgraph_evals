import pytest
from pydantic import ValidationError

from langgraph_evals.schemas import Judgment


def make_judgment(**overrides) -> Judgment:
    data = {
        "candidate_index": 0,
        "verdict": "supported",
        "issue": "none",
        "matched_reference_indexes": [0],
        "rationale": "Supported by the reference.",
    }
    data.update(overrides)
    return Judgment(**data)


def test_supported_requires_reference_match() -> None:
    with pytest.raises(ValidationError):
        make_judgment(matched_reference_indexes=[])


def test_supported_requires_no_issue() -> None:
    with pytest.raises(ValidationError):
        make_judgment(issue="wrong_fact")


def test_contradiction_requires_conflicting_reference_match() -> None:
    with pytest.raises(ValidationError):
        make_judgment(
            verdict="contradicted",
            issue="wrong_fact",
            matched_reference_indexes=[],
        )


def test_not_in_reference_rejects_reference_matches() -> None:
    with pytest.raises(ValidationError):
        make_judgment(
            verdict="not_in_reference",
            issue="other",
            matched_reference_indexes=[0],
        )


def test_reference_indexes_must_be_unique() -> None:
    with pytest.raises(ValidationError):
        make_judgment(matched_reference_indexes=[0, 0])
