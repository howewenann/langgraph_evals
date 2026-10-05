"""Deterministic scoring and result enrichment. No LLM calls here."""

from .schemas import (
    EvaluationResult,
    ExtractedAnswer,
    JudgeResult,
    JudgmentDetail,
    QuestionType,
)


def score(
    case_id: str | int | None,
    question_type: QuestionType,
    reference: ExtractedAnswer,
    response: ExtractedAnswer,
    judge: JudgeResult,
) -> EvaluationResult:
    # There must be exactly one judgment for every candidate unit.
    expected = set(range(len(response.units)))
    actual = {j.candidate_index for j in judge.judgments}

    if len(judge.judgments) != len(response.units) or actual != expected:
        raise ValueError(
            "Judge must return exactly one judgment per candidate unit"
        )

    # Validate all model-generated reference indexes before dereferencing them.
    for judgment in judge.judgments:
        for index in judgment.matched_reference_indexes:
            if index < 0 or index >= len(reference.units):
                raise ValueError(
                    f"Judge returned invalid reference index: {index}"
                )

    # Structured output should already be ordered, but normalize it so the
    # public result remains deterministic if the model returns the right set
    # of judgments in a different order.
    ordered_judgments = sorted(
        judge.judgments,
        key=lambda judgment: judgment.candidate_index,
    )

    supported = [
        judgment
        for judgment in ordered_judgments
        if judgment.verdict == "supported"
    ]

    matched_reference_indexes = {
        index
        for judgment in supported
        for index in judgment.matched_reference_indexes
    }

    missing_reference_units = [
        unit
        for index, unit in enumerate(reference.units)
        if index not in matched_reference_indexes
    ]

    precision = (
        len(supported) / len(response.units)
        if response.units
        else 0.0
    )

    recall = (
        len(matched_reference_indexes) / len(reference.units)
        if reference.units
        else 1.0
    )

    f1 = (
        0.0
        if precision + recall == 0
        else 2 * precision * recall / (precision + recall)
    )

    if not question_type.is_binary:
        polarity_match = None
    elif "unknown" in {reference.polarity, response.polarity}:
        polarity_match = None
    else:
        polarity_match = reference.polarity == response.polarity

    # Convert the judge's compact index-based output into useful
    # human-readable evidence.
    judgment_details = [
        JudgmentDetail(
            candidate_index=judgment.candidate_index,
            candidate_unit=response.units[judgment.candidate_index],
            verdict=judgment.verdict,
            issue=judgment.issue,
            matched_reference_indexes=judgment.matched_reference_indexes,
            matched_reference_units=[
                reference.units[index]
                for index in judgment.matched_reference_indexes
            ],
            rationale=judgment.rationale,
        )
        for judgment in ordered_judgments
    ]

    return EvaluationResult(
        id=case_id,
        question_is_binary=question_type.is_binary,
        polarity_match=polarity_match,
        reference_units=reference.units,
        response_units=response.units,
        precision=precision,
        recall=recall,
        f1=f1,
        rationale=judge.rationale,
        missing_reference_units=missing_reference_units,
        judgments=judgment_details,
    )
