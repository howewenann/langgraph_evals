"""Pydantic models for input, LLM output, and final output."""

from typing import Literal

from pydantic import BaseModel, Field, model_validator


UnitKind = Literal[
    "fact",
    "instruction",
    "constraint",
    "warning",
    "definition",
    "resource",
    "other",
]

Verdict = Literal[
    "supported",
    "contradicted",
    "not_in_reference",
    "unclear",
]

Issue = Literal[
    "none",
    "wrong_fact",
    "unsafe",
    "missing_precondition",
    "other",
]


class EvaluationCase(BaseModel):
    """One evaluation request."""

    id: str | int | None = None
    question: str
    reference: str
    response: str


class QuestionType(BaseModel):
    is_binary: bool


class Unit(BaseModel):
    """One atomic user-facing statement."""

    kind: UnitKind
    text: str


class ExtractedAnswer(BaseModel):
    polarity: Literal["yes", "no", "unknown"]
    units: list[Unit]


class Judgment(BaseModel):
    """Minimal judgment produced by the judge LLM."""

    candidate_index: int = Field(
        ge=0,
        description="0-based index of the candidate unit being judged.",
    )

    verdict: Verdict = Field(
        description=(
            "Evaluation result for the candidate. "
            "Must be one of: supported, contradicted, "
            "not_in_reference, unclear."
        )
    )

    issue: Issue = Field(
        description=(
            "Type of problem found. "
            "Must be one of: none, wrong_fact, unsafe, "
            "missing_precondition, other. "
            "IMPORTANT: not_in_reference is a verdict, NOT an issue. "
            "For a not_in_reference verdict with no more specific issue, "
            "use other."
        )
    )

    matched_reference_indexes: list[int] = Field(
        default_factory=list,
        description=(
            "Reference indexes that support or conflict with this candidate. "
            "Must be non-empty for supported and contradicted verdicts. "
            "Must be empty for not_in_reference and unclear verdicts."
        ),
    )

    rationale: str

    @model_validator(mode="after")
    def validate_judgment(self):
        """Enforce relationships between verdict, issue, and reference matches."""

        if len(self.matched_reference_indexes) != len(
            set(self.matched_reference_indexes)
        ):
            raise ValueError("reference indexes must be unique")

        if self.verdict == "supported":
            if not self.matched_reference_indexes:
                raise ValueError(
                    "supported judgments need at least one reference match"
                )

            if self.issue != "none":
                raise ValueError(
                    "supported judgments must have issue='none'"
                )

        elif self.verdict == "contradicted":
            if not self.matched_reference_indexes:
                raise ValueError(
                    "contradicted judgments need the reference units they conflict with"
                )

        elif self.matched_reference_indexes:
            raise ValueError(
                f"{self.verdict} judgments cannot contain reference matches"
            )

        return self


class JudgeResult(BaseModel):
    judgments: list[Judgment]
    rationale: str


class JudgmentDetail(BaseModel):
    """Human-readable judgment returned by the evaluator."""

    candidate_index: int
    candidate_unit: Unit

    verdict: Verdict
    issue: Issue

    matched_reference_indexes: list[int]
    matched_reference_units: list[Unit]

    rationale: str


class EvaluationResult(BaseModel):
    """Final public output."""

    id: str | int | None

    question_is_binary: bool
    polarity_match: bool | None

    reference_units: list[Unit]
    response_units: list[Unit]

    precision: float
    recall: float
    f1: float

    rationale: str

    missing_reference_units: list[Unit]
    judgments: list[JudgmentDetail]
