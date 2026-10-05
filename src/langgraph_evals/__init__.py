"""Public API for langgraph-evals."""

from typing import TYPE_CHECKING

from .schemas import EvaluationCase, EvaluationResult

if TYPE_CHECKING:
    from .evaluator import AutoTester, Evaluator

__all__ = ["Evaluator", "AutoTester", "EvaluationCase", "EvaluationResult"]


def __getattr__(name: str):
    # Avoid importing the model stack when callers only need schemas/results.
    if name in {"Evaluator", "AutoTester"}:
        from .evaluator import AutoTester, Evaluator

        globals().update(Evaluator=Evaluator, AutoTester=AutoTester)
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
