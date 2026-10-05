"""Thin FastAPI route."""

from fastapi import FastAPI
from langgraph_evals import EvaluationCase, EvaluationResult, Evaluator


app = FastAPI(title="LangGraph Evals")
evaluator = Evaluator()


@app.post("/evaluate", response_model=EvaluationResult)
def evaluate(case: EvaluationCase) -> EvaluationResult:
    return evaluator.evaluate(case)


@app.post("/evaluate/batch", response_model=list[EvaluationResult])
def evaluate_batch(cases: list[EvaluationCase]) -> list[EvaluationResult]:
    return evaluator.evaluate_many(cases)
