"""Thin FastAPI route."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

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
