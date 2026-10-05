"""Small public Python API around the evaluation graph."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import yaml
from langchain_anyllm import ChatAnyLLM

from .graph import build_graph
from .schemas import EvaluationCase, EvaluationResult


DEFAULT_CONFIG_PATH = Path("config/models.yml")


def _resolve_config_path(config_path: str | Path) -> Path:
    path = Path(config_path).expanduser()
    if path.is_absolute() or path.exists():
        return path

    # Make the repository default work even when called from another cwd
    # during an editable install.
    project_relative = Path(__file__).resolve().parents[2] / path
    if project_relative.exists():
        return project_relative

    return path


def _model_from_config(config_path: str | Path):
    path = _resolve_config_path(config_path)

    try:
        with path.open("r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Model config not found: {path}. "
            "Pass config_path explicitly or run from the repository root."
        ) from exc

    if not isinstance(raw_config, dict) or not isinstance(
        raw_config.get("model"), dict
    ):
        raise ValueError(
            f"Invalid model config in {path}: expected a top-level 'model' mapping"
        )

    config = raw_config["model"]
    model_name = config.get("name")
    if not model_name:
        raise ValueError(f"Invalid model config in {path}: model.name is required")

    return ChatAnyLLM(
        model=model_name,
        provider=config.get("provider"),
        api_base=config.get("api_base"),
        api_key=config.get("api_key"),
        temperature=config.get("temperature", 0),
    )


class Evaluator:
    """Evaluate candidate answers against authoritative reference answers."""

    def __init__(
        self,
        config_path: str | Path = DEFAULT_CONFIG_PATH,
        *,
        model: Any | None = None,
    ):
        # Model injection keeps the graph reusable with any compatible LangChain
        # chat model and makes deterministic tests possible without config files.
        self.graph = build_graph(
            model if model is not None else _model_from_config(config_path)
        )

    def evaluate(self, case: EvaluationCase) -> EvaluationResult:
        return self.graph.invoke(case.model_dump())["result"]

    def evaluate_many(
        self,
        cases: list[EvaluationCase],
        *,
        max_concurrency: int = 4,
    ) -> list[EvaluationResult]:
        states = self.graph.batch(
            [case.model_dump() for case in cases],
            config={"max_concurrency": max_concurrency},
        )
        return [state["result"] for state in states]

    def evaluate_many_as_completed(
        self,
        cases: list[EvaluationCase],
        *,
        max_concurrency: int = 4,
    ) -> Iterator[tuple[int, EvaluationResult]]:
        states = self.graph.batch_as_completed(
            [case.model_dump() for case in cases],
            config={"max_concurrency": max_concurrency},
        )

        for index, state in states:
            yield index, state["result"]


# Backwards-compatible alias for the project's previous public class name.
AutoTester = Evaluator
