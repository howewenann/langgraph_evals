import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from langgraph_evals import EvaluationCase, Evaluator


CONFIG_PATH = PROJECT_ROOT / "config" / "models.yml"


def main() -> None:
    evaluator = Evaluator(config_path=CONFIG_PATH)

    result = evaluator.evaluate(
        EvaluationCase(
            question="Can I reset my password online?",
            reference="Yes. Use the Forgot Password link.",
            response="Yes. Select Forgot Password.",
        )
    )

    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
