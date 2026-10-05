from pathlib import Path

from langgraph_evals import EvaluationCase, Evaluator


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yml"


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
