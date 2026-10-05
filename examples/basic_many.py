from pathlib import Path

from tqdm.auto import tqdm

from langgraph_evals import EvaluationCase, EvaluationResult, Evaluator


CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "models.yml"


CASES = [
    EvaluationCase(
        id="perfect_multi_unit",
        question="How do I reset my password?",
        reference=(
            "Open the login page. "
            "Select Forgot Password. "
            "Enter your registered email address. "
            "Then follow the reset link sent to your email."
        ),
        response=(
            "Go to the login page and select Forgot Password. "
            "Enter your registered email, then use the reset link sent to you."
        ),
    ),
    EvaluationCase(
        id="missing_unit",
        question="How do I reset my password?",
        reference=(
            "Open the login page. "
            "Select Forgot Password. "
            "Enter your registered email address. "
            "Then follow the reset link sent to your email."
        ),
        response=(
            "Open the login page and select Forgot Password. "
            "Enter your registered email address."
        ),
    ),
    EvaluationCase(
        id="contradiction",
        question="Can I change my username after registration?",
        reference=(
            "No. Usernames cannot be changed after registration. "
            "You may change your display name from Profile Settings."
        ),
        response=(
            "Yes. You can change your username from Profile Settings. "
            "Your display name can also be changed there."
        ),
    ),
    EvaluationCase(
        id="unsupported_extra",
        question="How do I contact support?",
        reference=(
            "Email support@example.com. "
            "Support is available Monday to Friday."
        ),
        response=(
            "Email support@example.com. "
            "You can also call 1800-555-1234. "
            "Support is available Monday to Friday."
        ),
    ),
    EvaluationCase(
        id="missing_precondition",
        question="Can I withdraw money from my fixed deposit early?",
        reference=(
            "Yes, early withdrawal is allowed only after contacting the bank. "
            "An early withdrawal fee may apply."
        ),
        response=(
            "Yes. You can withdraw the money early without contacting the bank. "
            "A fee may apply."
        ),
    ),
    EvaluationCase(
        id="mixed_correct_and_wrong",
        question="What documents do I need to open the account?",
        reference=(
            "You need your passport. "
            "You need proof of address dated within the last three months. "
            "You must also provide your tax identification number."
        ),
        response=(
            "Bring your passport. "
            "Bring a recent proof of address. "
            "You do not need to provide a tax identification number. "
            "You should also bring your birth certificate."
        ),
    ),
]


def main() -> None:
    evaluator = Evaluator(config_path=CONFIG_PATH)
    results: dict[int, EvaluationResult] = {}

    with tqdm(
        total=len(CASES),
        desc="Evaluating",
        unit="case",
    ) as progress:
        for index, result in evaluator.evaluate_many_as_completed(
            CASES,
            max_concurrency=4,
        ):
            results[index] = result
            progress.update(1)

    for index in range(len(CASES)):
        result = results[index]
        print()
        print("=" * 80)
        print(f"CASE: {result.id}")
        print("=" * 80)
        print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
