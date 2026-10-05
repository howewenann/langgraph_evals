from langgraph_evals.prompts import judge_prompt
from langgraph_evals.schemas import ExtractedAnswer, Unit


def test_judge_prompt_keeps_contradiction_reference_matches() -> None:
    reference = ExtractedAnswer(
        polarity="no",
        units=[Unit(kind="fact", text="Usernames cannot be changed.")],
    )
    response = ExtractedAnswer(
        polarity="yes",
        units=[Unit(kind="fact", text="Usernames can be changed.")],
    )

    prompt = judge_prompt("Can I change my username?", reference, response)

    assert "For every supported or contradicted candidate:" in prompt
    assert "For not_in_reference or unclear candidates:" in prompt


def test_judge_prompt_handles_empty_reference_without_negative_index_range() -> None:
    reference = ExtractedAnswer(polarity="unknown", units=[])
    response = ExtractedAnswer(polarity="unknown", units=[])

    prompt = judge_prompt("What is the answer?", reference, response)

    assert "There are no valid reference indexes" in prompt
    assert "0 through -1" not in prompt
