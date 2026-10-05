"""The complete LangGraph workflow."""

from typing import TypedDict
from langgraph.graph import END, START, StateGraph

from .prompts import classify_prompt, extract_prompt, judge_prompt
from .schemas import EvaluationResult, ExtractedAnswer, JudgeResult, QuestionType
from .scoring import score


class State(TypedDict, total=False):
    id: str | int | None
    question: str
    reference: str
    response: str

    question_type: QuestionType
    reference_extraction: ExtractedAnswer
    response_extraction: ExtractedAnswer
    judge_result: JudgeResult

    result: EvaluationResult


BINARY_PREFIXES = (
    "can ", "could ", "is ", "are ", "am ", "was ", "were ",
    "do ", "does ", "did ", "may ", "might ", "must ",
    "should ", "would ", "will ", "has ", "have ", "had ",
)


def build_graph(model):
    
    classifier = model.with_structured_output(QuestionType)
    extractor = model.with_structured_output(ExtractedAnswer)
    judge = model.with_structured_output(JudgeResult)

    def classify_question(state: State):
        question = state["question"]
        prefix_hint = question.strip().lower().startswith(BINARY_PREFIXES)
        return {
            "question_type": classifier.invoke(
                classify_prompt(question, prefix_hint)
            )
        }

    def extract_reference(state: State):
        return {
            "reference_extraction": extractor.invoke(
                extract_prompt(state["reference"])
            )
        }

    def extract_response(state: State):
        return {
            "response_extraction": extractor.invoke(
                extract_prompt(state["response"])
            )
        }

    def judge_response(state: State):
        return {
            "judge_result": judge.invoke(
                judge_prompt(
                    state["question"],
                    state["reference_extraction"],
                    state["response_extraction"],
                )
            )
        }

    def calculate_score(state: State):
        return {
            "result": score(
                state.get("id"),
                state["question_type"],
                state["reference_extraction"],
                state["response_extraction"],
                state["judge_result"],
            )
        }

    graph = StateGraph(State)

    graph.add_node("classify_question", classify_question)
    graph.add_node("extract_reference", extract_reference)
    graph.add_node("extract_response", extract_response)
    graph.add_node("judge_response", judge_response)
    graph.add_node("calculate_score", calculate_score)

    graph.add_edge(START, "classify_question")
    graph.add_edge(START, "extract_reference")
    graph.add_edge(START, "extract_response")

    graph.add_edge(
        ["classify_question", "extract_reference", "extract_response"],
        "judge_response",
    )

    graph.add_edge("judge_response", "calculate_score")
    graph.add_edge("calculate_score", END)

    return graph.compile()
