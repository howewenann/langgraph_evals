"""Thin CSV/Streamlit route."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import pandas as pd
import streamlit as st
from langgraph_evals import EvaluationCase, Evaluator


@st.cache_resource
def get_evaluator() -> Evaluator:
    return Evaluator()


def cell_text(value) -> str:
    """Convert CSV cells to text without turning missing values into 'nan'."""

    return "" if pd.isna(value) else str(value)


st.title("LangGraph Evals")
uploaded = st.file_uploader("Upload CSV", type="csv")

if uploaded is not None:
    df = pd.read_csv(uploaded)
    columns = list(df.columns)

    if not columns:
        st.error("The CSV has no columns.")
        st.stop()

    question_col = st.selectbox("Question", columns)
    reference_col = st.selectbox("Reference", columns)
    response_col = st.selectbox("Response", columns)

    if st.button("Evaluate"):
        cases = [
            EvaluationCase(
                id=i,
                question=cell_text(row[question_col]),
                reference=cell_text(row[reference_col]),
                response=cell_text(row[response_col]),
            )
            for i, (_, row) in enumerate(df.iterrows())
        ]

        with st.spinner("Evaluating..."):
            results = get_evaluator().evaluate_many(cases)

        scores = pd.DataFrame(
            [
                {
                    "precision": r.precision,
                    "recall": r.recall,
                    "f1": r.f1,
                    "polarity_match": r.polarity_match,
                    "rationale": r.rationale,
                }
                for r in results
            ]
        )

        # evaluate_many preserves input order. Add score columns by position and
        # avoid overwriting same-named columns that may already exist in the CSV.
        output = df.reset_index(drop=True).copy()
        for column in scores.columns:
            output_column = column if column not in output else f"eval_{column}"
            output[output_column] = scores[column]

        st.dataframe(output)
        st.download_button(
            "Download results",
            output.to_csv(index=False),
            "results.csv",
            mime="text/csv",
        )
