# LangGraph Evals

Reference-grounded evaluation for LLM answers, implemented as a small LangGraph workflow.

For each evaluation case, the graph:

1. classifies whether the question is fundamentally binary,
2. extracts atomic units from the reference and candidate answers in parallel,
3. judges every candidate unit against the authoritative reference,
4. calculates deterministic precision, recall, F1, missing reference content, and polarity agreement.

## Install

```bash
pip install -e .
```

Optional surfaces:

```bash
pip install -e ".[examples]"
pip install -e ".[api]"
pip install -e ".[ui]"
```

For quick source-tree testing, the runnable examples bootstrap `src/` themselves, so the project package does not need to be installed first (its third-party dependencies still need to exist in the environment):

```bash
python examples/basic.py
python examples/basic_many.py
```

## Configure a model

Edit `config/models.yml`:

```yaml
model:
  provider: openai
  name: google/gemma-4-e4b
  api_base: http://127.0.0.1:1234/v1
  api_key: lm-studio
  temperature: 0
```

The evaluator uses `langchain-anyllm`, so the graph can also be constructed with another compatible chat model by passing `model=` directly.

## Python API

```python
from langgraph_evals import EvaluationCase, Evaluator


evaluator = Evaluator()
result = evaluator.evaluate(
    EvaluationCase(
        question="Can I reset my password online?",
        reference="Yes. Use the Forgot Password link.",
        response="Yes. Select Forgot Password.",
    )
)

print(result.model_dump_json(indent=2))
```

`AutoTester` remains available as a backwards-compatible alias for `Evaluator`.

## Batch evaluation

```python
results = evaluator.evaluate_many(cases, max_concurrency=4)

for index, result in evaluator.evaluate_many_as_completed(
    cases,
    max_concurrency=4,
):
    print(index, result.f1)
```

## API

```bash
pip install -e ".[api]"
uvicorn apps.api:app --reload
```

## Streamlit

```bash
pip install -e ".[ui]"
streamlit run apps/streamlit_app.py
```

## Tests

```bash
pip install -e ".[dev]"
pytest
```

## Project layout

```text
langgraph-evals/
├── pyproject.toml
├── config/
│   └── models.yml
├── src/
│   └── langgraph_evals/
│       ├── __init__.py
│       ├── schemas.py
│       ├── prompts.py
│       ├── scoring.py
│       ├── graph.py
│       └── evaluator.py
├── apps/
│   ├── api.py
│   └── streamlit_app.py
├── examples/
│   ├── basic.py
│   └── basic_many.py
└── tests/
```

Core reading order:

`schemas.py -> prompts.py -> scoring.py -> graph.py -> evaluator.py`
