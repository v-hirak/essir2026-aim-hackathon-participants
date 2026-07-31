# Level 3 evaluation

`run_eval.py` runs q7-q9 through the real `/query` API and compares the responses with
`gold.json`. Gold answers are used only by this offline evaluator, never by the app.

## Metrics

- **Page recall**: how many gold evidence pages were returned in `sources`.
- **BERTScore F1**: semantic similarity between the generated answer and gold answer,
  calculated with `roberta-base`. Higher is more similar.
- **Number recall**: how many numbers from the gold answer appeared in the generated answer.
  It is reported only when a question has numerical gold facts.
- **Latency**: query time reported by the app.

Page recall measures grounding. BERTScore measures answer similarity, not factual correctness.
Number recall measures numerical completeness.

## Run the baseline

Install the evaluation-only dependency in a separate environment once:

```bash
python -m pip install -r evaluation/requirements.txt
```

Start Ollama and the Docker services, then run with that environment's Python:

```bash
python evaluation/run_eval.py
```

The script runs q7-q9, prints a small table, and saves each generated answer and sources,
the matching gold answer and evidence, its metrics, and the mean metrics to:

```text
results/evaluation/baseline-level3.json
```

After implementing Level 3, run the same evaluation with a different name:

```bash
python evaluation/run_eval.py --run-name improved-level3
```

Compare the two JSON files for the ablation in `TECHNICAL_NOTE.md`. Keep the PDF, model,
embedding model, and retrieval settings unchanged between runs.
