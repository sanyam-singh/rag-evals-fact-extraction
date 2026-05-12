# RAG Evals — Fact Extraction Evaluation Framework

A semantic evaluation framework for measuring **fact extraction quality** in RAG (Retrieval-Augmented Generation) pipelines. Computes **Precision**, **Recall**, **F1**, **F2**, and **F3** scores over atomically-extracted facts, with LLM-assisted semantic matching and contradiction detection.

---

## Overview

Standard RAG evals measure surface-level similarity (BLEU, ROUGE, exact match). This framework goes deeper: it extracts **atomic facts** from model responses, matches them against a golden fact set using GPT-4o as a semantic judge, and computes structured metrics that reveal *how much* a model knows, *how accurate* it is, and *how relevant* its outputs are.

Built for agricultural knowledge RAG pipelines (Bihar, India), but the evaluation framework is **domain-agnostic** and reusable for any RAG or fact-extraction task.

---

## Metrics

All metrics are computed at the **sample level** and averaged across the dataset.

### Core Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Fact Recall** (IOGT) | `Matches / Gold Facts` | How many ground-truth facts did the model capture? |
| **Fact Precision** | `Matches / Predicted Facts` | How many predicted facts were correct? |
| **Relevance** | `(Matches + Relevant Unmatched) / Predicted Facts` | How many predicted facts were on-topic? |

### F-Score Family

Computed from Fact Precision (P) and Fact Recall (R):

```
F_beta = (1 + beta²) × (P × R) / (beta² × P + R)
```

| Score | Beta | Weights |
|---|---|---|
| **F1** | β = 1 | Equal weight to Precision and Recall |
| **F2** | β = 2 | Recall weighted 2× over Precision |
| **F3** | β = 3 | Recall weighted 3× over Precision |

Use **F1** for balanced evaluation, **F2/F3** when missing ground-truth facts is more costly than hallucinating extras (e.g., safety-critical domains, medical advice, agricultural guidance).

---

## How It Works

```
Model Response
      │
      ▼
┌─────────────────────┐
│  Fact Extractor LLM │  (GPT-4o / fine-tuned model)
│  → Atomic JSON facts│
└─────────────────────┘
      │
      ▼
┌──────────────────────────┐
│  EnhancedFactEvaluator   │
│  ┌──────────────────┐    │
│  │ Semantic Matching│ ── GPT-4o judge (confidence ≥ 0.7)
│  ├──────────────────┤    │
│  │ Contradiction    │ ── Detects conflicting claims
│  │ Detection        │    │
│  ├──────────────────┤    │
│  │ Relevance Scorer │ ── Scores unmatched predicted facts (1–10)
│  └──────────────────┘    │
└──────────────────────────┘
      │
      ▼
Precision / Recall / F1 / F2 / F3
```

### Fact Classification

Each predicted fact is classified into one of four buckets:

| Status | Meaning |
|---|---|
| `matched` | Semantically matched a gold fact (confidence ≥ 0.7) |
| `contradictory` | Directly contradicts a gold fact |
| `unmatched_relevant` | Did not match any gold fact, but is on-topic (score ≥ 6/10) |
| `unmatched_irrelevant` | Off-topic or hallucinatory |

---

## Fact Schema

Facts are extracted as structured JSON:

```json
{
  "facts": [
    {
      "fact": "Apply neem oil at 3ml per liter concentration for aphid control",
      "category": "pest_disease",
      "location_dependency": "universal",
      "bihar_relevance": "high",
      "confidence": 0.9
    }
  ]
}
```

**Categories**: `crop_variety`, `pest_disease`, `soil_management`, `irrigation`, `seasonal_practice`, `input_management`

---

## Models Evaluated

The notebook benchmarks multiple models on agricultural fact extraction:

| Model | Type |
|---|---|
| GPT-4.1-mini (fine-tuned) | Fine-tuned |
| GPT-4o (fine-tuned, FT1 & FT2) | Fine-tuned |
| GPT-4.1 (base) | Base |
| Gemini 3 Pro | Base |
| Claude Sonnet / Haiku / Opus | Base |
| Baseline (F0) | Non-fine-tuned reference |

Experiments are run across two evaluation sets (E1: human golden, E2: new test data) and multiple model checkpoints (M1–M4).

---

## Repository Structure

```
rag-evals-fact-extraction/
├── evaluator.py     # EnhancedFactEvaluatorProduction — semantic matching, contradiction detection, CSV export
├── pipeline.py      # Threaded / multiprocess / batched pipeline runners
├── inference.py     # Model inference with retry logic and parallel execution
├── metrics.py       # Precision, Recall, F1, F2, F3 computation
├── prompts.py       # Fact extraction system prompt + judge prompts
├── utils.py         # load_jsonl, ProgressTracker
├── main.py          # CLI entry point (infer / eval / metrics)
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Your Data

Your data should be a JSONL file where each line has a `messages` array with `user` and `assistant` turns. The assistant turn contains the ground-truth facts in JSON format.

```jsonl
{"messages": [{"role": "user", "content": "What fertilizer should I use for wheat in Patna?"}, {"role": "assistant", "content": "{\"facts\": [{\"fact\": \"Apply urea at 40 kg/ha for wheat\", \"category\": \"input_management\", ...}]}"}]}
```

### 3. Run Inference

Use any LLM (via OpenAI, Anthropic, or Google APIs) to generate `predicted_facts` for each question in your dataset.

### 4. Run Inference

```bash
python main.py infer \
  --data test.jsonl \
  --model ft:gpt-4.1-mini-2025-04-14:your-org:model-name:id \
  --output predictions.csv \
  --workers 10
```

### 5. Run Evaluation

```bash
python main.py eval \
  --data predictions.csv \
  --output detailed_results.csv \
  --workers 40
```

Or directly in Python:

```python
from evaluator import EnhancedFactEvaluatorProduction

evaluator = EnhancedFactEvaluatorProduction(openai_api_key="sk-...")
metrics = evaluator.evaluate_fact_sft_model_enhanced(
    predicted_facts=model_output,
    golden_facts=ground_truth,
    question="What fertilizer for wheat?",
)
```

### 6. Compute Metrics

```bash
python main.py metrics --data detailed_results.csv
```

Or in Python:

```python
from metrics import compute_metrics_from_file
compute_metrics_from_file("detailed_results.csv")
```

---

## Parallel Evaluation

For large datasets, use the built-in threaded pipeline:

```python
results = integrate_enhanced_evaluation_v2_threaded(
    df=dataframe,
    openai_api_key="sk-...",
    max_workers=40        # tune to your API rate limits
)
```

---

## RAG Use Case

This framework is designed as a **post-retrieval evaluation layer** for RAG systems:

1. **Retrieval quality**: High Recall (F2/F3) ensures the RAG retriever surfaces enough relevant context for the generator to produce complete answers.
2. **Generation quality**: High Precision ensures the generator is not hallucinating or mixing in irrelevant facts from retrieved chunks.
3. **Contradiction rate**: Surfaces cases where the generator contradicts known facts — a strong signal of retrieval-generation misalignment.
4. **Relevance rate**: Measures whether retrieved context is being usefully synthesized vs. ignored.

---

## Requirements

- Python ≥ 3.9
- OpenAI API key (for GPT-4o semantic judge)
- Optional: Anthropic / Google API keys (for evaluating Claude / Gemini models)

---

## Citation

If you use this evaluation framework, please cite:

```
Digital Green Foundation — Agricultural RAG Evaluation Framework
Fact-Extraction-Based Recall, Precision, F1/F2/F3 for RAG Pipelines
https://github.com/sanyam-singh/rag-evals-fact-extraction
```

---

## License

MIT License
