#!/usr/bin/env python3
"""Week 3: sentiment analysis + named entity recognition.

Sentiment: VADER lexicon/rule-based scoring evaluated on TweetEval sentiment.
NER: spaCy en_core_web_sm evaluated on WNUT 2017 gold entity spans.

Example:
  pip install -r requirements.txt
  python -m spacy download en_core_web_sm
  python week3_nlp_pipeline.py --out_dir outputs

The script downloads public benchmark datasets with Hugging Face Datasets.
"""
from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import spacy
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

SENTIMENT_LABELS = {0: "negative", 1: "neutral", 2: "positive"}
WNUT_LABELS_FALLBACK = [
    "O", "B-corporation", "I-corporation", "B-creative-work", "I-creative-work",
    "B-group", "I-group", "B-location", "I-location", "B-person", "I-person",
    "B-product", "I-product",
]
SPACY_TO_WNUT = {
    "ORG": "corporation",
    "WORK_OF_ART": "creative-work",
    "NORP": "group",
    "GPE": "location",
    "LOC": "location",
    "FAC": "location",
    "PERSON": "person",
    "PRODUCT": "product",
}


def predict_vader_label(text: str, analyzer: SentimentIntensityAnalyzer) -> tuple[str, float]:
    """Map VADER compound score to TweetEval's 3 sentiment classes."""
    compound = float(analyzer.polarity_scores(text or "")["compound"])
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return label, compound


def run_sentiment(out_dir: Path, max_samples: int | None = None) -> dict[str, Any]:
    """Evaluate VADER on the official TweetEval sentiment test split."""
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    split = ds["test"]
    if max_samples:
        split = split.select(range(min(max_samples, len(split))))

    analyzer = SentimentIntensityAnalyzer()
    y_true: list[str] = []
    y_pred: list[str] = []
    compounds: list[float] = []
    texts: list[str] = []
    for row in split:
        text = str(row["text"])
        gold = SENTIMENT_LABELS[int(row["label"])]
        prediction, compound = predict_vader_label(text, analyzer)
        texts.append(text)
        y_true.append(gold)
        y_pred.append(prediction)
        compounds.append(compound)

    labels = ["negative", "neutral", "positive"]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    metrics = {
        "dataset": "cardiffnlp/tweet_eval (sentiment configuration)",
        "split": "test",
        "examples_evaluated": len(y_true),
        "label_counts": dict(Counter(y_true)),
        "method": "VADER lexicon and rule-based polarity scoring",
        "decision_thresholds": {"positive": "compound >= 0.05", "negative": "compound <= -0.05", "neutral": "-0.05 < compound < 0.05"},
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix_labels": labels,
        "confusion_matrix": cm.tolist(),
    }
    pd.DataFrame({
        "text": texts,
        "gold_label": y_true,
        "predicted_label": y_pred,
        "vader_compound": compounds,
    }).to_csv(out_dir / "sentiment_predictions.csv", index=False)
    (out_dir / "sentiment_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    image = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), labels=["Negative", "Neutral", "Positive"])
    ax.set_yticks(range(3), labels=["Negative", "Neutral", "Positive"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("TweetEval sentiment: VADER")
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f"{value:,}", ha="center", va="center")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04)
    fig.tight_layout()
    fig.savefig(out_dir / "sentiment_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return metrics


def load_wnut_dataset() -> Any:
    """Load WNUT 2017 from a maintained HF dataset card, with mirror fallback."""
    errors = []
    for dataset_id in ("leondz/wnut_17", "tner/wnut2017"):
        try:
            return load_dataset(dataset_id)
        except Exception as exc:
            errors.append(f"{dataset_id}: {exc}")
    raise RuntimeError("Could not load WNUT 2017. " + " | ".join(errors))


def get_tag_names(split: Any, dataset: Any, tag_column: str) -> list[str]:
    """Read BIO tag names from the dataset schema, with a known fallback."""
    feature = split.features[tag_column]
    if hasattr(feature, "feature") and hasattr(feature.feature, "names"):
        return list(feature.feature.names)
    if hasattr(feature, "names"):
        return list(feature.names)
    return WNUT_LABELS_FALLBACK


def split_bio_tag(tag: str) -> tuple[str, str]:
    if tag == "O":
        return "O", "O"
    if "-" not in tag:
        return "O", "O"
    prefix, entity_type = tag.split("-", 1)
    return prefix.upper(), entity_type.lower()


def gold_spans(tokens: list[str], tag_ids: list[int], label_names: list[str]) -> tuple[str, set[tuple[int, int, str]]]:
    """Convert gold BIO token tags to exact (start_char,end_char,type) spans."""
    text = " ".join(tokens)
    offsets: list[tuple[int, int]] = []
    cursor = 0
    for token in tokens:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token) + 1

    spans: set[tuple[int, int, str]] = set()
    current_start: int | None = None
    current_end: int | None = None
    current_type: str | None = None

    def close_span() -> None:
        nonlocal current_start, current_end, current_type
        if current_start is not None and current_end is not None and current_type:
            spans.add((current_start, current_end, current_type))
        current_start = current_end = None
        current_type = None

    for index, tag_id in enumerate(tag_ids):
        tag_text = label_names[int(tag_id)] if int(tag_id) < len(label_names) else "O"
        prefix, entity_type = split_bio_tag(tag_text)
        start, end = offsets[index]
        if prefix == "O":
            close_span()
        elif prefix == "B":
            close_span()
            current_start, current_end, current_type = start, end, entity_type
        elif prefix == "I":
            if current_start is None or current_type != entity_type:
                close_span()
                current_start, current_end, current_type = start, end, entity_type
            else:
                current_end = end
        else:
            close_span()
    close_span()
    return text, spans


def predict_spacy_spans(text: str, nlp: Any) -> set[tuple[int, int, str]]:
    doc = nlp(text)
    spans: set[tuple[int, int, str]] = set()
    for ent in doc.ents:
        category = SPACY_TO_WNUT.get(ent.label_)
        if category and ent.text.strip():
            spans.add((int(ent.start_char), int(ent.end_char), category))
    return spans


def token_tags_from_spans(tokens: list[str], spans: set[tuple[int, int, str]]) -> list[str]:
    """Project character spans onto original WNUT tokens for token accuracy."""
    tags = ["O"] * len(tokens)
    cursor = 0
    offsets = []
    for token in tokens:
        offsets.append((cursor, cursor + len(token)))
        cursor += len(token) + 1
    for ent_start, ent_end, category in spans:
        for idx, (tok_start, tok_end) in enumerate(offsets):
            if tok_start < ent_end and tok_end > ent_start:
                prefix = "B" if tok_start <= ent_start < tok_end else "I"
                tags[idx] = f"{prefix}-{category}"
    return tags


def run_ner(out_dir: Path, model_name: str = "en_core_web_sm", max_samples: int | None = None) -> dict[str, Any]:
    """Evaluate spaCy NER against WNUT 2017 with exact typed entity spans."""
    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' is missing. Install it with: "
            f"python -m spacy download {model_name}"
        ) from exc

    ds = load_wnut_dataset()
    split_name = "test" if "test" in ds else list(ds.keys())[-1]
    split = ds[split_name]
    token_col = "tokens"
    tag_col = "ner_tags" if "ner_tags" in split.column_names else "tags"
    if tag_col not in split.column_names:
        raise ValueError(f"Cannot find NER tag column in columns: {split.column_names}")
    if max_samples:
        split = split.select(range(min(max_samples, len(split))))
    label_names = get_tag_names(split, ds, tag_col)

    gold_all: set[tuple[int, int, str, int]] = set()
    pred_all: set[tuple[int, int, str, int]] = set()
    y_gold_tokens: list[str] = []
    y_pred_tokens: list[str] = []
    rows: list[dict[str, str]] = []
    entity_gold_counts: Counter[str] = Counter()
    entity_pred_counts: Counter[str] = Counter()
    entity_correct_counts: Counter[str] = Counter()

    for row_index, row in enumerate(split):
        tokens = [str(x) for x in row[token_col]]
        ids = [int(x) for x in row[tag_col]]
        text, gold = gold_spans(tokens, ids, label_names)
        predicted = predict_spacy_spans(text, nlp)
        for start, end, category in gold:
            gold_all.add((row_index, start, end, category))
            entity_gold_counts[category] += 1
        for start, end, category in predicted:
            pred_all.add((row_index, start, end, category))
            entity_pred_counts[category] += 1
        for start, end, category in gold & predicted:
            entity_correct_counts[category] += 1

        gold_tags = [label_names[int(x)] if int(x) < len(label_names) else "O" for x in ids]
        predicted_tags = token_tags_from_spans(tokens, predicted)
        y_gold_tokens.extend(gold_tags)
        y_pred_tokens.extend(predicted_tags)
        if len(rows) < 200:
            rows.append({
                "text": text,
                "gold_entities": " | ".join(f"{text[s:e]} [{t}]" for s, e, t in sorted(gold)),
                "predicted_entities": " | ".join(f"{text[s:e]} [{t}]" for s, e, t in sorted(predicted)),
            })

    true_positives = len(gold_all & pred_all)
    false_positives = len(pred_all - gold_all)
    false_negatives = len(gold_all - pred_all)
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives else 0.0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives else 0.0
    entity_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    token_accuracy = float(accuracy_score(y_gold_tokens, y_pred_tokens))

    per_type = {}
    for category in sorted(set(entity_gold_counts) | set(entity_pred_counts)):
        tp = entity_correct_counts[category]
        fp = entity_pred_counts[category] - tp
        fn = entity_gold_counts[category] - tp
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f = 2 * p * r / (p + r) if p + r else 0.0
        per_type[category] = {"gold_support": int(entity_gold_counts[category]), "predicted_support": int(entity_pred_counts[category]), "precision": p, "recall": r, "f1": f}

    entity_report = {
        "dataset": "leondz/wnut_17 (fallback: tner/wnut2017)",
        "split": split_name,
        "examples_evaluated": len(split),
        "model": model_name,
        "model_type": "pretrained spaCy statistical NER pipeline; no task-specific fine-tuning in this baseline",
        "entity_span_matching": "exact start character, end character, and mapped WNUT entity type",
        "token_accuracy": token_accuracy,
        "entity_precision_micro": precision,
        "entity_recall_micro": recall,
        "entity_f1_micro": entity_f1,
        "true_positive_spans": true_positives,
        "false_positive_spans": false_positives,
        "false_negative_spans": false_negatives,
        "per_entity_type": per_type,
        "class_mapping": SPACY_TO_WNUT,
        "note": "spaCy label set is mapped to the six WNUT categories; unsupported spaCy labels are omitted. Exact boundary/type matching is required for a true positive.",
    }
    (out_dir / "ner_metrics.json").write_text(json.dumps(entity_report, indent=2), encoding="utf-8")
    pd.DataFrame(rows).to_csv(out_dir / "ner_prediction_examples.csv", index=False)
    pd.DataFrame([
        {"entity_type": category, **values} for category, values in per_type.items()
    ]).to_csv(out_dir / "ner_entity_type_metrics.csv", index=False)
    token_labels = sorted(set(y_gold_tokens) | set(y_pred_tokens))
    pd.DataFrame([classification_report(
        y_gold_tokens, y_pred_tokens, labels=token_labels, output_dict=True, zero_division=0
    )]).to_json(out_dir / "ner_token_classification_report.json", orient="records", indent=2)
    return entity_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 3 sentiment analysis and NER pipeline")
    parser.add_argument("--out_dir", default="outputs", help="Folder for metrics, predictions, and charts")
    parser.add_argument("--max_sentiment_samples", type=int, default=None, help="Optional quick-test limit")
    parser.add_argument("--max_ner_samples", type=int, default=None, help="Optional quick-test limit")
    parser.add_argument("--spacy_model", default="en_core_web_sm", help="Installed spaCy model name")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sentiment = run_sentiment(out_dir, args.max_sentiment_samples)
    ner = run_ner(out_dir, args.spacy_model, args.max_ner_samples)
    summary = {
        "project": "Week 3: Implementing Sentiment Analysis and Named Entity Recognition",
        "run_metadata": {
            "run_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
            "spacy_version": spacy.__version__,
            "pandas_version": pd.__version__,
        },
        "sentiment": {
            "dataset": sentiment["dataset"],
            "examples_evaluated": sentiment["examples_evaluated"],
            "accuracy": sentiment["accuracy"],
            "macro_f1": sentiment["f1_macro"],
        },
        "ner": {
            "dataset": ner["dataset"],
            "examples_evaluated": ner["examples_evaluated"],
            "token_accuracy": ner["token_accuracy"],
            "entity_f1_micro": ner["entity_f1_micro"],
        },
        "interpretation": "These results are calculated by running this script on the downloaded test splits. Do not substitute smoke-test values or assume results without an actual run.",
    }
    (out_dir / "combined_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nAll results saved in: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
