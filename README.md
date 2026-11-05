# Week 3: Sentiment Analysis and Named Entity Recognition

This repository contains two complementary NLP mini-projects:

1. **Sentiment analysis:** VADER lexicon/rule-based scoring on the three-class TweetEval sentiment test set (negative, neutral, positive).
2. **Named Entity Recognition (NER):** pretrained spaCy `en_core_web_sm` predictions evaluated against WNUT 2017 gold entity spans.

The tasks use separate, public benchmark datasets because TweetEval contains gold sentiment labels while WNUT 2017 contains gold BIO entity annotations. Both are based on informal social/user-generated text. The project does not fabricate missing NER annotations for the sentiment corpus.

## Files

- `Week3_Sentiment_Analysis_and_NER_Report.docx` — full internship report with methodology, preprocessing, algorithms, evaluation strategy, discussion, limitations, and improvements.
- `week3_nlp_pipeline.py` — executable sentiment and NER evaluation pipeline.
- `requirements.txt` — Python dependencies.
- `submission_description.txt` — report description for the internship portal.

## Setup

Python 3.9 or newer is recommended.

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run both tasks

```bash
python week3_nlp_pipeline.py --out_dir outputs
```

Optional quick smoke test (not a reportable benchmark result):

```bash
python week3_nlp_pipeline.py --max_sentiment_samples 100 --max_ner_samples 100 --out_dir smoke_outputs
```

## Evaluation

- **Sentiment:** accuracy, macro precision/recall/F1, weighted-F1, classification report, and confusion matrix. VADER compound scores are mapped to negative (<= -0.05), neutral (-0.05 to 0.05), and positive (>= 0.05).
- **NER:** token accuracy plus exact typed-entity span precision/recall/micro-F1. A true-positive span must match start character, end character, and mapped entity class.

## Outputs

The script produces sentiment metrics and predictions, a sentiment confusion matrix, NER metrics, per-entity-type metrics, prediction examples, a token classification report, and a combined summary in the output directory.

## Important interpretation note

Performance numbers are calculated when the script runs on the complete test splits. No benchmark scores are pre-filled in the report because they must be measured in the execution environment. VADER is a lexicon-based baseline, and spaCy's English small pipeline is pretrained on a different corpus; both limitations are documented in the report.
