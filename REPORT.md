# Week 3 Internship Report
## Implementing Sentiment Analysis and Named Entity Recognition

**Prepared by:** Shaik Tilawath Ahmed  
**Institution:** Geethanjali College of Engineering and Technology  
**Programme:** B.Tech  
**Project components:** Sentiment Analysis and Named Entity Recognition  
**Date:** 5 November 2026

## Executive Summary

This project implements two complementary Natural Language Processing (NLP) tasks: sentiment analysis and Named Entity Recognition (NER). The sentiment module uses VADER, a lexicon-and-rule-based analyzer, to classify short social-media messages as negative, neutral, or positive. It is evaluated on the sentiment test split of TweetEval. The NER module uses the pretrained spaCy English pipeline to identify people, locations, corporations, products, groups, and creative works, and is evaluated against the manually annotated WNUT 2017 test split.

The benchmarks are kept separate because they provide different ground-truth annotations: TweetEval supplies sentiment labels, whereas WNUT 2017 supplies token-level BIO entity annotations. Both contain informal user-generated text, making them suitable complementary benchmarks for a shared short-text workflow. The Python script downloads the public datasets, applies the selected methods, calculates metrics, and saves predictions and visualizations. No performance numbers are invented in this report; the actual values are generated when the script is run on the full benchmark test splits.

## 1. Introduction and Objectives

Text analytics often requires understanding both the opinion expressed in a message and the real-world entities mentioned in it. Sentiment analysis identifies the polarity of a text, while NER locates entity spans and assigns categories. Applying both tasks to short user-generated content is challenging because messages may contain informal spelling, abbreviations, emojis, hashtags, missing punctuation, and emerging names.

The objectives of this work are to:
- Develop a reproducible sentiment-scoring module using a clearly defined polarity method.
- Develop an NER module that extracts entity spans and maps them to a consistent set of categories.
- Evaluate sentiment and entity extraction with metrics appropriate to each task.
- Document preprocessing, model selection, execution steps, observed limitations, and future improvements.

## 2. Dataset Selection and Justification

### 2.1 TweetEval sentiment benchmark

The sentiment module uses the sentiment configuration of TweetEval, a benchmark that organizes Twitter classification tasks into fixed training, validation, and test splits. Its sentiment task uses three labels: negative, neutral, and positive. This is appropriate for a lexicon baseline because it provides human-assigned polarity labels for short social-media messages and allows a three-class evaluation rather than forcing neutral messages into a binary class [1].

### 2.2 WNUT 2017 emerging-entity benchmark

The NER module uses WNUT 2017, a benchmark designed for recognizing unusual and emerging named entities in noisy user-generated text. It defines six entity types: person, location, corporation, product, creative work, and group. The data includes content from social platforms and discussion sites, and its BIO annotations provide gold entity boundaries and categories for evaluation [2, 3].

A single benchmark with reliable gold labels for both general sentiment and general-purpose NER is not assumed here. Instead, the project uses two complementary public datasets from a similar informal-text setting. This is methodologically safer than generating artificial NER labels for a sentiment corpus or treating sentiment words as named entities.

| Benchmark | Task labels | Why it is suitable |
|---|---|---|
| TweetEval — sentiment | Negative, neutral, positive | Short social messages with gold sentiment labels; evaluates polarity scoring. |
| WNUT 2017 — NER | Person, location, corporation, product, creative-work, group | Noisy user-generated text with gold BIO tags; evaluates entity boundaries and types. |

## 3. Overall Workflow and Data Preparation

The two modules share a project structure but retain task-specific preprocessing so that important signals are not removed.

1. Download the public datasets with the Hugging Face Datasets library and select their official test splits.
2. Validate that the expected text, label, token, and BIO-tag fields are present.
3. Apply conservative, task-aware text handling; preserve punctuation, casing, emojis, contractions, hashtags, and token order whenever they carry information.
4. Run the sentiment scorer and the NER model independently.
5. Compare predictions with the corresponding gold labels and calculate task-specific metrics.
6. Export metrics, prediction samples, confusion matrix, entity-type summaries, and a combined JSON summary.

## 4. Mini-Project A: Sentiment Analysis

### 4.1 Method selection: VADER

VADER (Valence Aware Dictionary and sEntiment Reasoner) is selected as an interpretable lexicon-based baseline suited to social-media language. It combines a sentiment lexicon with rules that account for features such as capitalization, punctuation emphasis, degree modifiers, negation, and emoticons. Unlike a supervised classifier, VADER does not learn model parameters from the TweetEval training split; it uses its published lexicon and scoring rules directly. The project therefore evaluates a fixed baseline rather than claiming to train a new sentiment model.

### 4.2 Preprocessing and scoring

The original message is passed to VADER with minimal modification. Lowercasing or stripping punctuation can remove cues used by the analyzer, so the module does not aggressively normalize text. Blank or missing text is handled as an empty string, while the original message is retained in the prediction output for review.

VADER returns a compound score between -1 and +1. The standard decision thresholds used by the script map compound scores to three labels:

| Compound score | Predicted sentiment |
|---|---|
| score >= +0.05 | Positive |
| -0.05 < score < +0.05 | Neutral |
| score <= -0.05 | Negative |

The neutral interval uses strict inequalities, so the boundary values are assigned to negative or positive.

### 4.3 Evaluation strategy

The predicted class is compared with the human-assigned TweetEval sentiment label on the test split. Accuracy measures the total proportion of correct labels. Macro precision, recall, and F1 give equal weight to each sentiment class; weighted-F1 reflects class support. A confusion matrix shows which classes are most often confused, particularly whether neutral messages are incorrectly classified as positive or negative.

## 5. Mini-Project B: Named Entity Recognition

### 5.1 Model selection: spaCy English pipeline

The NER module uses spaCy's pretrained `en_core_web_sm` pipeline, which includes a statistical entity-recognition component. The model is already trained and is used for inference; the Week 3 baseline does not fine-tune it on WNUT 2017. This keeps setup manageable while allowing a direct evaluation against gold annotations from a noisy, emerging-entity benchmark. Because the pretrained model was not specifically trained for the WNUT label scheme, the mapping and domain mismatch are treated as limitations, not hidden.

### 5.2 Preprocessing and entity mapping

WNUT 2017 supplies each example as a token sequence with BIO tags. The script reconstructs the text by joining the original tokens with single spaces and calculates character offsets for the gold entities. The input is not lowercased and punctuation is not removed, because NER depends on exact surface forms, capitalization, token boundaries, and offsets.

The mapping used for scoring is:

| spaCy label(s) | Mapped WNUT category |
|---|---|
| PERSON | person |
| ORG | corporation |
| GPE, LOC, FAC | location |
| PRODUCT | product |
| WORK_OF_ART | creative-work |
| NORP | group |

Only the listed mappings are scored. Other spaCy labels are not converted into WNUT categories. A predicted entity is counted as correct only when its start character, end character, and mapped category exactly match a gold entity. This strict span-level method penalizes both incorrect boundaries and incorrect entity types.

### 5.3 Evaluation strategy

Entity-level precision is the share of predicted spans that exactly match a gold span; recall is the share of gold spans found by the model; micro-F1 is the harmonic mean of the aggregate precision and recall. The script also projects predicted spans back onto the original WNUT tokens and reports token-level accuracy. Per-category precision, recall, and F1 help identify which entity classes are difficult for the pretrained model.

## 6. Implementation and Reproducibility

The project is implemented in Python. The main script is `week3_nlp_pipeline.py`. It downloads TweetEval and WNUT 2017, runs VADER and spaCy, computes metrics, and saves output files. Dataset retrieval requires an internet connection. The spaCy language model must be installed once before the NER module is run.

### Environment setup

```bash
python -m venv .venv
# Windows: .venv\\Scripts\\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### Run both mini-projects

```bash
python week3_nlp_pipeline.py --out_dir outputs
```

Optional quick checks can limit the number of samples using `--max_sentiment_samples` and `--max_ner_samples`. Metrics from a limited run are for debugging only and must not be reported as full test-set benchmark results.

### Generated outputs

| Output file | Contents |
|---|---|
| `sentiment_metrics.json` | Sentiment accuracy, macro/weighted metrics, class report, confusion matrix. |
| `sentiment_predictions.csv` | Original text, gold sentiment, VADER prediction, compound score. |
| `sentiment_confusion_matrix.png` | Three-class sentiment confusion matrix. |
| `ner_metrics.json` | Token accuracy, exact entity precision/recall/F1, and per-type metrics. |
| `ner_entity_type_metrics.csv` | Gold support, predicted support, precision, recall, and F1 by entity type. |
| `ner_prediction_examples.csv` | Example texts with gold and predicted entities for qualitative inspection. |
| `ner_token_classification_report.json` | Token-level classification metrics. |
| `combined_summary.json` | Combined run metadata and headline metrics for both tasks. |

## 7. Results, Interpretation, and Discussion

The executable workflow calculates benchmark scores at runtime. The full test splits were not executed while preparing this report, so numerical accuracy or F1 values are intentionally not inserted here. This avoids presenting unmeasured numbers as experimental findings. After running the script, the generated JSON files contain the measured values, and the CSV files allow individual predictions and entity types to be inspected.

For sentiment analysis, a high neutral-to-positive or neutral-to-negative confusion count would indicate that a fixed lexicon threshold is not fully capturing context. Sarcasm, mixed opinions, domain-specific vocabulary, and negation scope are common sources of error. For NER, missed rare names, hashtags, creative works, new products, and informal spellings are likely failure patterns. The comparison should be interpreted as a pretrained general-English baseline tested on noisy social text, not as a model trained specifically for the WNUT domain.

The two modules use task-specific evaluation measures:

| Task | Primary evaluation | Runtime result |
|---|---|---|
| Sentiment analysis | Accuracy and macro-F1 across negative, neutral, positive | Generated in `sentiment_metrics.json` |
| NER | Exact typed-entity micro-F1; token-level accuracy as a secondary measure | Generated in `ner_metrics.json` |

## 8. Challenges and Mitigation

| Challenge | Impact | Mitigation / next step |
|---|---|---|
| Sarcasm and context | Literal positive words can occur in negative messages. | Inspect errors; compare with a fine-tuned transformer sentiment classifier. |
| Negation and emphasis | Polarity can change with word order, capitalization, or punctuation. | Preserve original text; test domain-specific thresholds only on validation data. |
| Emerging / rare entities | New names and unusual spellings may not be known to a general model. | Fine-tune a token-classification model on WNUT train/validation data. |
| Different label inventories | spaCy labels do not exactly match WNUT categories. | Use an explicit documented mapping and report unsupported labels. |
| Domain mismatch | A model trained on formal text may not transfer to social posts. | Use in-domain validation and compare with a WNUT-trained model. |
| Benchmark separation | The two corpora do not provide both kinds of gold labels. | Keep separate task datasets; do not infer missing annotations. |

## 9. Potential Improvements

- Train and compare supervised sentiment models such as TF-IDF with Logistic Regression or LinearSVC using the TweetEval training split, and tune parameters using validation only.
- Fine-tune a transformer token-classification model such as DistilBERT on WNUT 2017, preserving BIO alignment and evaluating once on the held-out test split.
- Add error analysis for sarcasm, emoji-heavy texts, hashtags, abbreviations, rare entity types, and boundary disagreements.
- Report confidence intervals or repeated-seed results to better understand variance; keep test data out of model selection.
- Evaluate on another domain only after documenting its label policy, license, and distribution differences.

## 10. Conclusion

This Week 3 project demonstrates two complementary NLP capabilities within a single reproducible framework. VADER provides interpretable lexicon-based sentiment scoring for negative, neutral, and positive messages, while spaCy provides pretrained statistical NER for recognizing typed entity spans. TweetEval and WNUT 2017 provide task-appropriate gold labels, and the workflow evaluates each module with metrics suited to its output. The main limitations are lexicon-based sentiment handling of context and the pretrained NER model's mismatch with emerging social-media entities. Future work can address these limitations through supervised sentiment training, WNUT-specific NER fine-tuning, and structured error analysis.

## References

1. Cardiff NLP. TweetEval benchmark repository and sentiment task description. https://github.com/cardiffnlp/tweeteval
2. Derczynski, L., Nichols, E., van Erp, M., and Limsopatham, N. (2017). Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition. *Proceedings of the 3rd Workshop on Noisy User-generated Text*, pp. 140–147. https://aclanthology.org/W17-4418/
3. Hugging Face Datasets. WNUT 2017 dataset card (`leondz/wnut_17`). https://huggingface.co/datasets/leondz/wnut_17
4. Hutto, C. J. and Gilbert, E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media Text. *Proceedings of ICWSM*. https://ojs.aaai.org/index.php/ICWSM/article/view/14550
