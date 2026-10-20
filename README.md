# Review Sentiment: Word+Char N-gram LR vs TF-IDF Baseline

Motivation: Hypergraph-based models such as those in "HGATT_LR: transforming review text classification with hypergraphs attention layer and logistic regression" (2024, https://www.nature.com/articles/s41598-024-70565-6) aim to improve review text classification, but they are complex and hard to inspect. A well-tuned linear model on sparse n-gram features is a cheap, interpretable point of comparison for the same task.

Question: Can a cross-validated, word-and-character n-gram logistic regression with class balancing beat a standard TF-IDF logistic regression baseline at classifying the sentiment of short product, movie and restaurant reviews, while keeping word-level weights that can be inspected?

Dataset: UCI Sentiment Labelled Sentences (about 3,000 labelled sentences from Amazon, Yelp and IMDB), downloaded from the UCI archive. This is a public stand-in for review text classification, not the paper's own dataset.

Baseline vs improved model:
- Baseline: TF-IDF word unigrams with logistic regression at default C.
- Improved: TF-IDF word 1-2-grams plus character 2-5-grams, class-balanced logistic regression, with C chosen by 5-fold cross-validation on the training split. Its word weights can be read directly from `word_weights.csv`.

Split: one stratified 75/25 train/test split with seed 42. All tuning uses cross-validation on the training part only; the test set is used once.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (plus `word_weights.csv` and `figures/macro_f1.png`).

## Results (latest run)

On the UCI Sentiment Labelled Sentences dataset, a class-balanced word and character n-gram logistic regression reached 81.2% macro-F1 on the held-out test split, versus 78.8% for a standard TF-IDF word-unigram logistic regression, a small gain of 2.4 points from one split.

| Metric | Value |
|---|---|
| baseline_score | 0.7882 |
| improved_score | 0.8123 |
| improvement | 0.02412 |
| baseline_accuracy | 0.7882 |
| improved_accuracy | 0.8123 |
| baseline_log_loss | 0.4917 |
| improved_log_loss | 0.3883 |
| improved_cv_macro_f1 | 0.8376 |
| improved_best_C | 10 |

Run on 2026-10-20. Full output: [results.json](results.json) · [run.log](run.log)

Notes: Proxy dataset: short review sentences from Amazon, Yelp and IMDB, not the paper's own data; the paper's hypergraph-attention model was not re-implemented, so the baseline is a standard TF-IDF logistic regression instead. Single split on a small test set, so differences of a few points are within sampling noise. Both models are deterministic given the split (lbfgs LR, no random initialisation), so no multi-seed repeats were needed; the seed only affects the split and CV folds. Exact duplicate sentences were removed before splitting. Word weights for the improved model are written to word_weights.csv (char n-gram weights are not mapped to words). The test set was not used for any tuning or selection.

## Reference

This project is motivated by the following paper:

> S Pradeepa, E Jomy, S Vimal, MM Hassan (2024). *HGATT_LR: transforming review text classification with hypergraphs attention layer and logistic regression*. Scientific Reports. https://www.nature.com/articles/s41598-024-70565-6

Paper author: Prof. Dongwann Kang, Seoul National University of Science and Technology (dongwann@seoultech.ac.kr)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
