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

## Reference

This project is motivated by the following paper:

> S Pradeepa, E Jomy, S Vimal, MM Hassan (2024). *HGATT_LR: transforming review text classification with hypergraphs attention layer and logistic regression*. Scientific Reports. https://www.nature.com/articles/s41598-024-70565-6

Paper author: Prof. Dongwann Kang, Seoul National University of Science and Technology (dongwann@seoultech.ac.kr)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
