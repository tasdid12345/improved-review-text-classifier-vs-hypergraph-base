"""Baseline (TF-IDF word-unigram LR) vs improved (class-balanced word 1-2g + char 2-5g LR, C tuned by 5-fold CV)
on the UCI Sentiment Labelled Sentences corpus. Writes results.json, word_weights.csv and figures/."""
import io, json, os, sys, time, zipfile
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")  # headless plotting
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import f1_score, accuracy_score, log_loss

SEED = 42
URL = "https://archive.ics.uci.edu/static/public/331/sentiment+labelled+sentences.zip"
MAX_ROWS = 200_000  # compute guard; the corpus is ~3,000 rows so this is normally inactive


def download(url, tries=5):
    """GET with timeout=60; retry with exponential backoff on network errors, HTTP 429 and 5xx."""
    for attempt in range(tries):
        try:
            r = requests.get(url, timeout=60)
            if r.status_code == 200:
                return r.content
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code} (not retryable)")
            reason = f"HTTP {r.status_code}"
        except requests.RequestException as e:
            reason = str(e)
        if attempt == tries - 1:
            raise RuntimeError(f"download failed after {tries} tries: {reason}")
        print(f"  download problem ({reason}); retrying in {2 ** attempt}s")
        time.sleep(2 ** attempt)


def load_data():
    """Read every *_labelled.txt (sentence<TAB>label) from the zip into one DataFrame."""
    z = zipfile.ZipFile(io.BytesIO(download(URL)))
    frames = []
    for name in z.namelist():
        if name.endswith("_labelled.txt"):
            raw = z.read(name).decode("utf-8", errors="replace")
            df = pd.read_csv(io.StringIO(raw), sep="\t", header=None, names=["text", "label"],
                             quoting=3, on_bad_lines="skip")  # quoting=3 -> no quote handling
            df["source"] = name.split("/")[-1].replace("_labelled.txt", "")
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["text", "label"])
    df = df[df["label"].isin([0, 1])].copy()
    df["label"] = df["label"].astype(int)
    df["text"] = df["text"].astype(str).str.strip()
    # drop exact duplicate sentences so the same text cannot appear in both train and test
    return df.drop_duplicates(subset=["text"]).reset_index(drop=True)


def evaluate(model, X, y):
    """Shared evaluation: main metric macro-F1, secondary accuracy and log loss."""
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {"macro_f1": f1_score(y, pred, average="macro"),
            "accuracy": accuracy_score(y, pred),
            "log_loss": log_loss(y, proba, labels=model.classes_)}


def main():
    df = load_data()
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=SEED)
    if len(df) < 100:
        sys.exit("Too few rows to produce meaningful metrics")
    print(f"Loaded {len(df)} unique labelled sentences from sources {sorted(df['source'].unique())}")
    X, y = df["text"].values, df["label"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=SEED)
    print(f"Split once: train={len(Xtr)}, test={len(Xte)}")

    # ---- BASELINE: TF-IDF word unigrams + logistic regression with default C (no tuning) ----
    print("Fitting baseline (TF-IDF unigram LR, default C)...")
    baseline = Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 1))),
                         ("lr", LogisticRegression(max_iter=1000, random_state=SEED))])
    baseline.fit(Xtr, ytr)

    # ---- IMPROVED: word 1-2g + char 2-5g TF-IDF, class-balanced LR, C chosen by 5-fold CV on train only ----
    print("Tuning improved model (word+char n-grams, balanced LR) with 5-fold CV on training data...")
    feats = FeatureUnion([("word", TfidfVectorizer(ngram_range=(1, 2))),
                          ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5)))])
    pipe = Pipeline([("feats", feats),
                     ("lr", LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, {"lr__C": [0.1, 0.3, 1, 3, 10]}, scoring="f1_macro", cv=cv, n_jobs=-1)
    search.fit(Xtr, ytr)
    improved = search.best_estimator_
    print(f"  best C={search.best_params_['lr__C']}, CV macro-F1={search.best_score_:.4f}")

    # ---- Single final test evaluation for both models ----
    b, i = evaluate(baseline, Xte, yte), evaluate(improved, Xte, yte)
    print(f"Test macro-F1: baseline={b['macro_f1']:.4f}, improved={i['macro_f1']:.4f}")
    if not all(np.isfinite(v) for v in list(b.values()) + list(i.values())):
        sys.exit("Non-finite metric; aborting")

    # ---- Inspectable word weights from the improved model (word block is the first FeatureUnion block) ----
    vocab = improved.named_steps["feats"].transformer_list[0][1].get_feature_names_out()
    coef = improved.named_steps["lr"].coef_.ravel()[:len(vocab)]
    pd.DataFrame({"word": vocab, "weight": coef}).sort_values("weight").to_csv("word_weights.csv", index=False)
    order = np.argsort(coef)
    print("Most negative words:", list(vocab[order[:10]]))
    print("Most positive words:", list(vocab[order[::-1][:10]]))

    # ---- Chart ----
    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["baseline", "improved"], [b["macro_f1"], i["macro_f1"]], color=["grey", "steelblue"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("test macro-F1")
    ax.set_title("Held-out macro-F1")
    fig.tight_layout()
    fig.savefig("figures/macro_f1.png", dpi=120)
    plt.close(fig)

    results = {
        "question": "Can a cross-validated, word-and-character n-gram logistic regression with class balancing "
                    "beat a standard TF-IDF logistic regression baseline at classifying the sentiment of short "
                    "product, movie and restaurant reviews, while keeping word-level weights that can be inspected?",
        "method": "Single stratified 75/25 split (seed 42). Baseline and improved models are fit on the same training "
                  "part; only the improved model's C is tuned, by 5-fold stratified CV on training data. Both are "
                  "evaluated once on the test split.",
        "dataset": {
            "name": "UCI Sentiment Labelled Sentences",
            "source_url": URL,
            "n_rows": int(len(df)),
            "n_train": int(len(Xtr)),
            "n_test": int(len(Xte)),
            "same_as_paper": False,
            "baseline_is_papers_method": False,
        },
        "metric_name": "macro-F1 on held-out test split",
        "higher_is_better": True,
        "baseline": "TF-IDF word unigrams + logistic regression with default C (no class weighting, no tuning).",
        "improved": "TF-IDF word 1-2-grams plus character 2-5-grams (char_wb), logistic regression with "
                    "class_weight='balanced', C selected by 5-fold CV grid search on the training split.",
        "metrics": {
            "baseline_score": b["macro_f1"],
            "improved_score": i["macro_f1"],
            "improvement": i["macro_f1"] - b["macro_f1"],
            "baseline_accuracy": b["accuracy"],
            "improved_accuracy": i["accuracy"],
            "baseline_log_loss": b["log_loss"],
            "improved_log_loss": i["log_loss"],
            "improved_cv_macro_f1": float(search.best_score_),
            "improved_best_C": float(search.best_params_["lr__C"]),
        },
        "notes": "Proxy dataset: short review sentences from Amazon, Yelp and IMDB, not the paper's own data; the "
                 "paper's hypergraph-attention model was not re-implemented, so the baseline is a standard TF-IDF "
                 "logistic regression instead. Single split on a small test set, so differences of a few points "
                 "are within sampling noise. Both models are deterministic given the split (lbfgs LR, no random "
                 "initialisation), so no multi-seed repeats were needed; the seed only affects the split and CV folds. "
                 "Exact duplicate sentences were removed before splitting. Word weights for the improved model are "
                 "written to word_weights.csv (char n-gram weights are not mapped to words). The test set was not used "
                 "for any tuning or selection.",
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Wrote results.json")


if __name__ == "__main__":
    main()
