"""Baseline cổ điển: random, majority, TF-IDF + Logistic Regression (cấu hình như notebook 02)."""
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .data import SEED, TASKS
from .results import make_oof


def make_model(name):
    if name == "random":
        return DummyClassifier(strategy="uniform", random_state=SEED)
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "tfidf_lr":
        return Pipeline([
            ("features", FeatureUnion([
                ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("char", TfidfVectorizer(analyzer="char", ngram_range=(2, 5), min_df=2, sublinear_tf=True)),
            ])),
            ("classifier", LogisticRegression(class_weight="balanced", C=1.0, max_iter=2000, random_state=SEED)),
        ])
    raise ValueError(name)


def run_cv(df, model_name, texts, subtask):
    """5 fold theo cột df.fold. Trả về bảng OOF (có xác suất từng lớp)."""
    col, labels = TASKS[subtask]
    y = df[col].to_numpy()
    X = np.asarray(texts, dtype=object) if model_name == "tfidf_lr" else np.zeros((len(df), 1))
    proba = np.zeros((len(df), len(labels)))
    for k in sorted(df["fold"].unique()):
        tr, va = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
        m = make_model(model_name).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])
        order = [list(m.classes_).index(l) for l in labels]
        proba[va] = p[:, order]
    if model_name == "random":
        # DummyClassifier "uniform" cho xác suất bằng nhau; bốc nhãn ngẫu nhiên có seed.
        rng = np.random.default_rng(SEED)
        proba = proba + rng.random(proba.shape) * 1e-6
    return make_oof(df, subtask, proba)
