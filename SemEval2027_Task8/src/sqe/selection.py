"""Select decision strategies using existing OOF; no training or GPU imports.

For each held-out fold, fit both component selection and bias only on the other
folds. Base OOF training and strategy selection are not independent outer CV.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from .decision import GRID, apply_bias, average, greedy_select, tune_bias
from .metrics import fold_scores, summary
from .results import OOF_DIR, load_oof
from .validation import InvalidOOFError, validate_aligned, validate_oof

BASES = ("best single", "top-3 mean", "greedy")
STRATEGIES = tuple(s for base in BASES for s in (base, base + " + bias"))
SCOPE = "development_oof_decision_crossfit"
MIN_ENSEMBLE_GAIN = 0.005


def candidates(task):
    names, frames, rejected = [], [], []
    for path in sorted(OOF_DIR.glob(f"*_{task}.csv")):
        name = path.stem[:-2]
        if name.startswith(("random_", "majority_", "tfidf_lr_")) or name.endswith("_bias"):
            continue
        try:
            frame = load_oof(name, task)
        except InvalidOOFError as exc:
            rejected.append({"subtask": task, "model": name, "reason": str(exc)})
            continue
        names.append(name)
        frames.append(frame)
    if not frames:
        raise InvalidOOFError(f"No valid candidates for {task}")
    return names, frames, rejected


def choose_weights(oofs, labels, rows, base, max_steps=10):
    validate_aligned(oofs, labels)
    rows = np.asarray(rows)
    if rows.dtype != bool or rows.shape != (len(oofs[0]),) or not rows.any():
        raise InvalidOOFError("Selection rows must be a nonempty boolean mask")
    if base == "greedy":
        return greedy_select(oofs, labels, rows=rows, max_steps=max_steps).astype(float)
    if base not in BASES:
        raise ValueError(base)
    scores = [f1_score(o.loc[rows, "y_true"], o.loc[rows, "y_pred"],
                       labels=labels, average="macro", zero_division=0) for o in oofs]
    k = 1 if base == "best single" else min(3, len(oofs))
    order = np.argsort(-np.asarray(scores), kind="stable")
    weights = np.zeros(len(oofs))
    weights[order[:k]] = 1
    return weights


def fit_decision(oofs, labels, rows, strategy, grid=GRID):
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    base = strategy.removesuffix(" + bias")
    weights = choose_weights(oofs, labels, rows, base)
    combined = average(oofs, labels, weights)
    bias = np.zeros(len(labels))
    if strategy.endswith(" + bias"):
        P = combined.loc[rows, [f"p_{l}" for l in labels]].to_numpy()
        y = combined.loc[rows, "y_true"].map({l: i for i, l in enumerate(labels)}).to_numpy()
        bias = tune_bias(P, y, grid=grid)
    return weights, bias


def crossfit_strategies(oofs, labels, names, grid=GRID, progress=None):
    validate_aligned(oofs, labels)
    if len(names) != len(oofs) or len(set(names)) != len(names):
        raise ValueError("Candidate names must be unique and match frames")
    folds = sorted(oofs[0].fold.unique())
    if len(folds) < 2:
        raise ValueError("At least two folds are required")
    pcols = [f"p_{l}" for l in labels]
    out = {s: oofs[0].copy() for s in STRATEGIES}
    records = []
    for fold in folds:
        fit = (oofs[0].fold != fold).to_numpy()
        score = ~fit
        for base in BASES:
            weights = choose_weights(oofs, labels, fit, base)
            combined = average(oofs, labels, weights)
            y = combined.loc[fit, "y_true"].map({l: i for i, l in enumerate(labels)}).to_numpy()
            bias = tune_bias(combined.loc[fit, pcols].to_numpy(), y, grid=grid)
            for strategy, b in [(base, np.zeros(len(labels))), (base + " + bias", bias)]:
                out[strategy].loc[score, pcols] = apply_bias(combined.loc[score, pcols].to_numpy(), b)
                records.append({"strategy": strategy, "score_fold": int(fold),
                                "fit_folds": [int(k) for k in folds if k != fold],
                                "n_fit": int(fit.sum()), "n_score": int(score.sum()),
                                "components": {n: float(w) for n, w in zip(names, weights) if w > 0},
                                "bias": b.tolist()})
        if progress:
            progress(f"fold {fold}: selected weights and bias on other folds")
    for frame in out.values():
        frame["y_pred"] = np.asarray(labels)[frame[pcols].to_numpy().argmax(1)]
        validate_oof(frame, labels)
    return out, records


def strategy_tables(frames, labels, task):
    detail, rows = [], []
    baseline = fold_scores(frames["best single"], labels).set_index("fold").macro_f1
    for strategy, frame in frames.items():
        scores = fold_scores(frame, labels)
        delta = scores.set_index("fold").macro_f1 - baseline
        detail.append(scores.assign(subtask=task, strategy=strategy,
                                    delta_vs_best_single=scores["fold"].map(delta)))
        rows.append({"subtask": task, "strategy": strategy, **summary(scores),
                     "delta_vs_best_single": float(delta.mean()), "wins_vs_best_single": int((delta > 0).sum()),
                     "evaluation_scope": SCOPE})
    return pd.DataFrame(rows), pd.concat(detail, ignore_index=True)


def select_strategy(table):
    scores = table.set_index("strategy").macro_f1.reindex(STRATEGIES)
    if scores.isna().any() or not np.isfinite(scores).all():
        raise ValueError("All six strategy scores must be finite")
    simple = scores.loc[["best single", "best single + bias"]].idxmax()
    best = scores.idxmax()
    gain = float(scores[best] - scores[simple])
    pick = best if gain >= MIN_ENSEMBLE_GAIN else simple
    return pick, {"best_scoring_strategy": best, "best_simple_strategy": simple,
                  "gain_over_simple": gain, "minimum_ensemble_gain": MIN_ENSEMBLE_GAIN,
                  "rule": "Choose best ensemble only if gain over best single (with/without bias) >= 0.005"}


def export_review_table(frame, path):
    """Keep existing human annotations; refuse a different source at the same path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = pd.read_csv(path, keep_default_na=False)
        keys = [c for c in ["StereoQueerEval_id", "video_id", "yt_title", "yt_comment", "y_true", "y_pred", "error_type", "source_model"] if c in frame]
        if (not set(keys) <= set(old.columns) or
            not old.StereoQueerEval_id.is_unique or
            set(old.StereoQueerEval_id) != set(frame.StereoQueerEval_id)):
            raise ValueError(f"Existing review has a different source: {path}")
        a = old.set_index("StereoQueerEval_id").sort_index()
        b = frame.set_index("StereoQueerEval_id").sort_index()
        for c in keys:
            if c != "StereoQueerEval_id" and not a[c].astype(str).equals(b[c].astype(str)):
                raise ValueError(f"Existing review has different {c}: {path}")
        return old
    frame.to_csv(path, index=False, encoding="utf-8-sig")
    return frame
