"""Results registry and validated OOF storage."""
from datetime import date
import numpy as np
import pandas as pd
from .data import OUT_DIR, TASKS, load_en
from .metrics import fold_scores, summary
from .validation import InvalidOOFError, validate_oof, validate_probabilities

RESULTS_CSV = OUT_DIR / "results" / "all_results.csv"
OOF_DIR = OUT_DIR / "oof"
KEY = ["notebook", "model", "input", "subtask", "experiment", "evaluation_scope"]

def load_results(include_invalid=False):
    r = pd.read_csv(RESULTS_CSV, dtype={"notebook": str})
    r["notebook"] = r.notebook.str.zfill(2)
    for column, default in [("experiment", ""), ("evaluation_scope", "cv5_video"), ("status", "completed_valid")]:
        if column not in r:
            r[column] = default
        r[column] = r[column].fillna(default)
    return r if include_invalid else r[r.status.eq("completed_valid")].copy()

def log_result(notebook, model, input_name, subtask, scores, note="", *,
               experiment="", evaluation_scope="cv5_video", source_oof="", run_date=None):
    """Upsert one experiment; retain failed/pending records and all other experiments."""
    values = summary(scores)
    if not all(np.isfinite(v) for v in values.values()):
        raise ValueError("Cannot log nonfinite evaluation scores as a completed run.")
    row = {"date": run_date or date.today().isoformat(), "notebook": str(notebook).zfill(2),
           "model": model, "input": input_name, "subtask": subtask,
           "experiment": experiment, "evaluation_scope": evaluation_scope, "status": "completed_valid",
           "source_oof": source_oof, **values, "note": note}
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    old = load_results(include_invalid=True) if RESULTS_CSV.exists() else pd.DataFrame(columns=KEY)
    same = (old[KEY] == pd.Series({k: row[k] for k in KEY})).all(axis=1)
    new = pd.concat([old.loc[~same], pd.DataFrame([row])], ignore_index=True)
    tmp = RESULTS_CSV.with_suffix(".tmp")
    new.to_csv(tmp, index=False, encoding="utf-8")
    tmp.replace(RESULTS_CSV)
    return row

def oof_path(name, subtask):
    return OOF_DIR / f"{name}_{subtask}.csv"

def save_oof(oof, name, subtask):
    validate_oof(oof, TASKS[subtask][1], load_en(), TASKS[subtask][0])
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(oof_path(name, subtask), index=False, encoding="utf-8")

def load_oof(name, subtask):
    frame = pd.read_csv(oof_path(name, subtask), dtype={"y_true": str, "y_pred": str})
    try:
        return validate_oof(frame, TASKS[subtask][1], load_en(), TASKS[subtask][0])
    except InvalidOOFError as exc:
        raise InvalidOOFError(f"{name}/{subtask}: {exc}") from exc

def make_oof(df, subtask, proba):
    col, labels = TASKS[subtask]
    proba = validate_probabilities(proba)
    if proba.shape != (len(df), len(labels)):
        raise InvalidOOFError("Prediction shape does not match dataframe/task.")
    oof = df[["StereoQueerEval_id", "video_id", "fold"]].copy()
    oof["y_true"] = df[col].to_numpy()
    oof["y_pred"] = np.asarray(labels)[proba.argmax(1)]
    for j, label in enumerate(labels):
        oof[f"p_{label}"] = proba[:, j]
    return validate_oof(oof, labels)

def scores_of(name, subtask):
    return fold_scores(load_oof(name, subtask), TASKS[subtask][1])

