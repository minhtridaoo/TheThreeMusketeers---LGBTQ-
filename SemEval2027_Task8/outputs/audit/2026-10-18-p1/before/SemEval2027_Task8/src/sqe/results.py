"""Bảng kết quả chung (outputs/results/all_results.csv) và file OOF (outputs/oof/)."""
from datetime import date

import pandas as pd

from .data import OUT_DIR, TASKS
from .metrics import fold_scores, summary

RESULTS_CSV = OUT_DIR / "results" / "all_results.csv"
OOF_DIR = OUT_DIR / "oof"
KEY = ["notebook", "model", "input", "subtask"]


def log_result(notebook, model, input_name, subtask, scores, note=""):
    """Ghi (hoặc ghi đè nếu đã có cùng notebook/model/input/subtask) một dòng kết quả."""
    row = {"date": date.today().isoformat(), "notebook": str(notebook).zfill(2), "model": model,
           "input": input_name, "subtask": subtask, **summary(scores), "note": note}
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    old = load_results() if RESULTS_CSV.exists() else pd.DataFrame(columns=KEY)
    same = (old[KEY] == pd.Series({k: row[k] for k in KEY})).all(axis=1) if len(old) else []
    new = pd.concat([old[~same] if len(old) else old, pd.DataFrame([row])], ignore_index=True)
    new.to_csv(RESULTS_CSV, index=False, encoding="utf-8")
    return row


def load_results():
    r = pd.read_csv(RESULTS_CSV, dtype={"notebook": str})
    r["notebook"] = r["notebook"].str.zfill(2)      # "3" → "03"
    return r


def oof_path(name, subtask):
    return OOF_DIR / f"{name}_{subtask}.csv"


def save_oof(oof, name, subtask):
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    oof.to_csv(oof_path(name, subtask), index=False, encoding="utf-8")


def load_oof(name, subtask):
    return pd.read_csv(oof_path(name, subtask), dtype={"y_true": str, "y_pred": str})


def make_oof(df, subtask, proba):
    """Dựng bảng OOF chuẩn từ ma trận xác suất (n_mẫu × n_lớp, theo thứ tự TASKS)."""
    col, labels = TASKS[subtask]
    oof = df[["StereoQueerEval_id", "video_id", "fold"]].copy()
    oof["y_true"] = df[col].to_numpy()
    oof["y_pred"] = [labels[i] for i in proba.argmax(1)]
    for j, l in enumerate(labels):
        oof[f"p_{l}"] = proba[:, j]
    return oof


def scores_of(name, subtask):
    return fold_scores(load_oof(name, subtask), TASKS[subtask][1])
