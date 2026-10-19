"""Ghi file nộp Codabench và kiểm tra format (guideline mục 5).

Mỗi dòng: StereoQueerEval_id<TAB>nhãn. Không header (theo ví dụ trong guideline;
nếu Codabench yêu cầu header thì đổi HEADER = True), không kèm văn bản.
"""
from pathlib import Path

import pandas as pd

from .data import TASKS

HEADER = False


def write_submission(ids, preds, subtask, path):
    labels = TASKS[subtask][1]
    sub = pd.DataFrame({"StereoQueerEval_id": list(ids), "label": list(preds)})
    check_submission(sub, subtask, expected_ids=list(ids))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(path, sep="\t", index=False, header=HEADER, encoding="utf-8", lineterminator="\n")
    return path


def check_submission(sub, subtask, expected_ids):
    labels = set(TASKS[subtask][1])
    assert list(sub["StereoQueerEval_id"]) == list(expected_ids), "ID phải đủ và đúng thứ tự file test"
    assert sub["StereoQueerEval_id"].is_unique, "ID bị trùng"
    bad = set(sub["label"]) - labels
    assert not bad, f"Nhãn không hợp lệ cho subtask {subtask}: {bad}"


def read_submission(path, subtask):
    sub = pd.read_csv(path, sep="\t", header=0 if HEADER else None, dtype=str,
                      names=None if HEADER else ["StereoQueerEval_id", "label"])
    return sub
