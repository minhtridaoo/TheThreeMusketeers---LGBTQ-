"""Ensure decision fitting cannot inspect held-out gold; preserve manual work."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe.selection import crossfit_strategies, select_strategy, STRATEGIES, export_review_table


def frame(seed):
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.ones(3), size=24)
    labels = np.array(["no", "yes_implicit", "yes_explicit"])
    d = pd.DataFrame({"StereoQueerEval_id": [f"id{i}" for i in range(24)],
                      "video_id": [f"v{i}" for i in range(24)], "fold": np.repeat(range(3), 8),
                      "y_true": np.tile(labels, 8), "y_pred": labels[p.argmax(1)]})
    d[[f"p_{l}" for l in labels]] = p
    return d


def test_holdout_gold_cannot_change_fitted_decisions_or_predictions():
    labels = ["no", "yes_implicit", "yes_explicit"]
    original = [frame(7), frame(9), frame(11)]
    changed = [d.copy() for d in original]
    mask = original[0].fold.eq(0)
    rotate = dict(zip(labels, labels[1:] + labels[:1]))
    for d in changed: d.loc[mask, "y_true"] = d.loc[mask, "y_true"].map(rotate)
    a, fit_a = crossfit_strategies(original, labels, ["a", "b", "c"], grid=[-.2, 0, .2])
    b, fit_b = crossfit_strategies(changed, labels, ["a", "b", "c"], grid=[-.2, 0, .2])
    assert [x for x in fit_a if x["score_fold"] == 0] == [x for x in fit_b if x["score_fold"] == 0]
    for strategy in STRATEGIES:
        pd.testing.assert_frame_equal(a[strategy].loc[mask].drop(columns="y_true"),
                                      b[strategy].loc[mask].drop(columns="y_true"))
    assert all(x["score_fold"] not in x["fit_folds"] for x in fit_a)


def test_gain_rule_keeps_simple_or_accepts_material_gain():
    table = pd.DataFrame({"strategy": STRATEGIES, "macro_f1": [.70, .701, .704, .703, .702, .70]})
    assert select_strategy(table)[0] == "best single + bias"
    table.loc[table.strategy.eq("greedy"), "macro_f1"] = .71
    assert select_strategy(table)[0] == "greedy"


def test_review_rerun_preserves_notes_and_rejects_source_change(tmp_path):
    path = tmp_path / "review.csv"
    table = frame(3).assign(kieu_kho="", source_model="original")
    export_review_table(table, path)
    old = pd.read_csv(path, keep_default_na=False); old.loc[0, "kieu_kho"] = "User note"
    old.to_csv(path, index=False, encoding="utf-8-sig")
    before = path.read_bytes()
    export_review_table(table, path)
    assert path.read_bytes() == before
    different = table.copy(); different.loc[0, "source_model"] = "different"
    with pytest.raises(ValueError, match="different source_model"):
        export_review_table(different, path)
    assert path.read_bytes() == before
