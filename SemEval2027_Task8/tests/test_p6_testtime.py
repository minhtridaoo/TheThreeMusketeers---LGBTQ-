"""P6.2/P8.2: dự đoán test bằng CV bagging + stacking (chỉ dùng TF-IDF để chạy nhanh trên CPU)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe import stacking  # noqa: E402
from sqe.data import TASKS, load_en  # noqa: E402
from sqe.testtime import component_specs, predict_recipe  # noqa: E402
from sqe.validation import validate_probabilities  # noqa: E402


def test_current_recipe_specs_separate_single_task_runs():
    recipe = json.loads((ROOT / "outputs/submissions/recipe.json").read_text(encoding="utf-8"))
    specs, use = component_specs(recipe)
    for t in "AB":
        for n in recipe[t]["components"]:
            assert t in specs[use[(n, t)]]["tasks"]
    if "deberta_p7_comment" in recipe["A"]["components"] and "deberta_p7_comment" in recipe["B"]["components"]:
        assert use[("deberta_p7_comment", "A")] != use[("deberta_p7_comment", "B")]


def test_stacking_crossfit_matches_fit_predict_shapes():
    rng = np.random.default_rng(0)
    P1, P2 = rng.dirichlet([1, 1, 1], 120), rng.dirichlet([1, 1, 1], 120)
    X = stacking.features([P1, P2])
    y = np.array(TASKS["B"][1])[rng.integers(0, 3, 120)]
    folds = np.arange(120) % 4
    P = stacking.crossfit(X, y, folds, TASKS["B"][1])
    validate_probabilities(P)
    Pt, _ = stacking.fit_predict(X, y, X[:7], TASKS["B"][1])
    assert Pt.shape == (7, 3)


def test_predict_recipe_tfidf_only_pseudo_test(tmp_path):
    df = load_en()
    train = df[df.fold != 0].reset_index(drop=True)
    test = df[df.fold == 0].head(40).reset_index(drop=True)[["StereoQueerEval_id", "yt_title", "yt_description", "yt_comment"]]
    recipe = {"A": {"strategy": "stacking (logistic regression)",
                    "components": {"tfidf_comment": 1.0, "tfidf_comment_title": 1.0}},
              "B": {"strategy": "top-3 mean + bias", "components": {"tfidf_comment": 1.0, "tfidf_comment_title": 1.0}}}
    P, report, variants = predict_recipe(recipe, train, test, cache_dir=tmp_path, log=lambda *a: None)
    for t in "AB":
        assert P[t].shape == (len(test), len(TASKS[t][1]))
        validate_probabilities(P[t])
        assert 0 < report["system_fresh_oof"][t]["macro_f1"] <= 1
    assert len(list(tmp_path.glob("*.npz"))) == 4       # 2 input × 2 subtask, có cache để chạy tiếp
    assert set(variants["A"]) == {"mean_components", "mean_then_stack"}
    assert set(variants["B"]) == {"mean_components", "mean_then_bias", "ensemble_no_bias"}
    P2, _, _ = predict_recipe(recipe, train, test, cache_dir=tmp_path, log=lambda *a: None)
    for t in "AB":
        assert np.allclose(P[t], P2[t])                 # đọc từ cache ra đúng kết quả cũ
