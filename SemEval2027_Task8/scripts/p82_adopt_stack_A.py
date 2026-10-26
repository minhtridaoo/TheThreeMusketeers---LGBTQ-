"""P8.2a – Chuyển recipe A sang stacking (stack_A) theo quyết định ngày 26/10. Chỉ CPU, không train.

    C:\\Python314\\python.exe scripts/p82_adopt_stack_A.py

- Sao lưu recipe.json, selected_A.csv, all_results.csv, validated_results.csv vào outputs/audit/<ngày>-p82/before/.
- Tính lại OOF stack_A (cross-fit theo fold, cấu hình stacker cố định) từ các ứng viên hợp lệ của P3;
  phải khớp kết quả P8 (0,7587) – nếu lệch thì dừng.
- Ghi recipe A mới (strategy "stacking (logistic regression)"), giữ recipe A cũ trong trường previous_recipe.
- Master: dòng recipe A cũ → superseded_p8; thêm dòng recipe A mới.
"""
import hashlib
import json
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe import stacking  # noqa: E402
from sqe.data import TASKS  # noqa: E402
from sqe.metrics import fold_scores  # noqa: E402
from sqe.results import RESULTS_CSV, load_results, log_result, make_oof  # noqa: E402
from sqe.selection import SCOPE, candidates  # noqa: E402

EXPECTED = 0.7587          # macro-F1 stack_A ở notebook P8 lần 2
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    backup = ROOT / "outputs" / "audit" / f"{date.today().isoformat()}-p82" / "before"
    if backup.exists():
        raise SystemExit(f"Đã có {backup} – P8.2a có vẻ đã chạy. Kiểm tra trước khi chạy lại.")
    files = ["outputs/submissions/recipe.json", "outputs/selection/selected_A.csv",
             "outputs/results/all_results.csv", "outputs/results/validated_results.csv"]
    for rel in files:
        (backup / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, backup / rel)
    print("Đã sao lưu vào", backup)

    labels = TASKS["A"][1]
    names, frames, rejected = candidates("A")
    X = stacking.features([f[[f"p_{l}" for l in labels]].to_numpy() for f in frames])
    base = frames[0]
    P = stacking.crossfit(X, base.y_true.to_numpy(), base.fold.to_numpy(), labels)
    import pandas as pd
    from sqe.data import load_en
    df = load_en().set_index("StereoQueerEval_id").loc[base.StereoQueerEval_id].reset_index()
    oof = make_oof(df, "A", P)
    sc = fold_scores(oof, labels)
    score = float(sc.macro_f1.mean())
    print(f"stack_A tính lại: {score:.4f} ± {sc.macro_f1.std(ddof=0):.4f} | ứng viên: {names}")
    if abs(score - EXPECTED) > 5e-4:
        raise SystemExit(f"Không khớp P8 ({EXPECTED}) → dừng, không ghi gì thêm.")

    sel = ROOT / "outputs" / "selection"
    oof.to_csv(sel / "A_stacking.csv", index=False)
    oof.to_csv(sel / "selected_A.csv", index=False)

    recipe_path = ROOT / "outputs" / "submissions" / "recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    old = recipe["A"]
    configs = {}
    for n in names:
        if n.startswith("tfidf_"):
            continue
        for suffix in ("A", "AB"):
            p = ROOT / "outputs" / "runs" / f"{n}_{suffix}.json"
            if p.exists():
                configs[n] = {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(p),
                              "cfg": json.loads(p.read_text(encoding="utf-8"))["cfg"]}
                break
    recipe["A"] = {
        "strategy": "stacking (logistic regression)", "components": {n: 1.0 for n in names},
        "stacker": stacking.STACKER, "bias": [0.0, 0.0], "labels": labels,
        "development_macro_f1": score, "evaluation_scope": SCOPE,
        "selection_rule": {"rule": "P8 (26/10): nhận stacking nếu hơn recipe hiện tại ≥ 0,005 và thắng 5/5 fold",
                           "delta_vs_previous": round(score - old["development_macro_f1"], 4), "wins": "5/5",
                           "variants_tried": ["stack_A", "stack_A+B", "stack_B", "stack_B+A"]},
        "component_configs": configs, "selected_strategy_oof": "outputs/selection/selected_A.csv",
        "parameters_fit_scope": "Stacker được fit lại trên OOF mới khi dự đoán test (sqe.testtime)",
        "score_scope_note": "Development OOF cross-fit (stacker fit ngoài fold đang chấm); không phải điểm test.",
        "source_oof_sha256": {n: sha(ROOT / "outputs" / "oof" / f"{n}_A.csv") for n in names},
        "rejected_candidates": [r["model"] for r in rejected],
        "previous_recipe": {k: old[k] for k in ("strategy", "components", "bias", "development_macro_f1")},
    }
    recipe_path.write_text(json.dumps(recipe, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    master = load_results(include_invalid=True)
    cur = (master.notebook.eq("08") & master.experiment.eq("selected_recipe") & master.subtask.eq("A")
           & master.status.eq("completed_valid"))
    assert cur.sum() == 1, cur.sum()
    master.loc[cur, "status"] = "superseded_p8"
    master.loc[cur, "note"] = "Thay bằng stacking (P8.2, 26/10)"
    tmp = RESULTS_CSV.with_suffix(".tmp")
    master.to_csv(tmp, index=False)
    tmp.replace(RESULTS_CSV)
    log_result("08", "final: stacking (logistic regression)", "+".join(names), "A", sc,
               note="P8.2: stack_A, 10 ứng viên, LR C=1 balanced; development OOF cross-fit",
               experiment="selected_recipe", evaluation_scope=SCOPE, source_oof="outputs/selection/selected_A.csv")
    v = load_results()
    v.to_csv(ROOT / "outputs" / "results" / "validated_results.csv", index=False)
    print("Recipe A = stacking; master cập nhật. Recipe đang hiệu lực:")
    print(v[v.notebook.eq("08") & v.experiment.eq("selected_recipe")][["subtask", "model", "macro_f1"]].to_string())


if __name__ == "__main__":
    main()
