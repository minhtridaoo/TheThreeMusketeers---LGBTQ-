"""CPU-only P3: compare six strategies, save recipe, provenance, and report tables."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe.data import TASKS, DATA_DIR, load_en
from sqe.decision import average
from sqe.metrics import fold_scores, summary
from sqe.results import load_results, log_result, RESULTS_CSV
from sqe.selection import (SCOPE, candidates, crossfit_strategies, fit_decision,
                           select_strategy, strategy_tables)
from sqe.validation import validate_oof


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run():
    dest = ROOT / "outputs/selection"
    dest.mkdir(parents=True, exist_ok=True)
    all_tables, all_folds, all_candidates, rejected, fits, recipe = [], [], [], [], {}, {}
    selected_frames = {}
    inputs = {"data/splits/folds_en.csv": sha(ROOT / "data/splits/folds_en.csv"),
              "../StereoQueerEval_SemEval-2027/StereoQueerEval_training/StereoQueerEval_EN_training.tsv":
                  sha(DATA_DIR / "StereoQueerEval_EN_training.tsv")}
    for source in [Path(__file__), *[ROOT / "src/sqe" / f for f in
                    ["selection.py", "decision.py", "validation.py", "metrics.py", "results.py", "classical.py"]]]:
        inputs[str(source.relative_to(ROOT)).replace("\\", "/")] = sha(source)
    canonical = load_en()
    for task in "AB":
        names, oofs, bad = candidates(task)
        rejected.extend(bad)
        labels = TASKS[task][1]
        print(f"{task}: {len(names)} valid candidates; {len(bad)} rejected", flush=True)
        for name, frame in zip(names, oofs):
            path = ROOT / f"outputs/oof/{name}_{task}.csv"
            inputs[str(path.relative_to(ROOT)).replace("\\", "/")] = sha(path)
            all_candidates.append({"subtask": task, "model": name, **summary(fold_scores(frame, labels))})
        strategies, fits[task] = crossfit_strategies(oofs, labels, names, progress=lambda s: print(task, s, flush=True))
        table, detail = strategy_tables(strategies, labels, task)
        all_tables.append(table); all_folds.append(detail)
        pick, reason = select_strategy(table)
        weights, bias = fit_decision(oofs, labels, np.ones(len(oofs[0]), dtype=bool), pick)
        selected = strategies[pick]
        validate_oof(selected, labels, canonical, TASKS[task][0])
        selected_frames[task] = selected
        comps = {name: float(w) for name, w in zip(names, weights) if w > 0}
        configs = {}
        for name in names:
            for suffix in (task, "AB"):
                path = ROOT / f"outputs/runs/{name}_{suffix}.json"
                if path.exists():
                    rel = str(path.relative_to(ROOT)).replace("\\", "/")
                    inputs[rel] = sha(path)
                    if name in comps:
                        configs[name] = {"path": rel, "sha256": sha(path), "cfg": json.loads(path.read_text(encoding="utf-8"))["cfg"]}
                    break
        recipe[task] = {"strategy": pick, "components": comps, "bias": bias.tolist(),
                        "labels": labels, "development_macro_f1": float(table.set_index("strategy").loc[pick, "macro_f1"]),
                        "evaluation_scope": SCOPE, "selection_rule": reason, "component_configs": configs,
                        "selected_strategy_oof": f"outputs/selection/selected_{task}.csv",
                        "parameters_fit_scope": "all existing development OOF after strategy selection",
                        "score_scope_note": "Score evaluates fold-specific decision fits; final all-OOF recipe has no independent test score."}
        for name, frame in strategies.items():
            slug = name.replace(" + ", "_").replace(" ", "_")
            frame.to_csv(dest / f"{task}_{slug}.csv", index=False)
        selected.to_csv(dest / f"selected_{task}.csv", index=False)
        print(task, pick, comps, bias.tolist(), recipe[task]["development_macro_f1"], flush=True)
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    for task in "AB":
        recipe[task]["source_fingerprint"] = fingerprint
        recipe[task]["source_oof_sha256"] = {n: inputs[f"outputs/oof/{n}_{task}.csv"] for n in recipe[task]["components"]}
    table = pd.concat(all_tables, ignore_index=True)
    detail = pd.concat(all_folds, ignore_index=True)
    table["selected"] = [r.strategy == recipe[r.subtask]["strategy"] for r in table.itertuples()]
    table.to_csv(ROOT / "outputs/results/p3_strategies.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(ROOT / "outputs/results/p3_folds.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(all_candidates).to_csv(ROOT / "outputs/results/p3_candidates.csv", index=False, encoding="utf-8-sig")
    dump(dest / "fold_decisions.json", fits)
    dump(dest / "manifest.json", {"evaluation_scope": SCOPE, "source_fingerprint": fingerprint,
         "inputs_sha256": inputs, "rejected_candidates": rejected,
         "limitations": ["Development OOF only, not nested CV of full training pipeline.",
                         "Strategy chosen using these development scores; no independent selection estimate.",
                         "No training, checkpoint saving, GPU run or official submission in P3."],
         "selected_oof_sha256": {t: sha(dest / f"selected_{t}.csv") for t in "AB"}})
    dump(ROOT / "outputs/submissions/recipe.json", recipe)
    for task in "AB":
        log_result("08", "final: " + recipe[task]["strategy"], "+".join(recipe[task]["components"]), task,
                   fold_scores(selected_frames[task], TASKS[task][1]), experiment="selected_recipe",
                   evaluation_scope=SCOPE, source_oof=f"outputs/selection/selected_{task}.csv",
                   run_date="2026-10-18", note="P3: jointly fit component selection and bias outside scored fold; development estimate")
    master = load_results(include_invalid=True)
    old = master.notebook.eq("08") & master.status.eq("pending_revalidation")
    master.loc[old, "status"] = "superseded_p3"
    fresh = master.notebook.eq("08") & master.evaluation_scope.eq(SCOPE) & master.experiment.eq("selected_recipe")
    for idx in master.index[fresh]:
        t = master.loc[idx, "subtask"]
        master.loc[idx, "source_oof_sha256"] = sha(dest / f"selected_{t}.csv")
        master.loc[idx, "n_eval"] = len(canonical)
        master.loc[idx, "run_fingerprint"] = fingerprint
    master.to_csv(RESULTS_CSV, index=False)
    res = load_results()
    res.to_csv(ROOT / "outputs/results/validated_results.csv", index=False)
    rows = []
    for task in "AB":
        r = res[res.subtask.eq(task)]
        groups = [(r.model.eq("majority"), "majority"),
                  (r.model.eq("tfidf_lr"), "TF-IDF"),
                  (r.notebook.eq("03") & r.model.eq("roberta-base"), "RoBERTa comment"),
                  (r.notebook.eq("04") & r.model.eq("roberta-base"), "best context tried"),
                  (r.notebook.eq("04") & ~r.model.eq("roberta-base"), "best encoder tried"),
                  (r.notebook.eq("05"), "best 05 variant tried"),
                  (r.notebook.eq("06"), "Qwen fold 0 only"),
                  (r.notebook.eq("08") & r.evaluation_scope.eq(SCOPE), "selected P3 strategy")]
        for mask, stage in groups:
            if not mask.any(): continue
            best = r[mask].sort_values("macro_f1", ascending=False, kind="stable").iloc[0]
            row = best.to_dict(); row["stage"] = stage
            rows.append(row)
    pd.DataFrame(rows).to_csv(ROOT / "outputs/results/summary_table.csv", index=False, encoding="utf-8-sig")
    print(table[["subtask", "strategy", "macro_f1", "macro_f1_std", "selected"]].to_string(index=False), flush=True)
    print("Saved recipe, source hashes, fold decisions, selected OOF and summary. No training executed.", flush=True)
    return recipe


if __name__ == "__main__":
    run()
