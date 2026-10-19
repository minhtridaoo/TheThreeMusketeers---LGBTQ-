"""Train the frozen P3 recipe once per component; save and verify offline reload."""
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from .checkpoint import (environment, load_classical, load_transformer, save_classical,
                         save_transformer, sha256, verify_checkpoint, write_json)
from .data import ROOT, TASKS, load_en
from .decision import apply_bias
from .submission import write_submission, read_submission, check_submission
from .validation import validate_probabilities

MODES = ("smoke", "dryrun", "final")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def component_plan(recipe):
    """One shared AB checkpoint per multi-task component; others retain their task."""
    out = {}
    for task in "AB":
        r = recipe[task]
        if r["labels"] != TASKS[task][1]:
            raise ValueError("Recipe label order mismatch")
        weights = np.array(list(r["components"].values()), float)
        if not np.isfinite(weights).all() or (weights <= 0).any():
            raise ValueError("Invalid recipe component weight")
        for name in r["components"]:
            if name.startswith("tfidf_"):
                if name not in ("tfidf_comment", "tfidf_comment_title"):
                    raise ValueError(name)
                spec = {"name": name, "kind": "tfidf", "tasks": [task],
                        "cfg": {"input": name.removeprefix("tfidf_"), "model": "tfidf_lr"}}
            else:
                saved = r["component_configs"][name]
                path = ROOT / saved["path"]
                if sha256(path) != saved["sha256"]:
                    raise ValueError(f"Original run config changed: {path}")
                meta = json.loads(path.read_text(encoding="utf-8"))
                if meta["cfg"] != saved["cfg"] or task not in meta["tasks"]:
                    raise ValueError("Recipe/config mismatch")
                spec = {"name": name, "kind": "transformer", "tasks": meta["tasks"], "cfg": saved["cfg"]}
            if name in out and out[name] != spec:
                raise ValueError(f"Conflicting shared component: {name}")
            out[name] = spec
    return list(out.values())


def tfidf_texts(df, cfg):
    return (df.yt_comment if cfg["input"] == "comment" else df.yt_comment + "\n" + df.yt_title).tolist()


def data_for_mode(mode):
    if mode not in MODES:
        raise ValueError(mode)
    df = load_en()
    train, probe = df[df.fold != 0].copy(), df[df.fold == 0].copy()
    if mode == "smoke":
        groups = [g.sample(min(4, len(g)), random_state=42) for _, g in train.groupby(["stereotype", "hate_speech"])]
        first = pd.concat(groups)
        rest = train.drop(first.index).sample(64 - len(first), random_state=42)
        train = pd.concat([first, rest]).sort_index()
        probe = probe.head(16)
    elif mode == "final":
        train, probe = df.copy(), df.head(32).copy()
    if mode != "final":
        assert set(train.video_id).isdisjoint(probe.video_id)
    for task in "AB":
        assert set(train[TASKS[task][0]]) == set(TASKS[task][1])
    return train.reset_index(drop=True), probe.reset_index(drop=True)


def create_plan(mode):
    recipe_path = ROOT / "outputs/submissions/recipe.json"
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    specs = component_plan(recipe)
    train, probe = data_for_mode(mode)
    if mode == "smoke":
        specs = [{**s, "cfg": {**s["cfg"], "epochs": 1}} if s["kind"] == "transformer" else s for s in specs]
    sources = {str(p.relative_to(ROOT)).replace("\\", "/"): sha256(p)
               for p in sorted((ROOT / "src/sqe").glob("*.py"))}
    payload = {"mode": mode, "recipe": recipe, "recipe_sha256": sha256(recipe_path), "components": specs,
               "train_ids": train.StereoQueerEval_id.tolist(), "probe_ids": probe.StereoQueerEval_id.tolist(),
               "train_data_sha256": fingerprint(train.to_dict("records")),
               "probe_data_sha256": fingerprint(probe.to_dict("records")),
               "source_sha256": sources, "environment": environment(), "prediction_batch_size": 32,
               "purpose": "pipeline/reload check only; not independent performance evaluation"}
    payload["run_fingerprint"] = fingerprint(payload)
    return payload, train, probe


def _predict_saved(path, df, device, batch_size):
    import torch
    meta = verify_checkpoint(path)
    if meta["kind"] == "tfidf":
        model, meta = load_classical(path)
        task = meta["tasks"][0]
        raw = model.predict_proba(tfidf_texts(df, meta["cfg"]))
        return {task: validate_probabilities(raw[:, [list(model.classes_).index(l) for l in TASKS[task][1]]])}
    from .train import encode, predict_proba
    model, tok, meta = load_transformer(path, device)
    ids = encode(tok, df, meta["cfg"])
    result = predict_proba(model, ids, np.arange(len(df)), tok.pad_token_id, device, batch_size=batch_size)
    for values in result.values():
        validate_probabilities(values)
    del model, tok
    gc.collect()
    if device.type == "cuda": torch.cuda.empty_cache()
    return result


def train_suite(mode, output_dir, require_gpu=True):
    import torch
    from transformers import AutoTokenizer
    from .train import encode, fit, predict_proba
    from .classical import make_model
    if require_gpu and not torch.cuda.is_available():
        raise RuntimeError("GPU unavailable. Enable a Kaggle GPU before training P4.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plan, train, probe = create_plan(mode)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plan_path = output_dir / "plan.json"
    if plan_path.exists():
        old = json.loads(plan_path.read_text(encoding="utf-8"))
        if old["run_fingerprint"] != plan["run_fingerprint"]:
            raise ValueError("Existing P4 output belongs to different data/config/code/environment; use a new output directory.")
    write_json(plan_path, plan)
    text_cols = ["StereoQueerEval_id", "yt_title", "yt_description", "yt_comment"]
    probe[text_cols].to_csv(output_dir / "prediction_input.tsv", sep="\t", index=False, encoding="utf-8")
    probe[["StereoQueerEval_id", "stereotype", "hate_speech"]].to_csv(output_dir / "truth.csv", index=False)
    write_json(output_dir / "status.json", {"mode": mode, "status": "training", "run_fingerprint": plan["run_fingerprint"]})
    for i, spec in enumerate(plan["components"], 1):
        name, cfg, tasks = spec["name"], spec["cfg"], spec["tasks"]
        path = output_dir / "checkpoints" / name
        reference = output_dir / "reference" / f"{name}.npz"
        identity = fingerprint({"run": plan["run_fingerprint"], "component": spec})
        print(f"[{i}/{len(plan['components'])}] {name}: {mode}, train={len(train)}, probe={len(probe)}", flush=True)
        if path.exists():
            meta = verify_checkpoint(path, identity)
            if not reference.exists() or sha256(reference) != meta.get("reference_sha256"):
                raise ValueError(f"Missing/changed pre-save prediction reference for {name}; cannot certify reload")
            print("  resume: completed checkpoint + reference verified; skip training", flush=True)
            continue
        start = time.perf_counter()
        if spec["kind"] == "tfidf":
            task = tasks[0]
            model = make_model("tfidf_lr").fit(tfidf_texts(train, cfg), train[TASKS[task][0]].tolist())
            raw = model.predict_proba(tfidf_texts(probe, cfg))
            probs = {task: validate_probabilities(raw[:, [list(model.classes_).index(l) for l in TASKS[task][1]]])}
            reference.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(reference, **probs)
            save_classical(model, cfg, task, path, identity)
            del model
        else:
            tok = AutoTokenizer.from_pretrained(cfg["model"])
            ids = encode(tok, train, cfg)
            model, history = fit(train, np.arange(len(train)), cfg, tasks, ids, tok, device, log=lambda s: print(s, flush=True))
            probs = predict_proba(model, encode(tok, probe, cfg), np.arange(len(probe)), tok.pad_token_id, device,
                                  batch_size=plan["prediction_batch_size"])
            for values in probs.values(): validate_probabilities(values)
            reference.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(reference, **probs)
            save_transformer(model, tok, cfg, tasks, path, identity, history)
            del model, tok
        meta_path = path / "checkpoint.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        meta["reference_sha256"] = sha256(reference)
        meta["training_minutes"] = (time.perf_counter() - start) / 60
        write_json(meta_path, meta)
        print(f"  saved checkpoint; {meta['training_minutes']:.2f} minutes", flush=True)
        gc.collect()
        if device.type == "cuda": torch.cuda.empty_cache()
    write_json(output_dir / "status.json", {"mode": mode, "status": "trained_pending_fresh_reload", "run_fingerprint": plan["run_fingerprint"]})
    return plan


def verify_suite(output_dir):
    """Run in a fresh process with offline HF settings and an empty HF cache."""
    import torch
    from sklearn.metrics import f1_score
    output_dir = Path(output_dir)
    plan = json.loads((output_dir / "plan.json").read_text(encoding="utf-8"))
    write_json(output_dir / "status.json", {"mode": plan["mode"], "status": "verifying_offline", "run_fingerprint": plan["run_fingerprint"]})
    expected = {k: v for k, v in plan.items() if k != "run_fingerprint"}
    if fingerprint(expected) != plan["run_fingerprint"]:
        raise ValueError("Run plan fingerprint mismatch")
    for rel, digest in plan["source_sha256"].items():
        if sha256(ROOT / rel) != digest:
            raise ValueError(f"Reload source code changed: {rel}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if environment() != plan["environment"]:
        raise ValueError("Reload environment differs from reference; strict round-trip needs same environment/device")
    df = pd.read_csv(output_dir / "prediction_input.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert df.StereoQueerEval_id.tolist() == plan["probe_ids"]
    assert not set(["stereotype", "hate_speech", "target"]) & set(df.columns)
    before, after, checks = {}, {}, []
    for spec in plan["components"]:
        name = spec["name"]
        identity = fingerprint({"run": plan["run_fingerprint"], "component": spec})
        meta = verify_checkpoint(output_dir / "checkpoints" / name, identity)
        reference = output_dir / "reference" / f"{name}.npz"
        if sha256(reference) != meta["reference_sha256"]:
            raise ValueError("Pre-save reference hash mismatch")
        with np.load(reference, allow_pickle=False) as ref:
            before[name] = {t: ref[t].copy() for t in spec["tasks"]}
        after[name] = _predict_saved(output_dir / "checkpoints" / name, df, device, plan["prediction_batch_size"])
        for task in spec["tasks"]:
            a, b = before[name][task], after[name][task]
            validate_probabilities(a); validate_probabilities(b)
            same_labels = np.array_equal(a.argmax(1), b.argmax(1))
            close = bool(np.allclose(a, b, atol=1e-5, rtol=1e-4))
            checks.append({"component": name, "task": task, "labels_equal": same_labels,
                           "probabilities_close": close, "max_abs_diff": float(np.max(np.abs(a - b)))})
            if not same_labels or not close:
                write_json(output_dir / "reload_failure.json", checks)
                raise ValueError(f"Reload differs from pre-save prediction: {name}/{task}")
        print("offline reload OK:", name, flush=True)
    scored = []
    for task in "AB":
        r = plan["recipe"][task]
        def combine(source):
            P = sum(w * source[n][task] for n, w in r["components"].items()) / sum(r["components"].values())
            return apply_bias(P, np.array(r["bias"]))
        first, second = combine(before), combine(after)
        assert np.allclose(first, second, atol=1e-5, rtol=1e-4)
        pred = np.asarray(TASKS[task][1])[second.argmax(1)]
        assert np.array_equal(first.argmax(1), second.argmax(1))
        name = f"dryrun_fold0_{task}.tsv" if plan["mode"] == "dryrun" else f"{plan['mode']}_probe_{task}.tsv"
        path = write_submission(df.StereoQueerEval_id, pred, task, output_dir / "predictions" / name)
        back = read_submission(path, task)
        check_submission(back, task, df.StereoQueerEval_id.tolist())
        if plan["mode"] == "dryrun":
            truth = pd.read_csv(output_dir / "truth.csv").set_index("StereoQueerEval_id")
            score = f1_score(truth.loc[df.StereoQueerEval_id, TASKS[task][0]], pred,
                             labels=TASKS[task][1], average="macro", zero_division=0)
            scored.append({"task": task, "macro_f1_pipeline_check_only": float(score)})
    report = {"mode": plan["mode"], "status": "complete", "run_fingerprint": plan["run_fingerprint"],
              "fresh_process": True, "offline_reload": True, "n_train": len(plan["train_ids"]),
              "n_predictions": len(df), "checks": checks, "dryrun_scores": scored,
              "purpose": plan["purpose"], "checkpoint_count": len(plan["components"])}
    write_json(output_dir / "reload_report.json", report)
    write_json(output_dir / "status.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report
