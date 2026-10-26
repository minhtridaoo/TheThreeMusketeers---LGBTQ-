"""P6.2 – Dự đoán tập test bằng các model fold (CV bagging), hỗ trợ recipe stacking (P8.2).

Với mỗi thành phần trong recipe:
    train k model (mỗi model bỏ ra một fold của tập train) → (a) OOF trên tập train, (b) xác suất test của TỪNG model fold.
Sau đó:
    A: recipe "stacking" → fit lại stacker (cấu hình cố định trong stacking.py) trên OOF MỚI.
    B: trung bình có trọng số các thành phần; nếu recipe có bias thì bias được fit lại trên OOF mới.
Nguyên tắc: thủ tục đóng băng, tham số (stacker/bias) học lại trên chính các model sinh ra xác suất test.

Từ 27/10 (sau diễn tập P8.2b): stacker và bias được áp cho TỪNG model fold rồi mới lấy trung bình
("per_fold"). Lý do: mỗi dòng OOF đến từ 1 model, còn trung bình nhiều model cho xác suất "mềm" hơn
(|logit| test/OOF ≈ 0,94 trong diễn tập) → áp stacker/bias lên xác suất đã trung bình bị lệch phân phối.
Cách cũ ("mean_then_…") vẫn được tính để so sánh trong report.

Mỗi thành phần được lưu đệm ra đĩa (npz) → chạy bị ngắt thì gọi lại sẽ tiếp tục.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import stacking
from .data import ROOT, TASKS
from .decision import apply_bias, tune_bias
from .metrics import fold_scores
from .results import make_oof
from .validation import validate_probabilities

CACHE_VERSION = 2   # v2: lưu xác suất test của từng fold (testfolds_*)


def component_specs(recipe):
    """Mỗi model cần train = một spec, khoá theo file run gốc: "<tên>@<subtask của run>".

    Ví dụ: deberta_p7_comment@A và deberta_p7_comment@B là HAI model riêng (cùng tên, khác run);
    mt_twhate_comment@AB là một model đa nhiệm dùng cho cả A và B.
    Trả về (specs {khoá: spec}, use {(tên, task): khoá}).
    """
    specs, use = {}, {}
    for task in "AB":
        for name in recipe[task]["components"]:
            if name.startswith("tfidf_"):
                key = f"{name}@{task}"
                spec = {"name": name, "kind": "tfidf", "tasks": [task], "cfg": {"input": name.removeprefix("tfidf_")}}
            else:
                runs = {p.stem[len(name) + 1:]: p for p in (ROOT / "outputs" / "runs").glob(f"{name}_*.json")
                        if p.stem[len(name) + 1:] in ("A", "B", "AB")}
                suffix = task if task in runs else "AB"
                meta = json.loads(runs[suffix].read_text(encoding="utf-8"))
                if task not in meta["tasks"]:
                    raise ValueError(f"Run {name}_{suffix} không có subtask {task}")
                key = f"{name}@{suffix}"
                spec = {"name": name, "kind": "transformer", "tasks": list(meta["tasks"]), "cfg": meta["cfg"]}
            if key in specs and specs[key] != spec:
                raise ValueError(f"Cấu hình mâu thuẫn cho {key}")
            specs[key] = spec
            use[(name, task)] = key
    return specs, use


def _texts(df, cfg):
    return (df.yt_comment if cfg["input"] == "comment" else df.yt_comment + "\n" + df.yt_title).tolist()


def _key(name, spec, df_train, df_test, smoke):
    payload = {"v": CACHE_VERSION, "name": name, "spec": spec, "smoke": smoke,
               "train": df_train[["StereoQueerEval_id", "fold"]].astype(str).values.tolist(),
               "test": df_test["StereoQueerEval_id"].tolist()}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def cv_component(name, spec, df_train, df_test, cache_dir=None, smoke=False, log=print):
    """Trả về ({task: OOF n_train×k}, {task: xác suất test từng fold, K×n_test×k})."""
    key = _key(name, spec, df_train, df_test, smoke)
    cache = Path(cache_dir) / f"{name.replace('@', '_')}_{key}.npz" if cache_dir else None
    if cache is not None and cache.exists():
        z = np.load(cache)
        log(f"  [cache] {name}")
        return ({t: z[f"oof_{t}"] for t in spec["tasks"]}, {t: z[f"testfolds_{t}"] for t in spec["tasks"]})
    folds = sorted(df_train.fold.unique())
    oof = {t: np.zeros((len(df_train), len(TASKS[t][1]))) for t in spec["tasks"]}
    test = {t: np.zeros((len(folds), len(df_test), len(TASKS[t][1]))) for t in spec["tasks"]}
    t0 = time.time()
    if spec["kind"] == "tfidf":
        from .classical import make_model
        X = np.asarray(_texts(df_train, spec["cfg"]), dtype=object)
        Xt = np.asarray(_texts(df_test, spec["cfg"]), dtype=object)
        for t in spec["tasks"]:
            col, labels = TASKS[t]
            y = df_train[col].to_numpy()
            for i, k in enumerate(folds):
                tr, va = (df_train.fold != k).to_numpy(), (df_train.fold == k).to_numpy()
                m = make_model("tfidf_lr").fit(X[tr], y[tr])
                order = [list(m.classes_).index(l) for l in labels]
                oof[t][va] = m.predict_proba(X[va])[:, order]
                test[t][i] = m.predict_proba(Xt)[:, order]
    else:
        import torch
        from transformers import AutoTokenizer
        from . import train as T
        cfg = {**T.DEFAULT, **spec["cfg"]}
        if smoke:
            cfg["epochs"] = 1
        both = pd.concat([df_train, df_test], ignore_index=True)
        for t in spec["tasks"]:   # tập test không có nhãn → điền tạm nhãn đầu (fit() chỉ đọc nhãn dòng train)
            col, labels = TASKS[t]
            both[col] = both[col].fillna(labels[0]).replace("", labels[0]) if col in both else labels[0]
        device = T._device()
        tok = AutoTokenizer.from_pretrained(cfg["model"])
        ids = T.encode(tok, both, cfg)
        test_idx = np.arange(len(df_train), len(both))
        for i, k in enumerate(folds):
            tr = np.where(df_train.fold.to_numpy() != k)[0]
            va = np.where(df_train.fold.to_numpy() == k)[0]
            if smoke:
                tr = tr[:64]
            log(f"  {name} fold {k}: train {len(tr)} | oof {len(va)} | test {len(test_idx)}")
            model, _ = T.fit(both, tr, cfg, tuple(spec["tasks"]), ids, tok, device, log=lambda *a: None)
            P_va = T.predict_proba(model, ids, va, tok.pad_token_id, device)
            P_te = T.predict_proba(model, ids, test_idx, tok.pad_token_id, device)
            for t in spec["tasks"]:
                oof[t][va] = P_va[t]
                test[t][i] = P_te[t]
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    for t in spec["tasks"]:
        validate_probabilities(oof[t])
        for i in range(len(folds)):
            validate_probabilities(test[t][i])
    log(f"  xong {name} ({(time.time() - t0) / 60:.1f} phút)")
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f"oof_{t}": oof[t] for t in spec["tasks"]},
                 **{f"testfolds_{t}": test[t] for t in spec["tasks"]})
    return oof, test


def _norm(P):
    return P / P.sum(-1, keepdims=True)


def predict_recipe(recipe, df_train, df_test, cache_dir=None, smoke=False, log=print):
    """Chạy toàn bộ recipe.

    Trả về (xác suất test {A, B}, báo cáo, biến thể để so sánh {task: {tên biến thể: xác suất}}).
    Biến thể: mean_components (trung bình đều, không tham số học), mean_then_stack / mean_then_bias
    (cách cũ trước 27/10), ensemble_no_bias.
    """
    specs, use = component_specs(recipe)
    log(f"{len(specs)} model cần train: {list(specs)}")
    res, minutes = {}, {}
    for name, spec in specs.items():
        t0 = time.time()
        res[name] = cv_component(name, spec, df_train, df_test, cache_dir, smoke, log)
        minutes[name] = round((time.time() - t0) / 60, 2)

    def oof_of(n, t):
        return res[use[(n, t)]][0][t]          # n_train × k

    def tf_of(n, t):
        return res[use[(n, t)]][1][t]          # K × n_test × k

    out, variants = {}, {}
    report = {"minutes": minutes, "components_fresh_oof": {}, "system_fresh_oof": {}, "apply_mode": "per_fold"}
    for task in "AB":
        col, labels = TASKS[task]
        r = recipe[task]
        names = list(r["components"])
        y = df_train[col].to_numpy()
        for n in names:
            sc = fold_scores(make_oof(df_train, task, oof_of(n, task)), labels)
            report["components_fresh_oof"][f"{n}/{task}"] = round(float(sc.macro_f1.mean()), 4)
        K = tf_of(names[0], task).shape[0]
        v = {"mean_components": _norm(sum(tf_of(n, task).mean(0) for n in names))}
        bias = None
        if r["strategy"].startswith("stacking"):
            X = stacking.features([oof_of(n, task) for n in names])
            P_oof = stacking.crossfit(X, y, df_train.fold.to_numpy(), labels)          # chẩn đoán
            model = stacking.make_stacker().fit(X, y)
            order = [list(model.classes_).index(l) for l in labels]
            per_fold = [model.predict_proba(stacking.features([tf_of(n, task)[i] for n in names]))[:, order]
                        for i in range(K)]
            out[task] = _norm(np.mean(per_fold, 0))                                    # stack từng fold → trung bình
            Xm = stacking.features([tf_of(n, task).mean(0) for n in names])
            v["mean_then_stack"] = model.predict_proba(Xm)[:, order]                   # cách cũ, chỉ để so
        else:
            w = np.array([r["components"][n] for n in names], float)
            avg_oof = sum(wi * oof_of(n, task) for wi, n in zip(w, names)) / w.sum()
            avg_tf = sum(wi * tf_of(n, task) for wi, n in zip(w, names)) / w.sum()   # K × n_test × k
            if r["strategy"].endswith("+ bias"):
                yi = pd.Series(y).map({l: i for i, l in enumerate(labels)}).to_numpy()
                bias = tune_bias(avg_oof, yi)                                          # fit lại trên OOF mới
                P_oof = np.zeros_like(avg_oof)                                         # chẩn đoán: bias học ngoài fold
                for k in sorted(df_train.fold.unique()):
                    m = (df_train.fold == k).to_numpy()
                    P_oof[m] = apply_bias(avg_oof[m], tune_bias(avg_oof[~m], yi[~m]))
            else:
                bias, P_oof = np.zeros(len(labels)), avg_oof
            out[task] = _norm(np.mean([apply_bias(avg_tf[i], bias) for i in range(K)], 0))   # bias từng fold
            v["mean_then_bias"] = apply_bias(_norm(avg_tf.mean(0)), bias)                     # cách cũ
            v["ensemble_no_bias"] = _norm(avg_tf.mean(0))
        validate_probabilities(out[task])
        variants[task] = v
        sc = fold_scores(make_oof(df_train, task, P_oof), labels)
        report["system_fresh_oof"][task] = {
            "strategy": r["strategy"], "macro_f1": round(float(sc.macro_f1.mean()), 4),
            "std": round(float(sc.macro_f1.std(ddof=0)), 4),
            "bias_refit": None if bias is None else np.round(bias, 3).tolist()}
    return out, report, variants
