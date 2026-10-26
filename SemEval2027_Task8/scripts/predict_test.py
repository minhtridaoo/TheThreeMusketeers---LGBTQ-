"""P6.1 – Dự đoán tập test (hoặc diễn tập) theo recipe hiện hành, ghi file nộp Codabench.

Test thật (tháng 01/2027):
    python scripts/predict_test.py --test <file_test_EN.tsv>
Diễn tập có chấm điểm (fold K làm "test giả", train trên các fold còn lại, nhãn bị ẩn khi dự đoán):
    python scripts/predict_test.py --pseudo-test-fold 0
Thử nhanh (mỗi fold chỉ train 64 mẫu, 1 epoch, test 16 mẫu – KẾT QUẢ VÔ NGHĨA):
    python scripts/predict_test.py --pseudo-test-fold 0 --smoke

Tùy chọn: --out-dir, --cache-dir (tiếp tục khi bị ngắt), --precision auto|fp16|bf16|fp32.
Đầu ra (trong out-dir): A.tsv, B.tsv (id<TAB>nhãn, không header), probabilities.csv, report.json.
Cần GPU cho các thành phần transformer (CPU chạy được nhưng rất chậm).
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sqe.data import TASKS, load_en  # noqa: E402
from sqe.submission import read_submission, write_submission  # noqa: E402
from sqe.testtime import predict_recipe  # noqa: E402

TEST_COLS = ["StereoQueerEval_id", "yt_title", "yt_description", "yt_comment"]


def set_precision(p):
    if p == "auto":
        return "auto"
    import torch
    from sqe import train as T
    dt = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": None}[p]
    T._amp_dtype = lambda device: dt if device.type == "cuda" else None
    return p


def load_test(path):
    test = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8")
    missing = set(TEST_COLS) - set(test.columns)
    if missing:
        raise SystemExit(f"File test thiếu cột {sorted(missing)}")
    if not test.StereoQueerEval_id.is_unique or (test.StereoQueerEval_id == "").any():
        raise SystemExit("ID trong file test bị trùng hoặc rỗng")
    return test


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--test", help="file TSV test của BTC")
    g.add_argument("--pseudo-test-fold", type=int, help="diễn tập: fold này làm test giả")
    ap.add_argument("--out-dir")
    ap.add_argument("--cache-dir")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--precision", default="auto", choices=["auto", "fp16", "bf16", "fp32"])
    a = ap.parse_args()

    df = load_en()
    truth = None
    if a.test:
        train, test, mode = df, load_test(a.test), "test"
    else:
        k = a.pseudo_test_fold
        train = df[df.fold != k].reset_index(drop=True)
        held = df[df.fold == k].reset_index(drop=True)
        truth = held[["StereoQueerEval_id", "stereotype", "hate_speech"]].copy()
        test, mode = held[TEST_COLS].copy(), f"pseudo_fold{k}"
        assert set(train.video_id).isdisjoint(held.video_id)
    if a.smoke:
        test = test.head(16).reset_index(drop=True)
        truth = truth.head(16) if truth is not None else None
        mode += "_SMOKE"
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    out = Path(a.out_dir) if a.out_dir else ROOT / "outputs" / "submissions" / f"{stamp}_{mode}"
    cache = Path(a.cache_dir) if a.cache_dir else out / "cache"
    out.mkdir(parents=True, exist_ok=True)
    precision = set_precision(a.precision)
    recipe = json.loads((ROOT / "outputs" / "submissions" / "recipe.json").read_text(encoding="utf-8"))
    print(f"Chế độ {mode} | train {len(train)} mẫu / {train.fold.nunique()} fold | test {len(test)} | ra: {out}", flush=True)
    print("Recipe:", {t: recipe[t]["strategy"] for t in "AB"}, flush=True)

    t0 = time.time()
    P, report, variants = predict_recipe(recipe, train, test, cache_dir=cache, smoke=a.smoke,
                               log=lambda *m: print(*m, flush=True))
    probs = test[["StereoQueerEval_id"]].copy()
    for t in "AB":
        labels = TASKS[t][1]
        pred = np.asarray(labels)[P[t].argmax(1)]
        path = write_submission(test.StereoQueerEval_id, pred, t, out / f"{t}.tsv")
        back = read_submission(path, t)
        assert list(back.StereoQueerEval_id) == list(test.StereoQueerEval_id) and set(back.label) <= set(labels)
        for j, l in enumerate(labels):
            probs[f"{t}_p_{l}"] = P[t][:, j]
        report.setdefault("label_distribution", {})[t] = pd.Series(pred).value_counts(normalize=True).round(4).to_dict()
    probs.to_csv(out / "probabilities.csv", index=False)

    if truth is not None:
        from sklearn.metrics import f1_score
        report["pseudo_test_scores"] = {}
        for t in "AB":
            col, labels = TASKS[t]
            pred = read_submission(out / f"{t}.tsv", t).label
            f1 = f1_score(truth[col], pred, labels=labels, average=None, zero_division=0)
            report["pseudo_test_scores"][t] = {"macro_f1": round(float(f1.mean()), 4),
                                               **{f"f1_{l}": round(float(v), 4) for l, v in zip(labels, f1)}}
            # các biến thể chỉ để so sánh (không phải file nộp): cách cũ, trung bình đều, không bias
            report.setdefault("pseudo_test_variants", {})[t] = {
                name: round(float(f1_score(truth[col], np.asarray(labels)[Pv.argmax(1)], labels=labels,
                                           average="macro", zero_division=0)), 4)
                for name, Pv in variants[t].items()}
    import torch
    import transformers
    report.update({"mode": mode, "precision": precision, "minutes_total": round((time.time() - t0) / 60, 1),
                   "n_train": len(train), "n_test": len(test), "recipe_strategy": {t: recipe[t]["strategy"] for t in "AB"},
                   "environment": {"torch": torch.__version__, "transformers": transformers.__version__,
                                   "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}})
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: report[k] for k in report if k in ("system_fresh_oof", "pseudo_test_scores", "pseudo_test_variants",
                                                           "label_distribution", "minutes_total")},
                     ensure_ascii=False, indent=1))
    print("Đã ghi", out)


if __name__ == "__main__":
    main()
