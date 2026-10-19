"""Tính macro-F1 theo fold và so sánh hai cấu hình trên cùng các fold."""
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def fold_scores(oof, labels, pred_col="y_pred"):
    """Bảng điểm từng fold: macro_f1 + F1 từng lớp. `oof` có cột fold, y_true, y_pred."""
    rows = []
    for k, g in oof.groupby("fold"):
        per_class = f1_score(g["y_true"], g[pred_col], labels=labels,
                             average=None, zero_division=0)
        rows.append({"fold": int(k), "macro_f1": per_class.mean(),
                     **{f"f1_{l}": s for l, s in zip(labels, per_class)}})
    return pd.DataFrame(rows)


def summary(scores):
    """Một dòng: mean ± std (std tính qua 5 fold, ddof=0) cho mọi cột điểm."""
    cols = [c for c in scores.columns if c != "fold"]
    out = {f"{c}": scores[c].mean() for c in cols}
    out["macro_f1_std"] = scores["macro_f1"].std(ddof=0)
    return out


def fmt(scores):
    return f"{scores['macro_f1'].mean():.4f} ± {scores['macro_f1'].std(ddof=0):.4f}"


def paired_delta(scores_new, scores_base):
    """So sánh cặp theo fold: chênh lệch trung bình và số fold mà cấu hình mới thắng.

    Vì hai cấu hình chạy trên cùng 5 fold, nhìn chênh lệch TỪNG fold đáng tin hơn so
    hai con số trung bình. Thắng 5/5 fold thì khá chắc; thắng 3/5 thì coi như hoà.
    """
    a = scores_new.set_index("fold")["macro_f1"]
    b = scores_base.set_index("fold")["macro_f1"]
    d = (a - b).dropna()
    return {"delta_mean": d.mean(), "delta_std": d.std(ddof=0),
            "wins": int((d > 0).sum()), "n_folds": len(d)}


def pred_from_proba(oof, labels):
    """Nhãn dự đoán = argmax của các cột p_<nhãn>."""
    P = oof[[f"p_{l}" for l in labels]].to_numpy()
    return np.array(labels)[P.argmax(1)]
