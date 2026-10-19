"""Tinh chỉnh cách ra quyết định từ xác suất, và ensemble — luôn theo kiểu LỒNG NHAU (nested).

Vì sao phải lồng nhau?
    Nếu dò ngưỡng / trọng số ensemble trên chính 2989 mẫu OOF rồi chấm lại trên đúng các mẫu
    đó, điểm sẽ đẹp hơn thực tế (đã "học" luôn cả nhiễu của tập chấm). Cách trung thực:
    với mỗi fold k, chỉ dùng OOF của 4 fold còn lại để dò, rồi áp dụng lên fold k.
    Điểm nhận được ước lượng đúng điều sẽ xảy ra trên tập test.

Bias tuning
    Macro-F1 thưởng cho lớp hiếm. argmax của xác suất thường thiên về lớp đông.
    Cộng một độ lệch b_c vào log-xác suất của từng lớp (lớp đầu cố định b = 0) rồi mới argmax:
        nhãn = argmax_c ( log p_c + b_c )
    Với 2 lớp, việc này tương đương đổi ngưỡng 0,5 thành ngưỡng khác.
"""
import itertools

import numpy as np
from sklearn.metrics import f1_score

GRID = np.round(np.arange(-1.5, 1.51, 0.1), 2)


def _macro(y, pred, k):
    return f1_score(y, pred, labels=list(range(k)), average="macro", zero_division=0)


def tune_bias(P, y, grid=GRID):
    """Tìm vector bias (b_0 = 0) tối đa macro-F1 trên (P, y). P: n×k xác suất, y: chỉ số lớp."""
    k = P.shape[1]
    L = np.log(np.clip(P, 1e-9, 1))
    best_b, best_s = np.zeros(k), _macro(y, L.argmax(1), k)
    for rest in itertools.product(grid, repeat=k - 1):
        b = np.array((0.0, *rest))
        s = _macro(y, (L + b).argmax(1), k)
        if s > best_s + 1e-12:
            best_b, best_s = b, s
    return best_b


def apply_bias(P, b):
    Z = np.log(np.clip(P, 1e-9, 1)) + b
    Z = np.exp(Z - Z.max(1, keepdims=True))
    return Z / Z.sum(1, keepdims=True)


def nested_bias(oof, labels):
    """Trả về (bảng OOF mới với xác suất đã chỉnh, {fold: bias đã học})."""
    out = oof.copy()
    pcols = [f"p_{l}" for l in labels]
    y = oof["y_true"].map({l: i for i, l in enumerate(labels)}).to_numpy()
    P = oof[pcols].to_numpy()
    biases = {}
    for k in sorted(oof["fold"].unique()):
        tr, te = (oof["fold"] != k).to_numpy(), (oof["fold"] == k).to_numpy()
        b = tune_bias(P[tr], y[tr])
        biases[int(k)] = b
        out.loc[te, pcols] = apply_bias(P[te], b)
    out["y_pred"] = np.array(labels)[out[pcols].to_numpy().argmax(1)]
    return out, biases


def average(oofs, labels, weights=None):
    """Trung bình (có trọng số) xác suất của nhiều bảng OOF cùng thứ tự dòng."""
    pcols = [f"p_{l}" for l in labels]
    w = np.ones(len(oofs)) if weights is None else np.asarray(weights, float)
    P = sum(wi * o[pcols].to_numpy() for wi, o in zip(w, oofs)) / w.sum()
    out = oofs[0][["StereoQueerEval_id", "video_id", "fold", "y_true"]].copy()
    out[pcols] = P
    out["y_pred"] = np.array(labels)[P.argmax(1)]
    return out


def greedy_select(oofs, labels, rows=None, max_steps=10):
    """Chọn ensemble kiểu Caruana: mỗi bước thêm (có thể lặp lại) model làm macro-F1 tăng nhiều nhất.

    rows: mặt nạ các dòng được phép dùng để chọn (để chạy lồng nhau). Trả về số lần chọn mỗi model.
    """
    pcols = [f"p_{l}" for l in labels]
    rows = np.ones(len(oofs[0]), bool) if rows is None else rows
    y = oofs[0]["y_true"].map({l: i for i, l in enumerate(labels)}).to_numpy()[rows]
    Ps = [o[pcols].to_numpy()[rows] for o in oofs]
    counts, cur, best = np.zeros(len(oofs), int), None, -1
    for _ in range(max_steps):
        cand = []
        for i, P in enumerate(Ps):
            S = P if cur is None else (cur * counts.sum() + P) / (counts.sum() + 1)
            cand.append(_macro(y, S.argmax(1), len(labels)))
        i = int(np.argmax(cand))
        if cand[i] <= best + 1e-9:
            break
        best = cand[i]
        cur = Ps[i] if cur is None else (cur * counts.sum() + Ps[i]) / (counts.sum() + 1)
        counts[i] += 1
    return counts


def nested_greedy(oofs, labels, max_steps=10):
    """Ensemble greedy lồng nhau: chọn trên 4 fold, chấm trên fold còn lại."""
    out = average(oofs, labels)
    pcols = [f"p_{l}" for l in labels]
    picks = {}
    for k in sorted(out["fold"].unique()):
        te = (out["fold"] == k).to_numpy()
        counts = greedy_select(oofs, labels, rows=~te, max_steps=max_steps)
        picks[int(k)] = counts
        ens = average(oofs, labels, weights=np.maximum(counts, 0) + 1e-12 * (counts.sum() == 0))
        out.loc[te, pcols] = ens.loc[te, pcols].to_numpy()
    out["y_pred"] = np.array(labels)[out[pcols].to_numpy().argmax(1)]
    return out, picks
