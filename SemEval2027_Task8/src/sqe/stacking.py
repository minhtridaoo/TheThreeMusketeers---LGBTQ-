"""Stacking (P8/P8.2): hồi quy logistic trên log-xác suất của nhiều model ứng viên.

Cấu hình cố định (đặt trước, không dò): chuẩn hoá đặc trưng, LogisticRegression(C=1,
class_weight="balanced"). Đặc trưng mỗi model: 2 lớp → logit log(p1/p0); 3 lớp → log p từng lớp.
Dùng chung cho: cross-fit OOF ở P3/P8 và dự đoán test ở P6.2.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

STACKER = {"C": 1.0, "class_weight": "balanced", "standardize": True, "max_iter": 5000,
           "features": "binary: logit log(p1/p0); multiclass: log p per class; clip [1e-6, 1]"}


def features(prob_list):
    """prob_list: danh sách ma trận xác suất (n × k), cùng thứ tự dòng, mỗi model một ma trận."""
    cols = []
    for P in prob_list:
        P = np.clip(np.asarray(P, float), 1e-6, 1.0)
        cols.append(np.log(P[:, 1] / P[:, 0])[:, None] if P.shape[1] == 2 else np.log(P))
    return np.hstack(cols)


def make_stacker():
    return make_pipeline(StandardScaler(), LogisticRegression(C=STACKER["C"], class_weight=STACKER["class_weight"],
                                                              max_iter=STACKER["max_iter"]))


def fit_predict(X_fit, y_fit, X_pred, labels):
    """Fit stacker rồi trả xác suất cho X_pred, cột theo đúng thứ tự `labels`."""
    model = make_stacker().fit(X_fit, y_fit)
    P = model.predict_proba(X_pred)
    return P[:, [list(model.classes_).index(l) for l in labels]], model


def crossfit(X, y, folds, labels):
    """OOF của stacker: với mỗi fold k, fit trên các fold khác, dự đoán fold k."""
    P = np.zeros((len(y), len(labels)))
    for k in sorted(np.unique(folds)):
        fit = folds != k
        P[~fit], _ = fit_predict(X[fit], y[fit], X[~fit], labels)
    return P
