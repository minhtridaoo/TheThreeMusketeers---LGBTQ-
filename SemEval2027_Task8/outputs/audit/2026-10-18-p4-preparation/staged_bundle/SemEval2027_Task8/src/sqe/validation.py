"""Validate OOF identity and probabilities before scoring or combining systems."""
import numpy as np

class InvalidOOFError(ValueError):
    pass

def validate_probabilities(p):
    p = np.asarray(p, dtype=float)
    if p.ndim != 2 or p.shape[0] == 0 or p.shape[1] < 2:
        raise InvalidOOFError("Probability matrix must be nonempty, n x classes.")
    if not np.isfinite(p).all():
        raise InvalidOOFError("Nonfinite probabilities (NaN/Inf); failed run cannot be scored.")
    if (p < 0).any() or (p > 1 + 1e-7).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise InvalidOOFError("Probabilities must be in [0,1] and sum to one.")
    return p

def validate_oof(frame, labels, reference=None, target_column=None):
    keys = ["StereoQueerEval_id", "video_id", "fold", "y_true", "y_pred"]
    missing = set(keys + [f"p_{label}" for label in labels]) - set(frame.columns)
    if missing:
        raise InvalidOOFError(f"Missing OOF columns: {sorted(missing)}")
    if frame[keys].isna().any().any() or not frame.StereoQueerEval_id.is_unique:
        raise InvalidOOFError("OOF has missing values or duplicate IDs.")
    if not set(frame.y_true) <= set(labels) or not set(frame.y_pred) <= set(labels):
        raise InvalidOOFError("OOF label outside task vocabulary.")
    if not frame.groupby("video_id").fold.nunique().eq(1).all():
        raise InvalidOOFError("One video appears in multiple folds.")
    p = validate_probabilities(frame[[f"p_{label}" for label in labels]].to_numpy())
    if not np.array_equal(np.asarray(labels)[p.argmax(axis=1)], frame.y_pred):
        raise InvalidOOFError("Predictions do not match probability argmax.")
    if reference is not None:
        ref = reference.set_index("StereoQueerEval_id")
        if not ref.index.is_unique or len(frame) != len(ref) or set(frame.StereoQueerEval_id) != set(ref.index):
            raise InvalidOOFError("OOF IDs/count do not match the canonical dataset.")
        aligned = ref.loc[frame.StereoQueerEval_id]
        for got, expected in [("video_id", "video_id"), ("fold", "fold"), ("y_true", target_column)]:
            if not np.array_equal(frame[got], aligned[expected]):
                raise InvalidOOFError(f"OOF {got} does not match canonical data.")
    return frame

def validate_aligned(oofs, labels):
    if not oofs:
        raise InvalidOOFError("At least one OOF is required.")
    keys = ["StereoQueerEval_id", "video_id", "fold", "y_true"]
    for frame in oofs:
        validate_oof(frame, labels)
        if not np.array_equal(frame[keys].to_numpy(), oofs[0][keys].to_numpy()):
            raise InvalidOOFError("OOFs have different row order, IDs, folds, videos or gold labels.")
    return oofs

