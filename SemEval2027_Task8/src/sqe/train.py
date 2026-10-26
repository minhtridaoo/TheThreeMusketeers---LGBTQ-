"""Fine-tune transformer (roberta-base, deberta-v3-base, ...) theo 5 fold.

Ý tưởng chính
-------------
* Một encoder (ví dụ roberta-base) đọc câu, lấy vector của token đầu <s> làm "tóm tắt câu".
* Trên vector đó gắn một hoặc nhiều "đầu" phân loại (head):
    - đơn nhiệm: chỉ head A (2 lớp) hoặc chỉ head B (3 lớp)
    - đa nhiệm (notebook 05): cả head A và head B dùng CHUNG encoder, loss = L_A + L_B
* Mỗi fold: train trên 4 fold, dự đoán fold còn lại → ghép lại thành OOF cho cả 2989 mẫu.
* Số epoch CỐ ĐỊNH, lấy model ở epoch cuối. Không chọn "epoch tốt nhất theo fold validation",
  vì như vậy là nhìn trộm đáp án của chính fold đang chấm → điểm bị thổi lên.

Kết quả được lưu đệm (cache) vào outputs/oof/ và outputs/runs/. Chạy lại notebook sẽ đọc
cache thay vì train lại, trừ khi gọi với force=True hoặc đổi cấu hình.
"""
import json
import os
import random
import time
import unicodedata

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import numpy as np
import torch
import transformers
from torch import nn
from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup

transformers.logging.set_verbosity_error()      # ẩn bảng "LOAD REPORT" dài dòng
try:
    transformers.logging.disable_progress_bar()
except AttributeError:                          # transformers bản cũ (ví dụ trên Kaggle)
    pass

from .data import OUT_DIR, SEED, TASKS, build_inputs
from .metrics import fold_scores
from .results import load_oof, make_oof, oof_path, save_oof

RUNS_DIR = OUT_DIR / "runs"

# Cấu hình mặc định (notebook nào cần thì ghi đè vài khoá).
DEFAULT = dict(
    model="roberta-base",
    input="comment",
    max_length=128,
    lr=2e-5,
    epochs=4,
    batch_size=16,        # số mẫu mỗi lần cập nhật = batch_size × accum
    accum=1,              # tích luỹ gradient khi GPU không đủ bộ nhớ cho batch lớn
    warmup=0.1,           # 10% số bước đầu: learning rate tăng dần từ 0
    weight_decay=0.01,
    dropout=0.1,
    class_weight=False,   # True: lớp hiếm được phạt nặng hơn khi đoán sai
    loss_weights=None,    # đa nhiệm: {"A": 1.0, "B": 1.0}
    grad_ckpt=False,      # tiết kiệm bộ nhớ GPU (chậm hơn ~30%)
    seed=SEED,
)


# Các encoder đã thử: tên ngắn (dùng trong tên file OOF) → cấu hình riêng.
ENCODERS = {
    "roberta": dict(model="roberta-base"),
    # deberta lớn hơn → batch 8 × tích luỹ 2 (batch hiệu dụng vẫn 16) cho vừa GPU 4 GB
    "deberta": dict(model="microsoft/deberta-v3-base", batch_size=8, accum=2),
    "twhate": dict(model="cardiffnlp/twitter-roberta-base-hate-latest"),
}
SHORT = {v["model"]: k for k, v in ENCODERS.items()}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class MultiHeadClassifier(nn.Module):
    """Encoder dùng chung + mỗi subtask một head (dense → tanh → linear, giống RoBERTa)."""

    def __init__(self, model_name, heads, dropout=0.1, grad_ckpt=False, encoder_config=None):
        super().__init__()
        # Rebuild saved architecture locally; load state_dict separately for offline inference.
        # dtype=float32: transformers 5.x mặc định nạp đúng kiểu đã lưu trong checkpoint.
        # deberta-v3-base lưu ở fp16 → train thẳng trên trọng số fp16 → NaN (P7, 26/10).
        # roberta-base / twitter-roberta vốn đã là fp32 nên không đổi gì với các run cũ.
        self.encoder = (AutoModel.from_config(encoder_config) if encoder_config is not None
                        else AutoModel.from_pretrained(model_name, dtype=torch.float32))
        if grad_ckpt:
            self.encoder.gradient_checkpointing_enable()
        h = self.encoder.config.hidden_size
        self.heads = nn.ModuleDict({
            t: nn.Sequential(nn.Dropout(dropout), nn.Linear(h, h), nn.Tanh(),
                             nn.Dropout(dropout), nn.Linear(h, n))
            for t, n in heads.items()
        })

    def forward(self, input_ids, attention_mask):
        hidden = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        cls = hidden[:, 0]                       # vector của token đầu tiên
        return {t: head(cls) for t, head in self.heads.items()}


# ---------------------------------------------------------------------------
# Tokenize + chia batch
# ---------------------------------------------------------------------------
def encode(tok, df, cfg):
    """Danh sách input_ids (chưa pad) cho từng mẫu.

    Với cặp câu, truncation="longest_first" cắt bớt phần DÀI hơn trước — thường là
    phần context (title/description), nên comment ít khi bị mất.
    """
    a, b = build_inputs(df, cfg["input"])
    if cfg.get("normalize") == "nfkc":
        # Tokenizer fast của DeBERTa-v3 (transformers 5.x) thiếu bước chuẩn hoá NFKC mà
        # SentencePiece gốc có → "…", emoji ghép thành [UNK]. Chỉ bật khi cfg yêu cầu (P7).
        nfkc = lambda xs: [unicodedata.normalize("NFKC", x) for x in xs]
        a, b = nfkc(a), (nfkc(b) if b is not None else None)
    kw = dict(truncation=True, max_length=cfg["max_length"])
    enc = tok(a, b, **kw) if b is not None else tok(a, **kw)
    return enc["input_ids"]


def _pad(ids, idx, pad_id, device):
    L = max(len(ids[i]) for i in idx)
    x = torch.full((len(idx), L), pad_id, dtype=torch.long)
    m = torch.zeros((len(idx), L), dtype=torch.long)
    for r, i in enumerate(idx):
        x[r, :len(ids[i])] = torch.tensor(ids[i])
        m[r, :len(ids[i])] = 1
    return x.to(device), m.to(device)


def _amp_dtype(device):
    """bf16 trên GPU đời mới (RTX 30xx / A2000 trở lên); fp16 trên T4/P100 của Kaggle; CPU thì tắt."""
    if device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _autocast(device):
    dt = _amp_dtype(device)
    return torch.autocast("cuda", dtype=dt) if dt else torch.autocast(device.type, enabled=False)


@torch.no_grad()
def predict_proba(model, ids, idx, pad_id, device, batch_size=64):
    """Xác suất từng lớp (softmax) cho các mẫu idx. Trả về {subtask: mảng n×k}."""
    model.eval()
    # sắp theo độ dài để pad ít → nhanh hơn, rồi trả về đúng thứ tự cũ
    order = sorted(range(len(idx)), key=lambda r: len(ids[idx[r]]))
    out = {}
    for s in range(0, len(order), batch_size):
        rows = order[s:s + batch_size]
        x, m = _pad(ids, [idx[r] for r in rows], pad_id, device)
        with _autocast(device):
            logits = model(x, m)
        for t, lg in logits.items():
            out.setdefault(t, np.zeros((len(idx), lg.shape[1]), dtype=np.float32))
            out[t][rows] = torch.softmax(lg.float(), -1).cpu().numpy()
    return out


def _class_weights(y, k, device):
    cnt = np.bincount(y, minlength=k).astype(float)
    return torch.tensor(len(y) / (k * np.maximum(cnt, 1)), dtype=torch.float, device=device)


def fit(df, train_idx, cfg, tasks, ids, tok, device, eval_idx=None, log=print):
    """Train một model trên các mẫu train_idx. Nếu có eval_idx thì chấm sau mỗi epoch."""
    set_seed(cfg["seed"])
    labels = {t: TASKS[t][1] for t in tasks}
    y = {t: df[TASKS[t][0]].map({l: i for i, l in enumerate(labels[t])}).to_numpy() for t in tasks}

    model = MultiHeadClassifier(cfg["model"], {t: len(labels[t]) for t in tasks},
                                cfg["dropout"], cfg["grad_ckpt"]).to(device)
    no_decay = ("bias", "LayerNorm.weight", "layernorm", "norm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": cfg["weight_decay"]},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0},
    ]
    opt = torch.optim.AdamW(groups, lr=cfg["lr"])
    bs, accum = cfg["batch_size"], cfg["accum"]
    steps_per_epoch = int(np.ceil(len(train_idx) / (bs * accum)))
    total = steps_per_epoch * cfg["epochs"]
    sched = get_linear_schedule_with_warmup(opt, int(cfg["warmup"] * total), total)

    lw = cfg["loss_weights"] or {t: 1.0 for t in tasks}
    loss_fn = {
        t: nn.CrossEntropyLoss(weight=_class_weights(y[t][train_idx], len(labels[t]), device)
                               if cfg["class_weight"] else None)
        for t in tasks
    }
    # fp16 dễ tràn số khi tính gradient → GradScaler nhân loss lên rồi chia lại (bf16 không cần)
    scaler = torch.amp.GradScaler("cuda", enabled=_amp_dtype(device) == torch.float16)
    rng = np.random.default_rng(cfg["seed"])
    history = []
    for ep in range(cfg["epochs"]):
        model.train()
        perm = rng.permutation(train_idx)
        run_loss, n_batches = 0.0, 0
        opt.zero_grad()
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            x, m = _pad(ids, b, tok.pad_token_id, device)
            with _autocast(device):
                logits = model(x, m)
                loss = sum(lw[t] * loss_fn[t](logits[t].float(),
                                              torch.tensor(y[t][b], device=device))
                           for t in tasks)
            if not torch.isfinite(loss).item():
                raise FloatingPointError(f"Nonfinite training loss at epoch {ep + 1}, batch {n_batches + 1}; no checkpoint saved.")
            scaler.scale(loss / accum).backward()
            n_batches += 1
            run_loss += loss.item()
            if n_batches % accum == 0 or s + bs >= len(perm):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad()
        rec = {"epoch": ep + 1, "train_loss": run_loss / n_batches}
        if eval_idx is not None:
            P = predict_proba(model, ids, eval_idx, tok.pad_token_id, device)
            for t in tasks:
                o = make_oof(df.iloc[eval_idx], t, P[t])
                rec[f"val_f1_{t}"] = fold_scores(o, labels[t])["macro_f1"].iloc[0]
        history.append(rec)
        log("    epoch {epoch}: loss {train_loss:.4f}".format(**rec)
            + "".join(f" | val macro-F1 {t} {rec[f'val_f1_{t}']:.4f}" for t in tasks if f"val_f1_{t}" in rec))
    return model, history


def _device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_cv(df, name, cfg=None, tasks=("A",), force=False, log=print):
    """Chạy 5 fold. Trả về {subtask: bảng OOF}. Lưu OOF + meta (cấu hình, lịch sử, thời gian).

    name  : tên thí nghiệm, ví dụ "roberta_comment" → outputs/oof/roberta_comment_A.csv
    tasks : ("A",), ("B",) hoặc ("A", "B") cho đa nhiệm
    """
    cfg = {**DEFAULT, **(cfg or {})}
    tasks = tuple(tasks)
    meta_path = RUNS_DIR / f"{name}_{''.join(tasks)}.json"
    if not force and meta_path.exists() and all(oof_path(name, t).exists() for t in tasks):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta["cfg"] == cfg and tuple(meta["tasks"]) == tasks:
            log(f"[cache] {name}: đọc kết quả đã train ({meta['minutes']:.1f} phút lúc chạy)")
            return {t: load_oof(name, t) for t in tasks}
        log(f"[cache] {name}: cấu hình đã đổi → train lại")

    device = _device()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    ids = encode(tok, df, cfg)
    probs = {t: np.zeros((len(df), len(TASKS[t][1])), dtype=np.float32) for t in tasks}
    histories, t0 = {}, time.time()
    for k in sorted(df["fold"].unique()):
        tr = np.where(df["fold"] != k)[0]
        va = np.where(df["fold"] == k)[0]
        log(f"  fold {k}: train {len(tr)} | valid {len(va)}")
        model, hist = fit(df, tr, cfg, tasks, ids, tok, device, eval_idx=va, log=log)
        P = predict_proba(model, ids, va, tok.pad_token_id, device)
        for t in tasks:
            probs[t][va] = P[t]
        histories[int(k)] = hist
        del model
        torch.cuda.empty_cache()
    minutes = (time.time() - t0) / 60
    oofs = {}
    for t in tasks:
        oofs[t] = make_oof(df, t, probs[t])
        save_oof(oofs[t], name, t)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"cfg": cfg, "tasks": list(tasks), "minutes": minutes,
                                     "device": str(device), "history": histories},
                                    ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"  xong {name} trong {minutes:.1f} phút")
    return oofs


def load_history(name, tasks=("A",)):
    """Lịch sử từng epoch của từng fold (để vẽ learning curve)."""
    import pandas as pd
    meta = json.loads((RUNS_DIR / f"{name}_{''.join(tasks)}.json").read_text(encoding="utf-8"))
    rows = [{"fold": int(k), **r} for k, h in meta["history"].items() for r in h]
    return pd.DataFrame(rows), meta


def fit_full_and_predict(df_train, df_test, cfg, tasks, log=print):
    """Train trên TOÀN BỘ train rồi dự đoán df_test (dùng khi có tập test thật)."""
    cfg = {**DEFAULT, **cfg}
    device = _device()
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    both = __import__("pandas").concat([df_train, df_test], ignore_index=True)
    ids = encode(tok, both, cfg)
    tr = np.arange(len(df_train))
    te = np.arange(len(df_train), len(both))
    # các cột nhãn của df_test có thể rỗng; fit() chỉ đọc nhãn ở chỉ số train
    for t in tasks:
        col = TASKS[t][0]
        if col not in both:
            both[col] = TASKS[t][1][0]
        both[col] = both[col].replace("", TASKS[t][1][0]).fillna(TASKS[t][1][0])
    model, _ = fit(both, tr, cfg, tasks, ids, tok, device, log=log)
    P = predict_proba(model, ids, te, tok.pad_token_id, device)
    del model
    torch.cuda.empty_cache()
    return P
