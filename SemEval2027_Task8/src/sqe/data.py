"""Đọc dữ liệu EN, gắn fold, và dựng input cho model."""
from pathlib import Path

import pandas as pd


def find_root(start=None):
    """Tìm thư mục dự án (chứa data/splits/folds_en.csv) từ thư mục hiện tại đi lên."""
    p = Path(start or Path.cwd()).resolve()
    for cand in [p, *p.parents]:
        if (cand / "data" / "splits" / "folds_en.csv").exists():
            return cand
    raise FileNotFoundError("Không tìm thấy thư mục dự án SemEval2027_Task8")


ROOT = find_root(Path(__file__).parent)
DATA_DIR = ROOT.parent / "StereoQueerEval_SemEval-2027" / "StereoQueerEval_training"
OUT_DIR = ROOT / "outputs"
SEED = 42

# subtask -> (tên cột nhãn, danh sách nhãn theo thứ tự cố định)
TASKS = {
    "A": ("stereotype", ["no", "yes"]),
    "B": ("hate_speech", ["no", "yes_implicit", "yes_explicit"]),
}


def load_en():
    """Train EN (2989 dòng) đã gắn video_id và fold (0–4) từ folds_en.csv."""
    df = pd.read_csv(DATA_DIR / "StereoQueerEval_EN_training.tsv", sep="\t",
                     dtype=str, keep_default_na=False, encoding="utf-8")
    folds = pd.read_csv(ROOT / "data" / "splits" / "folds_en.csv", dtype={"fold": int})
    df = df.merge(folds[["StereoQueerEval_id", "video_id", "fold"]],
                  on="StereoQueerEval_id", how="left", validate="one_to_one")
    # Kiểm tra nhanh: nếu file TSV bị Excel làm hỏng thì dừng ngay ở đây.
    assert len(df) == 2989, f"EN phải có 2989 dòng, đang có {len(df)}"
    assert df["fold"].notna().all()
    assert df.groupby("video_id")["fold"].nunique().eq(1).all()
    return df


# ---------------------------------------------------------------------------
# Các cách đưa input vào model (dùng cho context ablation ở notebook 04)
#   comment        : chỉ comment
#   pair_title     : cặp câu (comment, title)
#   pair_title_desc: cặp câu (comment, title + description)
#   concat_tdc     : ghép thẳng "title. description. comment" thành một chuỗi
# Với cặp câu, tokenizer tự chèn token ngăn cách </s></s> giữa hai phần.
# ---------------------------------------------------------------------------
INPUT_VARIANTS = ["comment", "pair_title", "pair_title_desc", "concat_tdc"]
# max_length dùng cho từng kiểu input (có description thì cần dài hơn)
INPUT_MAXLEN = {"comment": 128, "pair_title": 128, "pair_title_desc": 256, "concat_tdc": 256}


def build_inputs(df, variant):
    """Trả về (text_a, text_b). text_b = None nếu input chỉ có một chuỗi."""
    c, t, d = df["yt_comment"], df["yt_title"], df["yt_description"]
    if variant == "comment":
        return c.tolist(), None
    if variant == "pair_title":
        return c.tolist(), t.tolist()
    if variant == "pair_title_desc":
        return c.tolist(), (t + " | " + d).tolist()
    if variant == "concat_tdc":
        return (t + ". " + d + ". " + c).tolist(), None
    raise ValueError(f"Input không hợp lệ: {variant}")
