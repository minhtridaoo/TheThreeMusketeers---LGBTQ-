# SemEval-2027 Task 8 (StereoQueerEval) – Subtask A + B, English

**Tiến độ kiểm chứng 20/10/2026: P0–P4 hoàn thành, chuyển P5 báo cáo/bàn giao.**
Đã kiểm chứng smoke/dryrun/final Kaggle, có 7 checkpoint final train toàn EN; trọng số đầy đủ giữ trong `../kaggle/results/sqe_p4_final.zip`.
Xem [ROADMAP duy nhất](ROADMAP.md), [handoff](../HANDOFF_CS221.md), [worklog](../WORKLOG_CS221.md) và [kiểm chứng P4](outputs/audit/2026-10-20-p4-results/P4_REVIEW.md).
Điểm P3 là development CV; dryrun chỉ kiểm tra pipeline, chưa có test độc lập/nộp BTC. Nội dung cấu trúc EDA ban đầu bên dưới chỉ mô tả một phần dự án.

```
SemEval2027_Task8/
├── notebooks/
│   └── 01_dataset_audit_eda.ipynb   # audit + EDA + tạo folds (đã chạy sẵn, có output)
├── data/splits/
│   ├── folds_en.csv                 # 5-fold, chia theo video, stratify A×B  ← DÙNG CHO MỌI THÍ NGHIỆM
│   ├── folds_it.csv, folds_nl.csv
│   └── folds_all.csv
├── docs/Guidelines_for_Participants_v1.pdf   # guideline chính thức (lấy lại từ lịch sử git repo BTC)
└── outputs/eda/
    ├── figures/*.png                # 8 biểu đồ cho báo cáo/slide
    ├── en_reading_sample.csv        # 90 mẫu (15/tổ hợp A×B) để đọc tay, có cột ghi chú
    └── eda_summary.json             # các con số chính
```

Dữ liệu gốc: `..\StereoQueerEval_SemEval-2027\StereoQueerEval_training\` (đổi `DATA_DIR` ở cell cấu hình nếu chuyển máy).

## Chạy notebook
- Dùng **Python 3.14** (`C:\Python314\python.exe`) làm kernel: đã có đủ pandas / sklearn / transformers.
  Kernel Python 3.13 mặc định thiếu `transformers` **và** bị Windows *Controlled Folder Access* chặn ghi vào `Documents`
  (lỗi `FileNotFoundError` khi tạo thư mục). Hoặc: thêm Python 3.13 vào danh sách *Allow an app through Controlled folder access*.
- Lần chạy đầu sẽ tải 3 tokenizer nhỏ (roberta-base, deberta-v3-base, xlm-roberta-base).

## Dùng folds
```python
folds = pd.read_csv("data/splits/folds_en.csv")
df = df.merge(folds[["StereoQueerEval_id", "fold"]], on="StereoQueerEval_id")
dev = df[df.fold == 0]; train = df[df.fold != 0]      # dev nhanh; báo cáo = trung bình 5 fold
```
Không dùng split ngẫu nhiên: test chính thức lấy từ các video khác train.
