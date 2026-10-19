# Roadmap – SemEval-2027 Task 8 (StereoQueerEval), Subtask A + B, English

Cập nhật: 05/10/2026

## Tiến độ (11/10/2026)

| Bước | Trạng thái |
|---|---|
| 1–2 | Notebook 01, 02 xong |
| 3 | ✅ Notebook 03 đã chạy trên máy nhà: roberta-base A = 0,727, B = 0,586 (TF-IDF: 0,700 / 0,525) |
| 4, 5, 7, 8 | Notebook đã viết sẵn, **chưa chạy**: chạy trên Kaggle theo `..\kaggle\HUONG_DAN_KAGGLE.md` |
| 6 | Notebook đã viết sẵn, chạy trên máy nhà (Ollama + `qwen2.5:7b` đã tải) |

Code dùng chung nằm ở `src/sqe/`. Kernel trên máy nhà: **Python (sqe-gpu)** (`C:\Users\ADMIN\venvs\sqe`, có PyTorch GPU).

## 0. Bài toán trong 5 dòng

- **Input:** một bộ ba (tiêu đề video, mô tả video, comment YouTube).
- **Subtask A:** comment có chứa định kiến (stereotype) về LGBTQIA+ không → `yes` / `no`.
- **Subtask B:** comment có phải hate speech không → `no` / `yes_implicit` / `yes_explicit`.
- **Metric:** macro-F1 (mỗi lớp nặng như nhau, kể cả lớp ít).
- **Dữ liệu EN:** 2989 mẫu train, thuộc 243 video. Test có 743 mẫu, lấy từ **các video khác**, sẽ phát hành vào tháng 01/2027 (không có nhãn).

Mốc thời gian: Codabench mở trong tháng 10/2026 → pha đánh giá 10–31/01/2027 → system paper 02/2027.
**Hạn nộp đồ án của môn: CHƯA XÁC NHẬN, cần hỏi thầy.** Nếu hạn nộp trước tháng 01/2027 thì báo cáo môn dùng điểm cross-validation (bước 2–7), còn điểm test sẽ bổ sung sau.

## Quy tắc chung (áp dụng cho mọi bước)

1. **Luôn dùng `data/splits/folds_en.csv`** (5 fold, chia theo video). Báo cáo = trung bình ± độ lệch chuẩn qua 5 fold. Không dùng `train_test_split` ngẫu nhiên vì điểm sẽ bị thổi lên.
2. **Ghi mọi kết quả vào một file chung** `outputs/results/all_results.csv` với các cột: ngày, notebook, model, input, subtask, macro-F1 mean, std, F1 từng lớp. Đây chính là bảng kết quả cho báo cáo.
3. **Lưu dự đoán out-of-fold (OOF)** của mọi model, gồm xác suất từng lớp cho cả 2989 mẫu. Bước error analysis và ensemble cần đến file này.
4. Cố định `seed = 42`. Kernel Python 3.14. **Không mở file `.tsv` gốc bằng Excel**: ngày 05/10 Excel đã làm hỏng file này một lần.
5. Mỗi bước là một notebook, đánh số `0X_ten.ipynb`, có phần kết luận ngắn ở cuối.

---

## Bước 1 – Đọc hiểu dữ liệu bằng tay *(1–2 buổi, không cần code)*

**Mục tiêu:** hiểu thế nào là stereotype, thế nào là implicit và explicit, trước khi đưa cho máy học.

- Đọc `docs/Guidelines_for_Participants_v1.pdf`, mục 3 (Labels).
- Mở `outputs/eda/en_reading_sample.csv` (file này mở bằng Excel thoải mái). Đọc 90 mẫu và điền cột ghi chú: vì sao mẫu này là implicit chứ không phải explicit? Có cần đọc tiêu đề video mới hiểu được không?
- Xem lại phần kết luận của `notebooks/01_dataset_audit_eda.ipynb`.

**Xong khi:** bạn tự giải thích được 3 lớp của B cho người khác nghe, và có danh sách khoảng 5 "kiểu khó" ghi lại được. Danh sách này sẽ đưa vào phần error analysis của báo cáo.

## Bước 2 – Notebook 02: baseline cổ điển *(1 tuần, chạy trên CPU)*

**Mục tiêu:** có mốc điểm để mọi model sau so sánh vào, và dựng sẵn khung code dùng chung.

1. Tạo `src/` gồm `data.py` (đọc TSV + gắn fold), `metrics.py` (macro-F1, F1 từng lớp, confusion matrix), `submission.py` (ghi file `id<TAB>label` đúng format Codabench).
2. Baseline 1–2: random và majority. Mốc sàn từ EDA: majority = 0,357 (A) / 0,213 (B).
3. Baseline 3: TF-IDF (word 1–2-gram + char 2–5-gram) + Logistic Regression (`class_weight="balanced"`), làm riêng cho A và B.
4. So sánh input: (a) chỉ comment; (b) comment + title. Description để sau, vì EDA cho thấy ghép thẳng description làm giảm điểm khoảng 0,05.
5. In confusion matrix của B để thấy implicit và explicit bị nhầm với nhau đến mức nào.

**Xong khi:** `all_results.csv` có khoảng 8 dòng (3 baseline × 2 subtask, cộng các biến thể input) và có file OOF của TF-IDF.

## Bước 3 – Notebook 03: fine-tune transformer *(1–2 tuần, chạy GPU trên Kaggle/Colab)*

**Mục tiêu:** có model chính.

- Chạy trên **Kaggle Notebook** (GPU T4/P100 miễn phí khoảng 30 giờ/tuần). GPU 4GB của máy nhà chỉ đủ debug với batch nhỏ.
- Model: `roberta-base`. Input chỉ có comment, `max_length=128`.
- Cấu hình khởi điểm: lr 2e-5, batch 16, 4 epoch, warmup 10%, fp16. Train 5 fold × 2 subtask.
- Lưu OOF cho cả 2989 mẫu và lưu cả 5 checkpoint của từng fold (để dự đoán tập test sau này).

**Xong khi:** có điểm macro-F1 5 fold của roberta-base cho A và B, đặt cạnh TF-IDF trong bảng.

## Bước 4 – Notebook 04: context ablation *(1–2 tuần, đây là câu hỏi nghiên cứu chính)*

Guideline nhấn mạnh rằng comment cần được hiểu **cùng với tiêu đề và mô tả video**. Bước này đo xem context giúp được bao nhiêu và nên đưa vào theo cách nào.

| Biến thể | Input | max_length |
|---|---|---|
| C0 | comment | 128 |
| C1 | (comment, title) dạng cặp câu | 128 |
| C2 | (comment, title + description) dạng cặp câu, chỉ cắt phần context | 256 |
| C3 | ghép thẳng title + description + comment (để đối chứng) | 256 |

- Giữ cố định model tốt nhất của bước 3, chỉ thay input.
- Nếu còn thời gian, đổi encoder: `microsoft/deberta-v3-base`, hoặc model đã học về hate speech như `cardiffnlp/twitter-roberta-base-hate-latest` hay `GroNLP/hateBERT`.

**Xong khi:** có bảng ablation, kèm 1–2 câu kết luận về việc context giúp hay hại và giúp lớp nào.

## Bước 5 – Notebook 05: tận dụng quan hệ A–B *(1 tuần)*

A và B liên quan khá chặt (Cramér's V = 0,53; P(stereotype | explicit) = 80%).

- **Multi-task:** một encoder dùng chung, hai đầu ra A và B, loss = L_A + L_B.
- **Tối ưu cho macro-F1:** dùng class weight, và dò ngưỡng (A) hoặc hệ số nhân xác suất (B) trên OOF.
- So sánh với hai model riêng lẻ của bước 4.

## Bước 6 *(tuỳ chọn)* – LLM zero-shot / few-shot

- Ban tổ chức có một baseline zero-shot dùng Qwen2.5-7B-Instruct. Có thể chạy model tương tự qua Ollama (đã có trong stack `D:\LLM CỦA TRÍ`) để so sánh, hoặc thử few-shot với vài ví dụ lấy từ train.
- Chỉ làm nếu các bước 2–5 đã xong. Kết quả là một dòng so sánh hay để đưa vào paper.

## Bước 7 – Notebook 07: error analysis *(1 tuần)*

Dùng OOF của model tốt nhất:
- Confusion matrix của B; tách lỗi implicit ↔ explicit và hate ↔ no.
- Điểm theo từng video, theo độ dài comment, theo việc comment có nhắc tới nhóm bị nhắm hay không.
- Đọc khoảng 30 mẫu sai và xếp vào các "kiểu khó" đã ghi ở bước 1.

**Xong khi:** có 2–3 hình và 1 bảng ví dụ lỗi cho báo cáo.

## Bước 8 – Hệ thống cuối + ensemble *(tháng 12)*

- Trung bình xác suất của 2–3 model tốt nhất. Chọn tổ hợp dựa trên OOF, **không** dựa trên điểm test.
- Dự đoán test bằng cách lấy trung bình 5 checkpoint của 5 fold.
- Chạy thử `src/submission.py` trên train để chắc chắn file đúng format.

## Bước 9 – Nộp bài lên Codabench *(10–31/01/2027)*

- Theo dõi Google Group của task để nhận link Codabench (trong tháng 10/2026), đăng ký đội.
- Khi có test: dự đoán, rồi nộp **riêng** cho Subtask A và Subtask B (EN).
- File nộp gồm mỗi dòng một cặp `StereoQueerEval_id<TAB>label`; ID copy y nguyên từ file test, không kèm văn bản, nhãn đúng chính tả.
- Số lượt nộp có giới hạn (sẽ công bố sau), nên chọn trước 1–3 run tốt nhất.

## Bước 10 – Viết báo cáo / paper

- Báo cáo môn: phần giới thiệu, dữ liệu + EDA (đã có hình), phương pháp, kết quả (bảng ở bước 2–5), ablation, error analysis, kết luận.
- System paper SemEval (02/2027): format ACL. Ghi đủ tài nguyên ngoài đã dùng (pretrained model, Ollama/LLM...).

---

## Lịch gợi ý

| Thời gian | Việc |
|---|---|
| Tuần 05–11/10 | Bước 1 + bắt đầu bước 2; **hỏi thầy hạn nộp** |
| 12–25/10 | Xong bước 2, bắt đầu bước 3 (lập tài khoản Kaggle) |
| 26/10–08/11 | Bước 3 + bước 4 |
| 09–22/11 | Bước 5 (+ bước 6 nếu dư thời gian) |
| 23/11–06/12 | Bước 7, bắt đầu viết báo cáo |
| Tháng 12 | Bước 8, hoàn thiện báo cáo môn |
| 10–31/01/2027 | Bước 9 – nộp test |
| Tháng 02/2027 | Bước 10 – system paper |

## Gợi ý chia việc trong nhóm

- **Người 1:** khung `src/` + bước 2 + quản lý `all_results.csv`.
- **Người 2:** bước 3–4 trên Kaggle.
- **Cả nhóm:** bước 1 (đọc tay) và bước 7 (error analysis). Sau bước 4 thì chia nhau bước 5 và bước 6.



    ## Cập nhật tin tức từ BTC 
    - Thông tin về cuộc thi sắp tới: Thank you for your interest in our shared task. We are currently working on the evaluation/scoring script and on getting the three Codabench platforms ready (it will be three different pages for the three subtasks). We will make them available in approximately 2 weeks. As soon as they are ready we will announce it on the Google Group.
    - Release test triplets: Dear StereoQueerEval participants,

    We are happy to announce that we have decided to release 100 Persian (Farsi) text triplets (title, description, comments) as part of the development set. The remaining Persian data will be released as part of the test set in January, when the evaluation period starts.


    The archive containing the training data is password-protected. You can open it with the following password.

    Password Farsi Development Data: Rainbow2027!

    Once again, the new file is distributed as a password-protected RAR archive. We recommend extracting it with WinRAR on Windows; on macOS you can use the command-line version of WinRAR or a free tool that supports encrypted RAR archives, such as The Unarchiver or Keka; on Linux, unrar x archive.rar.

    Happy experimenting!

    Best,
    The StereoQueerEval Organizers 🌈