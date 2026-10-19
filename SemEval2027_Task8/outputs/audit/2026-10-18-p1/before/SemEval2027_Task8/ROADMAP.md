# Roadmap – SemEval-2027 Task 8 (StereoQueerEval), Subtask A + B, English

Cập nhật: **18/10/2026** — đã kiểm tra output hiện có.

## Tiến độ hiện hành (18/10/2026)

| Bước | Trạng thái có bằng chứng |
|---|---|
| 1–2 | Notebook 01, 02 đã chạy; dữ liệu/folds/baseline đã kiểm tra |
| 3 | RoBERTa có OOF hợp lệ: A = 0,7274, B = 0,5856 (mean 5 fold) |
| 4 | Đã chạy context/encoder; RoBERTa và Twitter-RoBERTa hợp lệ; **DeBERTa A/B bị NaN, đánh dấu failed** |
| 5 | Đã chạy single/MT/cw/bias, có OOF; bảng tổng hợp ghi đè biến thể, cần phục hồi |
| 6 | **Qwen full đã hoàn thành và kiểm chứng: 4×599 mẫu, 0 lỗi; đã merge 4 dòng** |
| 7 | Đã xuất phân tích lỗi; **0/30** mẫu B được điền kiểu lỗi |
| 8 | Đã chạy ensemble/dry-run; **nhánh greedy lỗi NaN, cần chọn lại trên OOF sạch**; chưa thấy checkpoint lưu |
| 9–10 | Chưa xác nhận nộp test; còn hoàn thiện kết luận notebook và báo cáo |

Cả 8 notebook có execution count đầy đủ; điều này không thay cho kiểm tra tính hợp lệ của kết quả.
Notebook 04–08 còn mục kết luận chờ viết.

- Đọc [HANDOFF hiện hành](../HANDOFF_CS221.md) để tiếp tục công việc.
- [PIPELINE](PIPELINE.md) ghi luồng dữ liệu, kế hoạch P0–P5 và điều kiện hoàn thành.
- [WORKLOG](../WORKLOG_CS221.md) ghi các mốc đã kiểm chứng.
- [Báo cáo kiểm tra 18/10](outputs/audit/2026-10-18/REVIEW.md) và audit JSON là bằng chứng hiện hành.
- Bảng master hiện **28 dòng**, gồm 4 dòng Qwen; vẫn còn các vấn đề của 04/05/08 nêu trên.

**Việc tiếp theo: P1 — làm sạch nguồn kết quả và kiểm tra lại ensemble bằng OOF đã có, chưa cần train model mới.**
Sau đó đọc/ghi chú 30 lỗi B, đồng bộ kết quả và kết luận, chốt hệ thống, bổ sung lưu/nạp checkpoint.

Code dùng chung nằm ở `src/sqe/`. Notebook 06 được chạy bằng runner Kaggle riêng tại
`../kaggle/kaggle_runner_06.ipynb`; hiện đã có kết quả full trên máy, không cần chạy lại.

## 0. Bài toán trong 5 dòng

- **Input:** một bộ ba (tiêu đề video, mô tả video, comment YouTube).
- **Subtask A:** comment có chứa định kiến (stereotype) về LGBTQIA+ không → `yes` / `no`.
- **Subtask B:** comment có phải hate speech không → `no` / `yes_implicit` / `yes_explicit`.
- **Metric:** macro-F1 (mỗi lớp nặng như nhau, kể cả lớp ít).
- **Dữ liệu EN đã kiểm tra:** 2.989 mẫu train, 243 nhóm video. Tài liệu BTC đang lưu mô tả test EN 743 mẫu từ video khác, dự kiến tháng 01/2027; lịch và quy cách cần xác minh trước lúc nộp.

**Mốc BTC ghi trong kế hoạch cũ, chưa xác minh lại ngày 18/10:** Codabench trong tháng 10/2026 → đánh giá 10–31/01/2027 → system paper 02/2027. Kiểm tra thông báo chính thức trước khi dùng làm hạn nộp.
**Hạn nộp đồ án của môn: CHƯA XÁC NHẬN, cần hỏi thầy.** Nếu hạn nộp trước tháng 01/2027 thì báo cáo môn dùng điểm cross-validation (bước 2–7), còn điểm test sẽ bổ sung sau.

## Quy tắc chung (cập nhật sau audit 18/10)

1. **Dùng `data/splits/folds_en.csv`**, chia theo video. Supervised báo cáo mean ± std 5 fold (ddof=0); riêng notebook 06 chỉ fold 0, 599 mẫu, std để trống. Không chia ngẫu nhiên comment cùng video sang cả train/validation.
2. **Ghi kết quả có định danh riêng** vào `outputs/results/all_results.csv`, gồm experiment/run_id và evaluation_scope. Khóa logger cũ đang làm mất biến thể notebook 05: phải sửa và phục hồi trước khi lấy bảng này làm bảng báo cáo cuối. Kiểm tra lại điểm từ dự đoán gốc khi có mâu thuẫn.
3. **OOF supervised:** đủ 2.989 ID, đúng nhãn/fold/video, xác suất hữu hạn, tổng hàng bằng 1 và argmax khớp nhãn. File failed không được vào ensemble. Qwen chỉ có nhãn của 599 mẫu, lưu ở `outputs/llm_predictions/`, không giả lập xác suất hoặc trộn vào OOF 5 fold.
4. Cố định `seed = 42`; ghi phiên bản Python/thư viện/precision thực tế của mỗi môi trường, không coi mọi lượt đều dùng Python 3.14. **Không mở file `.tsv` gốc bằng Excel**: ngày 05/10 Excel đã làm hỏng file này một lần.
5. Mỗi bước là một notebook, có phần kết luận dựa trên số liệu; cập nhật handoff/worklog sau mỗi mốc đã kiểm chứng. Điểm chọn model/bias/ensemble trên OOF là điểm phát triển, chưa phải test độc lập hoặc nested CV toàn pipeline.

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

- Các lượt train lớn có thể chạy trên Kaggle; kiểm tra accelerator/quota thực tế khi sử dụng. Notebook 03 đã có OOF, không cần train lại chỉ để cập nhật bảng.
- Model: `roberta-base`. Input chỉ có comment, `max_length=128`.
- Cấu hình khởi điểm: lr 2e-5, batch 16, 4 epoch, warmup 10%, fp16. Train 5 fold × 2 subtask.
- Đã có OOF cho 2.989 mẫu. **Chưa thấy checkpoint trong dự án**; bổ sung lưu model/tokenizer/config trong P4. Kế hoạch hiện tại là train cấu hình đã chốt trên toàn bộ train rồi lưu/reload; trung bình checkpoint 5 fold chỉ là phương án khác nếu triển khai và kiểm chứng riêng.

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

## Bước 6 — LLM zero-shot / few-shot: đã hoàn thành và kiểm chứng

- Qwen2.5-7B-Instruct Q4_K_M qua Ollama trên Kaggle, 4 cấu hình ×599 mẫu fold 0.
- Zero-shot: A **0,607411**, B **0,461242**; few-shot: A **0,692715**, B **0,518695**.
- Few-shot A 4 ví dụ, B 6 ví dụ từ fold 1–4; không trùng video của fold 0.
- 0 failed attempt; ID/nhãn/cache/fingerprint và điểm đều được kiểm tra ngày 18/10.
- So với TF-IDF cùng fold 0 (A 0,705628; B 0,524467), Qwen few-shot chưa vượt.
  Giữ đây là thí nghiệm đối chiếu, không đưa vào ensemble xác suất hiện tại.
- `all_results.csv` đã ghép 4 dòng Qwen; chi tiết trong [báo cáo kiểm tra](outputs/audit/2026-10-18/REVIEW.md).
- Kết quả tính toán đã xong; phần kết luận bằng lời trong notebook vẫn cần viết.

## Bước 7 – Notebook 07: error analysis *(1 tuần)*

Dùng OOF của model tốt nhất:
- Confusion matrix của B; tách lỗi implicit ↔ explicit và hate ↔ no.
- Điểm theo từng video, theo độ dài comment, theo việc comment có nhắc tới nhóm bị nhắm hay không.
- Đọc khoảng 30 mẫu sai và xếp vào các "kiểu khó" đã ghi ở bước 1.

**Xong khi:** có 2–3 hình và 1 bảng ví dụ lỗi cho báo cáo.

## Bước 8 — Hệ thống cuối + ensemble: cần tái xác nhận

- Công thức cũ đã lưu: A top-3 mean, B MT best single; điểm cũ A 0,741860 / B 0,600616.
- Greedy bị nhiễm NaN từ DeBERTa kể cả trọng số bằng 0. **Sửa P1, chạy lại lựa chọn trên OOF sạch ở P3** trước khi gọi đây là hệ thống cuối.
- Dry-run đã xuất 599 dòng A/B, ID và nhãn hợp lệ; điểm dry-run chỉ kiểm tra quy trình.
- `predict_recipe` hiện train lại thành phần từ dữ liệu đưa vào; chưa có bằng chứng đã lưu đủ checkpoint để chỉ nạp và dự đoán.
- P4: lưu/reload model + tokenizer + cấu hình, kiểm tra ở phiên mới, sau đó train full-train cấu hình đã chốt nếu cần.
- Không cần Run All 01–08 để sửa bảng; tách bước tính ensemble trên OOF khỏi dry-run có train.

## Bước 9 – Nộp bài lên Codabench *(10–31/01/2027)*

- Theo dõi Google Group của task để nhận link Codabench (trong tháng 10/2026), đăng ký đội.
- Khi có test: dự đoán, rồi nộp **riêng** cho Subtask A và Subtask B (EN).
- File nộp gồm mỗi dòng một cặp `StereoQueerEval_id<TAB>label`; ID copy y nguyên từ file test, không kèm văn bản, nhãn đúng chính tả.
- Số lượt nộp có giới hạn (sẽ công bố sau), nên chọn trước 1–3 run tốt nhất.

## Bước 10 – Viết báo cáo / paper

- Báo cáo môn: phần giới thiệu, dữ liệu + EDA (đã có hình), phương pháp, kết quả (bảng ở bước 2–5), ablation, error analysis, kết luận.
- System paper SemEval (02/2027): format ACL. Ghi đủ tài nguyên ngoài đã dùng (pretrained model, Ollama/LLM...).

---

## Lịch công việc tiếp theo (không giả định hạn môn)

| Thứ tự | Việc | Ước lượng |
|---|---|---|
| P0 — đã xong 18/10 | Audit 06/OOF, merge Qwen, đồng bộ tài liệu | Hoàn tất |
| P1 — tiếp theo | Chặn OOF hỏng, sửa key kết quả/NaN ensemble, phục hồi bảng 05 | 1 buổi CPU |
| P2 | Đọc/ghi chú 30 lỗi B, giải thích context/MT/LLM, viết nháp kết luận | 1–2 buổi |
| P3 | Chọn lại model/ensemble, cập nhật 07 và bảng tổng hợp | 1 buổi; tận dụng OOF |
| P4 | Đóng gói lưu/nạp model, dry-run, train cấu hình cuối khi cần | GPU theo runtime đo được |
| P5 | Hoàn thiện báo cáo; xác minh quy cách/lịch BTC trước khi nộp | 2–3 buổi viết, lịch nộp riêng |

Đây là ước lượng để sắp xếp công việc; ưu tiên và tiêu chí hoàn thành nằm ở [PIPELINE.md](PIPELINE.md).
Hạn môn chưa xác nhận. Có thể viết dữ liệu/phương pháp trong lúc làm P1.

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