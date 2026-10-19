# Handoff hiện hành — CS221 / StereoQueerEval
Cập nhật và kiểm chứng: **18/10/2026**. Phạm vi: **Subtask A + B, English**.

Đọc file này khi tiếp tục công việc. [ROADMAP duy nhất](SemEval2027_Task8/ROADMAP.md) đã gộp pipeline, tiến độ và kế hoạch;
[worklog](WORKLOG_CS221.md) ghi lịch sử đã kiểm chứng. PIPELINE.md chỉ còn liên kết chuyển tiếp.

## 1. Trạng thái và hành động tiếp theo
- **P0 và P1 đã hoàn thành; vị trí hiện tại là P2 — đọc lỗi và viết kết luận.**
- Notebook 06 full đã kiểm chứng; không cần chạy lại Qwen.
- Đã sửa kiểm tra OOF, khóa logger và NaN ensemble; 35 test đạt.
- Phục hồi đủ 12 dòng notebook 05; master hiện 38 dòng: 34 completed_valid, 2 failed_nonfinite, 2 pending_revalidation.
- Smoke greedy trên 9 ứng viên hợp lệ/task: A 0,745465, B 0,595965; chưa chọn recipe mới.
- Cả 8 notebook từng chạy hết trước P1. Code 05/07/08 đã sửa, output phụ thuộc đã xóa;
  chưa chạy lại notebook sau sửa. Bản đã chạy cũ nằm trong backup P1.
- 30 mẫu lỗi B còn **0/30** có ghi chú. Việc tiếp theo: đọc 10 mẫu đầu, đối chiếu guideline và ghi kiểu khó.
- P3 còn chạy lại phần lựa chọn/summary/07; P4 còn checkpoint; P5 còn báo cáo.
- Phần chạy thí nghiệm đi trước lịch gốc; không thể kết luận sát hạn môn vì chưa biết ngày nộp.
- Xem [báo cáo P1](SemEval2027_Task8/outputs/audit/2026-10-18-p1/P1_REVIEW.md).

## 2. Notebook 06 — đủ bằng chứng hoàn thành
Nguồn: [kết quả kiểm tra](SemEval2027_Task8/outputs/audit/2026-10-18/REVIEW.md).

- Model `qwen2.5:7b-instruct-q4_K_M`, Ollama 0.40.3, context 4096.
- `execution_status.json`: `mode=full, status=complete`; notebook 5/5 code cell đã chạy.
- 4 cấu hình × 599 mẫu = **2.396 dự đoán**; 0 failed attempts.
- Đã đối chiếu ID, fold, video, nhãn thật, JSON trả về, nhãn dự đoán, fingerprint model/prompt;
  tính lại macro-F1 và F1 từng lớp đều khớp.
- Few-shot A có 4 ví dụ, B có 6 ví dụ, từ fold 1–4; không trùng ID hoặc video với fold 0.
- Log ghi lượt chạy ngày **14/10/2026**; ngày **18/10/2026** là ngày kiểm tra và đồng bộ trên máy.
- Chỉ chấm **fold 0**, không có std 5 fold và không có xác suất để đưa vào ensemble hiện tại.

| Hệ thống — cùng fold 0 | A macro-F1 | B macro-F1 |
|---|---:|---:|
| Qwen zero-shot | 0,6074 | 0,4612 |
| Qwen few-shot | 0,6927 | 0,5187 |
| TF-IDF comment | 0,7056 | 0,5245 |
| RoBERTa comment | 0,7314 | 0,6171 |
| Multi-task Twitter-RoBERTa, không class weight/bias | 0,7431 | 0,6111 |

Few-shot cải thiện Qwen, đặc biệt F1 implicit B từ **0,1611 → 0,3402**; F1 explicit giảm
0,5362 → 0,5132. Chưa có bằng chứng để thay hệ thống fine-tune bằng Qwen.
Input Qwen có title + description + comment; các baseline trong bảng dùng comment,
nên đây là so sánh cấu hình hệ thống, không cô lập tác động kiến trúc.

## 3. Kết quả supervised có thể dùng làm căn cứ phát triển
Các con số dưới đây được tính lại từ OOF đạt kiểm tra, là **mean ± std của 5 fold theo video**.
Đây là kết quả phát triển/CV; không gọi là điểm test độc lập của BTC.

| Cấu hình | A | B |
|---|---:|---:|
| TF-IDF comment | 0,7002 ± 0,0209 | 0,5249 ± 0,0138 |
| RoBERTa comment | 0,7274 ± 0,0098 | 0,5856 ± 0,0179 |
| Twitter-RoBERTa-hate comment | 0,7335 ± 0,0115 | 0,5870 ± 0,0114 |
| Multi-task cùng encoder, không cw/bias | 0,7354 ± 0,0111 | 0,6006 ± 0,0259 |

Context hiện chưa có lợi ổn định: A thêm title+description tăng 0,0014 nhưng thắng 3/5 fold;
B thêm title giảm 0,0048, thêm title+description giảm 0,0164; ghép thẳng giảm 0,0350.
Giữ comment-only làm cấu hình nền; giữ kết quả context như thí nghiệm đối chứng.

## 4. Các mốc đã xử lý và việc còn mở
| Mã | Trạng thái | Bằng chứng / bước tiếp |
|---|---|---|
| P1.1 | Xong | load/save/make OOF có kiểm tra; DeBERTa A/B failed_nonfinite, không được vào ứng viên |
| P1.2 | Xong | KEY thêm experiment/evaluation_scope; đã khôi phục 12 dòng 05 từ OOF, không train lại |
| P1.3 | Xong | Bỏ trọng số 0 trước nhân, từ chối NaN/ID lệch ở thành phần đang dùng; test và smoke trên dữ liệu thật đạt |
| P2 | Hiện tại | 0/30 mẫu lỗi B có ghi chú; user đọc tay, trợ lý hỗ trợ phân tích/kiểm tra |
| P3 | Chưa xong | Recipe cũ A top-3/B single chưa thay; cần so lại tất cả chiến lược/bias và cập nhật bảng/07 |
| P4 | Chưa xong | Không thấy checkpoint model; cần save/reload và kiểm tra suy luận phiên mới |
| P5 | Chưa xong | Viết báo cáo, xác minh yêu cầu/lịch BTC trước khi nộp |

Dry-run A/B có 599 ID hợp lệ mỗi file, macro-F1 lần lượt 0,7414 / 0,6111. Đây là kiểm tra
quy trình/file nộp trên fold 0 đã tham gia phát triển, không phải phép đánh giá độc lập.

## 5. File quan trọng
- `SemEval2027_Task8/data/splits/folds_en.csv`: split chuẩn, EN 2.989 mẫu / 243 nhóm video.
- `outputs/llm_predictions/`, `outputs/llm_cache/`: bằng chứng Qwen.
- `outputs/results/llm_results_fold0.csv`: 4 kết quả Qwen; `llm_comparison_fold0.csv`: 14 dòng so cùng fold.
- `outputs/results/all_results.csv`: **38 dòng** sau P1, có status; 2 failed +2 pending bị loại khi gọi load_results() mặc định.
- `outputs/results/nb05_restored.csv`: 12 dòng biến thể 05; `validated_results.csv`: 34 dòng hợp lệ.
- `outputs/results/summary_table.csv`: bảng trình bày cũ của notebook 08, chưa tái tạo sau P1; dùng bảng restored/validated để xem số liệu hiện hành.
- `outputs/audit/2026-10-18-p1/audit_after_p1.json`: bằng chứng hiện hành; `ensemble_smoke.json`: smoke greedy.
- `outputs/audit/2026-10-18-p1/before/`: bản sao code, notebook, bảng và tài liệu trước sửa.
- `outputs/audit/2026-10-18/audit_after_merge.json`: lịch sử audit P0.
- `outputs/audit/2026-10-18/all_results.before_llm_merge.csv`: bản sao 24 dòng trước merge.
- `scripts/audit_artifacts.py`: kiểm tra lại artifact, không train và không sửa đầu vào.
- `outputs/submissions/recipe.json`: công thức cũ cần tái xác nhận sau P1.
- `outputs/error_analysis/sample_30_B.csv`: mẫu đọc tay.

Các đường dẫn `outputs/...` và `scripts/...` ở trên tính từ `SemEval2027_Task8/`.

## 6. Quy tắc tiếp tục công việc
1. Giữ scope A/B EN; Persian, các ngôn ngữ khác và subtask C ngoài kế hoạch hiện tại.
2. Mọi thay đổi lựa chọn mô hình/ngưỡng phải tách dữ liệu chọn và dữ liệu chấm. Cách hậu xử lý
   trên 5-fold OOF hiện có chỉ là cross-fitting tầng quyết định; không tuyên bố toàn bộ pipeline
   là nested CV độc lập vì các base model của fold khác có thể đã học từ fold đang đánh giá.
3. Không kết luận từ `no/no` hay một điểm trung bình; xem từng lớp, chênh lệch từng fold và ví dụ lỗi.
4. Không chạy lại toàn bộ 01–08. P1 dùng OOF đã lưu; tách phần notebook 08 cần train trong dry-run ra khỏi phần chọn ensemble.
5. Không chạy `kaggle/nb06_tools/build_bundle.py` tùy tiện: bản hiện tại sinh lại notebook 06 và có thể ghi đè bản đã chạy. Sao lưu trước nếu cần đóng gói lại.
6. Mỗi mốc đã kiểm chứng: thêm worklog, đồng bộ handoff và ROADMAP.md (đã bao gồm pipeline).
   Không ghi việc dự định làm thành việc đã hoàn thành.
7. Không sửa file TSV dữ liệu gốc bằng Excel. Không chép nguyên bảng master cũ từ Kaggle đè lên kết quả mới.
8. Kiểm tra lịch/quy cách BTC từ nguồn chính thức trước khi nộp; lịch cũ trong roadmap chưa được xác minh lại trong phiên 18/10.


**Lưu ý môi trường:** bundle Kaggle cũ chưa chứa các sửa P1. Không chạy lại bundle cũ để ghi đè bảng mới.
