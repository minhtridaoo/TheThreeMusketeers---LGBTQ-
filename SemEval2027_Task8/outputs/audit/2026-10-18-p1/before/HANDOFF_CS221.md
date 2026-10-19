# Handoff hiện hành — CS221 / StereoQueerEval
Cập nhật và kiểm chứng: **18/10/2026**. Phạm vi: **Subtask A + B, English**.

Đọc file này khi tiếp tục công việc. [Roadmap](SemEval2027_Task8/ROADMAP.md) ghi các mốc;
[pipeline](SemEval2027_Task8/PIPELINE.md) ghi luồng xử lý và tiêu chí hoàn thành;
[worklog](WORKLOG_CS221.md) ghi lịch sử đã kiểm chứng. Handoff ngày 11/10 được giữ làm bản lịch sử.

## 1. Trạng thái và hành động tiếp theo
- Cả 8 notebook đều có execution count ở mọi code cell, không có error output lưu trong notebook.
- **Notebook 06 đã hoàn thành full và đã kiểm chứng**, không cần chạy lại Qwen.
- **Ưu tiên tiếp theo: P1 trong pipeline — làm sạch kết quả và chạy lại riêng phần chọn ensemble trên OOF hợp lệ.**
- Chưa chốt hệ thống cuối: 2 file OOF DeBERTa hỏng, bảng kết quả notebook 05 bị ghi đè biến thể,
  và nhánh greedy notebook 08 bị nhiễm NaN. Chưa sửa ba vấn đề này trong phiên kiểm tra 18/10.
- 30 mẫu lỗi B đã được xuất nhưng cột `kieu_kho` còn trống **0/30**.
- Mục kết luận notebook 04–08 còn là nội dung chờ viết.
- Hạn môn học chưa được xác nhận. Lịch tiếp theo dùng số buổi, không giả định ngày nộp.

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

## 4. Những việc phải xử lý trước khi chốt báo cáo
| Mã | Bằng chứng | Hành động và điều kiện xong |
|---|---|---|
| P1.1 | 30/32 OOF hợp lệ; DeBERTa A/B có toàn bộ xác suất NaN, loss NaN ở 20 epoch/fold-record mỗi task | Đánh dấu failed, loại khỏi ứng viên bằng kiểm tra dữ liệu; không coi điểm majority là kết quả DeBERTa thành công |
| P1.2 | `results.py` dùng KEY notebook/model/input/subtask; notebook 05 log nhiều biến thể cùng key, hiện chỉ còn 2 dòng | Thêm experiment/run_id, khôi phục bảng 12 biến thể single/MT/MT+cw × raw/bias × A/B từ OOF, không train lại |
| P1.3 | `average([good,bad], weights=[1,0])` vẫn toàn NaN; notebook 08 từng cho greedy điểm majority | Chặn OOF hỏng, kiểm tra thứ tự ID, loại trọng số 0 trước phép nhân; chạy lại lựa chọn ensemble từ OOF sạch |
| P2 | 0/30 mẫu lỗi B có ghi chú | User đọc và gán kiểu lỗi, đối chiếu vài mẫu Qwen few-shot và model chính trên cùng fold 0 |
| P3 | Recipe cũ: A top-3 mean 0,7419; B best single 0,6006 | Chỉ chốt lại sau P1; các điểm cũ còn ở file để truy vết, không mặc định sai nhưng chưa đủ để khẳng định phương án tốt nhất |
| P4 | Không thấy checkpoint `.pt/.pth/.safetensors/.bin` trong dự án | Bổ sung lưu/reload model, tokenizer, cấu hình và nhãn; kiểm tra suy luận ở phiên mới trước khi gọi hệ thống sẵn sàng |

Dry-run A/B có 599 ID hợp lệ mỗi file, macro-F1 lần lượt 0,7414 / 0,6111. Đây là kiểm tra
quy trình/file nộp trên fold 0 đã tham gia phát triển, không phải phép đánh giá độc lập.

## 5. File quan trọng
- `SemEval2027_Task8/data/splits/folds_en.csv`: split chuẩn, EN 2.989 mẫu / 243 nhóm video.
- `outputs/llm_predictions/`, `outputs/llm_cache/`: bằng chứng Qwen.
- `outputs/results/llm_results_fold0.csv`: 4 kết quả Qwen; `llm_comparison_fold0.csv`: 14 dòng so cùng fold.
- `outputs/results/all_results.csv`: **28 dòng** sau merge; vẫn chứa 2 kết quả DeBERTa lỗi và các hạn chế P1.2/P1.3.
- `outputs/results/summary_table.csv`: bảng cũ của notebook 08, **chưa đồng bộ Qwen và chưa sửa mất biến thể MT**.
- `outputs/audit/2026-10-18/audit_after_merge.json`: bằng chứng máy đọc, hash đầu ra và các kiểm tra.
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
6. Mỗi mốc đã kiểm chứng: thêm một mục vào worklog, đồng bộ trạng thái handoff/roadmap/pipeline.
   Không ghi việc dự định làm thành việc đã hoàn thành.
7. Không sửa file TSV dữ liệu gốc bằng Excel. Không chép nguyên bảng master cũ từ Kaggle đè lên kết quả mới.
8. Kiểm tra lịch/quy cách BTC từ nguồn chính thức trước khi nộp; lịch cũ trong roadmap chưa được xác minh lại trong phiên 18/10.

