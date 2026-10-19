# Handoff hiện hành — CS221 / StereoQueerEval
Cập nhật và kiểm chứng: **18/10/2026**. Phạm vi: **Subtask A + B, English**.

Đọc file này khi tiếp tục công việc. [ROADMAP duy nhất](SemEval2027_Task8/ROADMAP.md) đã gộp pipeline, tiến độ và kế hoạch;
[worklog](WORKLOG_CS221.md) ghi lịch sử đã kiểm chứng. PIPELINE.md chỉ còn liên kết chuyển tiếp.

## 1. Trạng thái và hành động tiếp theo
- **P0/P1/P2/P3 đã xong. P4 đã chuẩn bị code/bundle, chờ user chạy Kaggle smoke rồi dryrun.**
- Người dùng đã sửa 9 ghi chú được góp ý. 30/30 mẫu khớp nguồn, không sửa gold/prediction.
- P3 so 6 chiến lược/task, 9 ứng viên hợp lệ/task. Thành phần/trọng số/bias cùng được fit trên 4 fold ngoài fold đang chấm.
- **A greedy+bias: 0,746020 ± 0,010278; B top-3 mean+bias: 0,607455 ± 0,012700.** Là điểm phát triển, không phải test độc lập.
- Quy tắc gain ≥0,005 so với single tốt nhất có/không bias. A hơn 0,0122, B hơn 0,0068.
- B F1 implicit tăng 0,4313 → 0,4631; F1 no giảm, explicit gần như giữ nguyên.
- Master 40 dòng: 36 completed_valid, 2 failed_nonfinite, 2 superseded_p3. 12 dòng 05 và 4 dòng Qwen giữ nguyên.
- **43 test đạt** sau chuẩn bị P4. Notebook 08/07 đã chạy CPU ở P3, lần lượt 9/12 code cell, 0 lỗi. 05 chưa train/chạy lại toàn notebook.
- 54 artifact nguồn không đổi. CSV P2 giữ nguyên; 07 xuất lỗi P3 vào thư mục theo hash, chạy lại cùng nguồn giữ ghi chú.
- Đã bổ sung checkpoint.py/p4.py/run_p4.py và runner Kaggle riêng. Chỉ train model nhỏ để test CPU; chưa train lại model thật.
- Runner P4: smoke 64/16 (1 epoch) → dryrun 2390/599 (4 epoch); final 2989/32 probe là lượt riêng khi cần.
- Có 7 component: 5 transformer +2 TF-IDF; MT/MTcw AB dùng chung. Resume theo model đã xong, chưa resume giữa epoch.
- 08 vẫn RUN_P4_DRYRUN=False/RUN_OFFICIAL_TEST=False; dùng runner P4 để có lưu/nạp checkpoint.
- Hành động user ngay: upload bundle P4 Private, import notebook, bật GPU/Internet, chạy hai ô đầu và gửi log.
- Việc người dùng cần đọc: mục 1–5 notebook 08, bảng p3_strategies/p3_folds, [hướng dẫn P3](SemEval2027_Task8/outputs/selection/P3_REVIEW.md).
- Phần thí nghiệm đi trước lịch gốc; chưa biết hạn môn nên không kết luận dư/thiếu thời gian tổng thể.

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
| P2 | Xong phần đọc/kiểm tra/tổng hợp | 30/30 user đã đọc; 8 ví dụ, đối chiếu 7 mẫu Qwen và kết luận trong P2_REVIEW.md |
| P3 | Xong phần kỹ thuật | Recipe A greedy+bias/B top-3+bias; bảng/07/08/summary và hash đã kiểm chứng |
| P4 | Đã chuẩn bị, chưa có output Kaggle | 43 test CPU đạt; cần smoke rồi dryrun T4, 7 checkpoint và 599 ID/task, tải về kiểm chứng |
| P5 | Chưa xong | Viết báo cáo, xác minh yêu cầu/lịch BTC trước khi nộp |

Dry-run cũ A/B có 599 ID hợp lệ/file, điểm 0,7414 / 0,6111. Đây là kiểm tra quy trình của recipe cũ,
không phải kết quả recipe P3 mới hoặc test độc lập. Cần dry-run sau save/reload ở P4.

## 5. File quan trọng
- `SemEval2027_Task8/data/splits/folds_en.csv`: split chuẩn, EN 2.989 mẫu / 243 nhóm video.
- `outputs/llm_predictions/`, `outputs/llm_cache/`: bằng chứng Qwen.
- `outputs/results/llm_results_fold0.csv`: 4 kết quả Qwen; `llm_comparison_fold0.csv`: 14 dòng so cùng fold.
- `outputs/results/all_results.csv`: **40 dòng** sau P3; 2 failed +2 superseded bị loại mặc định.
- `outputs/results/nb05_restored.csv`: 12 dòng biến thể 05; `validated_results.csv`: 36 dòng hợp lệ.
- `outputs/results/summary_table.csv`: đã tái tạo P3, có evaluation_scope và các biến thể/nguồn.
- `outputs/audit/2026-10-18-p1/audit_after_p1.json`: bằng chứng hiện hành; `ensemble_smoke.json`: smoke greedy.
- `outputs/audit/2026-10-18-p1/before/`: bản sao code, notebook, bảng và tài liệu trước sửa.
- `outputs/audit/2026-10-18/audit_after_merge.json`: lịch sử audit P0.
- `outputs/audit/2026-10-18/all_results.before_llm_merge.csv`: bản sao 24 dòng trước merge.
- `scripts/audit_artifacts.py`: kiểm tra lại artifact, không train và không sửa đầu vào.
- `outputs/submissions/recipe.json`: recipe P3 có thứ tự nhãn, trọng số/bias, config/hash và phạm vi điểm.
- `outputs/results/p3_strategies.csv`, `p3_folds.csv`, `p3_candidates.csv`: bảng lựa chọn hiện hành.
- `outputs/selection/manifest.json`, `fold_decisions.json`, `selected_A.csv`, `selected_B.csv`: bằng chứng P3.
- `outputs/selection/P3_REVIEW.md`: giải thích lựa chọn và hướng dẫn người dùng đọc kết quả.
- `outputs/audit/2026-10-18-p3/validation.json`: 38 test, notebook execution, kiểm tra 54 hash nguồn.
- `scripts/run_p3_selection.py`: tái tạo P3 bằng CPU, không train.
- `outputs/error_analysis/sample_30_B.csv`: 30/30 ghi chú người dùng; không ghi đè khi tái tạo phân tích.
- `outputs/error_analysis/P2_REVIEW.md`: góp ý cách diễn giải, 8 ví dụ, đối chiếu Qwen và kết luận thực nghiệm.
- `outputs/audit/2026-10-18-p2/sample_review.json`: kết quả kiểm tra và hash; `before/` giữ CSV/tài liệu trước cập nhật.

- `../kaggle/kaggle_runner_p4.ipynb`, `../kaggle/sqe_p4_bundle.zip`: notebook/bundle riêng cho P4.
- [Hướng dẫn P4](kaggle/HUONG_DAN_KAGGLE_P4.md): từng bước smoke/dryrun/final và tải output.
- `outputs/audit/2026-10-18-p4-preparation/validation.json`: bằng chứng 43 test/ZIP hợp lệ, chưa chạy T4.

Các đường dẫn `outputs/...` và `scripts/...` ở trên tính từ `SemEval2027_Task8/`.

## 6. Quy tắc tiếp tục công việc
1. Giữ scope A/B EN; Persian, các ngôn ngữ khác và subtask C ngoài kế hoạch hiện tại.
2. Mọi thay đổi lựa chọn mô hình/ngưỡng phải tách dữ liệu chọn và dữ liệu chấm. Cách hậu xử lý
   trên 5-fold OOF hiện có chỉ là cross-fitting tầng quyết định; không tuyên bố toàn bộ pipeline
   là nested CV độc lập vì các base model của fold khác có thể đã học từ fold đang đánh giá.
3. Không kết luận từ `no/no` hay một điểm trung bình; xem từng lớp, chênh lệch từng fold và ví dụ lỗi.
4. Không chạy lại toàn bộ 01–08. P3 dùng OOF đã lưu; train/test trong 08 đã được tách và mặc định tắt. Các hàm train cũ ở 08 chưa lưu checkpoint; runner P4 riêng đã hỗ trợ lưu/nạp nhưng chưa chạy Kaggle thật.
5. Không chạy `kaggle/nb06_tools/build_bundle.py` tùy tiện: bản hiện tại sinh lại notebook 06 và có thể ghi đè bản đã chạy. Sao lưu trước nếu cần đóng gói lại.
6. Mỗi mốc đã kiểm chứng: thêm worklog, đồng bộ handoff và ROADMAP.md (đã bao gồm pipeline).
   Không ghi việc dự định làm thành việc đã hoàn thành.
7. Không sửa file TSV dữ liệu gốc bằng Excel. Không chép nguyên bảng master cũ từ Kaggle đè lên kết quả mới.
8. Kiểm tra lịch/quy cách BTC từ nguồn chính thức trước khi nộp; lịch cũ trong roadmap chưa được xác minh lại trong phiên 18/10.


**Lưu ý môi trường:** bundle Kaggle cũ chưa chứa các sửa P1/P3. Không chạy lại bundle cũ để ghi đè bảng mới.
