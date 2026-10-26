# P3 — Chọn hệ thống cuối từ OOF đã lưu

Kiểm chứng ngày **18/10/2026**. Scope A/B English. **P3 phần kỹ thuật đã hoàn thành**, chưa train mới, chưa lưu checkpoint hoặc nộp test. P4 là bước tiếp theo.

## 1. Bản ghi chú P2 đã ổn

Người dùng đã sửa 9 ghi chú được góp ý: mẫu 1, 12, 14, 16, 20, 21, 22, 23, 28. Đã kiểm tra lại 30/30 ghi chú, ID và trường nguồn không lệch. Cách diễn đạt về title, giả thuyết nguyên nhân và ranh giới nhãn đã được chỉnh. Mẫu 20 có 134 từ khi tách theo khoảng trắng; đó không phải số token model.

## 2. P3 giải quyết việc gì?

Từ các model đã chạy, chọn cách kết hợp và ngưỡng quyết định để dùng tiếp ở P4. OOF là dự đoán cho các mẫu không nằm trong train fold của model tạo ra dự đoán đó.

- **Best single:** trên 4 fold chọn một model, dùng cho fold còn lại.
- **Top-3 mean:** chọn ba model rồi lấy trung bình xác suất.
- **Greedy:** lần lượt thêm thành phần khi điểm trên dữ liệu dùng để chọn tăng; một model có thể được chọn lặp lại để tăng trọng số.
- **Bias:** điều chỉnh điểm log xác suất của từng nhãn trước khi lấy nhãn cao nhất. Bias cũng chỉ được học từ 4 fold ngoài fold đang chấm.

Có **9 ứng viên hợp lệ/task**, loại DeBERTa NaN. Không đưa random/majority, bản TF-IDF trùng, OOF đã chỉnh bias hoặc Qwen chỉ có nhãn fold 0 vào nhóm ứng viên này.

Đã sửa cách xử lý bias: mỗi fold giữ cố định lựa chọn thành phần của chính lượt đó, fit bias trên 4 fold còn lại rồi mới chấm. Không tiếp tục nối nested_bias lên một bảng ensemble đã được chọn riêng ở các fold khác như trước.

## 3. Kết quả so sánh

Macro-F1 **trung bình ± độ lệch chuẩn của 5 fold**; các quyết định chọn thành phần/trọng số/bias được cross-fit trên OOF hiện có.

| Chiến lược | A | B |
|---|---:|---:|
| Best single | 0,7323 ± 0,0146 | 0,6006 ± 0,0259 |
| Best single + bias | 0,7338 ± 0,0143 | 0,5962 ± 0,0160 |
| Top-3 mean | 0,7419 ± 0,0102 | 0,6043 ± 0,0164 |
| Top-3 mean + bias | 0,7396 ± 0,0095 | **0,6075 ± 0,0127** |
| Greedy | 0,7455 ± 0,0142 | 0,5960 ± 0,0149 |
| Greedy + bias | **0,7460 ± 0,0103** | 0,6015 ± 0,0196 |

Quy tắc đã đặt trước: dùng ensemble nếu hơn phương án single tốt nhất (có/không bias) ít nhất **0,005 macro-F1**. A tăng 0,0122 so với single+bias, B tăng 0,0068 so với single. Cả hai qua ngưỡng.

A thắng best single không bias ở 4/5 fold, nhưng giảm ở fold 0. B thắng ở 3/5 fold, giảm ở fold 1 và 3. Số fold thắng chỉ mô tả dữ liệu, không chứng minh ý nghĩa thống kê.

Với B, F1 implicit tăng **0,4313 → 0,4631**; F1 no giảm **0,7501 → 0,7391**, explicit gần như giữ nguyên **0,6204 → 0,6202**. Chọn theo macro-F1 tổng thể, không đòi mọi lớp đều tăng.

Lưu ý: dòng best single A là quy trình chọn model trên 4 fold và chấm fold còn lại, nên không nhất thiết bằng điểm của một model cố định như MT 0,7354 trong bảng trước.

## 4. Recipe dùng cho P4

Các tham số sau được fit trên **toàn bộ OOF sau khi chọn chiến lược**, và lưu trong [recipe.json](../submissions/recipe.json).

| Task | Thành phần | Trọng số khi lấy trung bình |
|---|---|---:|
| A | mt_twhate_comment | 1/10 |
| A | mtcw_twhate_comment | 1/10 |
| A | roberta_pair_title | 1/10 |
| A | roberta_pair_title_desc | 1/10 |
| A | tfidf_comment | 4/10 |
| A | tfidf_comment_title | 2/10 |
| B | mt_twhate_comment | 1/3 |
| B | mtcw_twhate_comment | 1/3 |
| B | twhate_comment | 1/3 |

Bias A theo thứ tự no/yes là **[0, 0]**: sau fit toàn OOF, thuật toán không cần dịch điểm, dù bias ở từng lượt cross-fit có thể khác 0. Bias B theo no/implicit/explicit là `[0.0, 0.2, 0.1]`.

Recipe A có cả model dùng context và TF-IDF. Việc một thành phần tham gia ensemble không chứng minh nó tốt nhất khi chạy riêng, hay context luôn có ích; quyết định dựa trên hiệu quả kết hợp. Hai model MT được dùng chung giữa A/B. P4 cần triển khai đúng thành phần, trọng số, label order và config đã lưu.

**Giới hạn đánh giá:** đây là cross-fitting ở tầng quyết định trên OOF phát triển, không phải nested CV độc lập cho toàn bộ quá trình train/chọn model. Các chiến lược cuối cũng được chọn bằng điểm này. Điểm đánh giá các lượt fit theo fold, không phải điểm test độc lập của recipe fit toàn OOF. Qwen fold 0 được giữ riêng trong bảng báo cáo.

## 5. Bạn cần làm gì để nắm và bàn giao P3?

1. Mở [notebook 08](../../notebooks/08_ensemble_submission.ipynb), đọc **mục 1–5**. Notebook đã có output mới. Giữ `RECOMPUTE_P3=False` khi chỉ xem; P4 dry-run/test mặc định tắt.
2. Mở [p3_strategies.csv](../results/p3_strategies.csv) bằng Excel: tìm hai dòng selected=True; xem macro_f1, std, delta và F1 implicit của B. Dùng [p3_folds.csv](../results/p3_folds.csv) để kiểm tra fold nào tăng/giảm.
3. Đọc mục 4 của tài liệu này để hiểu recipe. Không cần chỉnh nhãn hay train lại khi đọc bảng.
4. Mở [notebook 07](../../notebooks/07_error_analysis.ipynb), xem confusion matrix và phần kết luận đã cập nhật. Ghi chú 30 mẫu P2 vẫn nguyên vẹn; mẫu P3 mới nằm ở thư mục riêng, không yêu cầu đọc lại 30 mẫu để đóng P3.
5. Khi trình bày, dùng đoạn kết luận bên dưới và nêu rõ giới hạn CV. Các điều kiện kỹ thuật đóng P3 đã kiểm chứng ở mục 6.

Đoạn kết luận có thể dùng:

> Sau khi loại OOF không hợp lệ, chúng tôi so sánh sáu chiến lược trên chín ứng viên cho mỗi subtask. Việc chọn thành phần, trọng số và bias được thực hiện trên các fold ngoài fold đang chấm. Theo quy tắc ưu tiên hệ thống đơn giản nếu ensemble tăng dưới 0,005 macro-F1, chúng tôi chọn greedy+bias cho A (0,7460 ± 0,0103) và top-3 mean+bias cho B (0,6075 ± 0,0127). Ở B, F1 implicit tăng từ 0,4313 lên 0,4631, với sự đánh đổi ở lớp no. Đây là kết quả phát triển từ OOF, chưa phải đánh giá test độc lập.

Nếu muốn tự tái tạo P3 bằng CPU, chạy từ thư mục `SemEval2027_Task8`:

```powershell
C:/Python314/python.exe -X utf8 scripts/run_p3_selection.py
```

Sau đó chạy mục 1–5 của 08 để đọc kết quả, và Run All notebook 07 để tái tạo phân tích. Không cần Kaggle cho P3. Trong 08, các bước train/dự đoán P4 chỉ chạy khi chủ động bật cờ tương ứng; chúng hiện chưa bổ sung save/reload nên chưa phải P4 hoàn chỉnh.

## 6. Bằng chứng hoàn thành

- 12 dòng chiến lược, 60 dòng điểm theo fold, 60 bản ghi tham số fit; fold đang chấm không nằm trong fit_folds.
- Recipe có hash nguồn, thứ tự nhãn, config thành phần và phạm vi điểm. OOF được chọn đủ 2.989 ID/task, nhãn/fold/video/xác suất hợp lệ.
- **38 test đạt**, gồm kiểm tra thay gold ở fold giữ lại không đổi tham số/dự đoán fit cho fold đó và bảo toàn ghi chú khi xuất lại.
- Notebook 08: 9 code cell đã chạy; notebook 07: 12 code cell đã chạy; 0 lỗi. Không bật train/test P4.
- Đã cập nhật recipe/master/validated/summary. Master **40 dòng** = 36 completed_valid, 2 failed_nonfinite, 2 superseded_p3. Hai recipe cũ được giữ lịch sử, không còn là kết quả hiện hành.
- 54 artifact nguồn không đổi; 12 dòng notebook 05 và 4 dòng Qwen được bảo toàn. File ghi chú P2 giữ nguyên byte.
- Output lỗi P3 ở [thư mục riêng](../error_analysis/p3/2904d685b241/); cùng nguồn chạy lại sẽ giữ nguyên ghi chú, khác nguồn phải dùng thư mục khác.

Nguồn kiểm chứng: [validation.json](../audit/2026-10-18-p3/validation.json), [manifest.json](manifest.json), [fold_decisions.json](fold_decisions.json), [summary_table.csv](../results/summary_table.csv). Bản trước thay đổi nằm trong `../audit/2026-10-18-p3/before/`.

**Bước tiếp theo P4:** bổ sung lưu/nạp các model thành phần, kiểm tra dự đoán sau nạp ở phiên mới, rồi train cấu hình chốt khi cần. Chưa chạy bước này trong P3.
