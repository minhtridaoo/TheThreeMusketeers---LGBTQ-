# Worklog — Đồ án CS221 / StereoQueerEval
Nhật ký bắt đầu ngày **18/10/2026**. Các mốc cũ dưới đây được dựng lại từ artifact/handoff,
không phải nhật ký đã ghi tại thời điểm diễn ra.

Nguồn trạng thái hiện hành: [handoff](HANDOFF_CS221.md) và [ROADMAP đã gộp pipeline](SemEval2027_Task8/ROADMAP.md).

## Các mốc trước khi mở worklog
| Mốc | Bằng chứng đang có | Giới hạn |
|---|---|---|
| Đến 11/10/2026 | Handoff 11/10 và notebook 01–03 đã chạy; baseline TF-IDF/RoBERTa có OOF | Không suy ngày chạy riêng của 01/02 từ ngày sửa file |
| 13/10/2026 | Bảng results/run logs ghi kết quả 04/05/08; 07 có output phân tích lỗi | Chạy hết cell chưa có nghĩa mọi kết quả hợp lệ; lỗi DeBERTa được ghi bên dưới |
| 14/10/2026 | Journal Qwen từ 05:52:09 đến 06:07:03 UTC; status full/complete; 4×599 dự đoán | Đây là thời gian journal, không bao gồm tải model và toàn bộ setup |
| 18/10/2026 | User báo đã cập nhật kết quả notebook 06 và yêu cầu audit/kế hoạch/handoff/worklog | Ngày kiểm chứng/đồng bộ tại máy |

## 18/10/2026 — Kiểm tra 06, rà soát artifact và lập kế hoạch tiếp theo
**Yêu cầu:** xác nhận notebook 06 chạy xong; lập hướng đi/pipeline; cập nhật handoff và tạo worklog.

**Đã thực hiện**
- Đọc roadmap, handoff lịch sử, code liên quan và toàn bộ trạng thái execution của 8 notebook.
- Tạo `SemEval2027_Task8/scripts/audit_artifacts.py`; chạy kiểm tra trước/sau merge.
- Xác nhận notebook 06 full/complete, 5/5 code cell có output, 4 cấu hình ×599 mẫu, không failed attempt.
- Đối chiếu ID/fold/video/gold, JSON gốc, dự đoán, dấu vân tay cấu hình/prompt, few-shot không rò rỉ video;
  tính lại điểm và so bảng 14 dòng cùng fold 0: khớp.
- Quét 32 file OOF: 30 hợp lệ, 2 DeBERTa không hợp lệ; loss DeBERTa NaN ở 20 bản ghi epoch/task.
- Tái hiện lỗi NaN trong trung bình xác suất khi thành phần hỏng có trọng số 0; chưa sửa thuật toán.
- Xác nhận lỗi key ghi kết quả làm các biến thể notebook 05 đè nhau; chưa sửa schema/khôi phục bảng 05.
- Sao lưu bảng master 24 dòng; ghép 4 dòng Qwen thành **28 dòng**, kiểm tra giữ nguyên 24 dòng cũ.
- Kiểm tra 2 file dry-run: mỗi file đủ 599 ID/nhãn; ghi rõ điểm chỉ phục vụ kiểm tra quy trình.
- Xác nhận `sample_30_B.csv` có 30 dòng, 0 dòng được điền kiểu lỗi; chưa thấy checkpoint model trong dự án.
- Tạo handoff hiện hành, pipeline, báo cáo kiểm tra; cập nhật roadmap và gắn liên kết từ handoff cũ.

**Kết quả 06 đã xác nhận**
| Chế độ | A macro-F1 fold 0 | B macro-F1 fold 0 |
|---|---:|---:|
| Zero-shot | 0,607411 | 0,461242 |
| Few-shot | 0,692715 | 0,518695 |

Few-shot tăng A +0,085305, B +0,057453. F1 implicit B tăng 0,161137 → 0,340176;
F1 explicit giảm 0,536232 → 0,513208. Chưa vượt TF-IDF/transformer trong so sánh cùng fold 0.

**File/evidence**
- [Báo cáo kiểm tra](SemEval2027_Task8/outputs/audit/2026-10-18/REVIEW.md)
- [Audit trước merge](SemEval2027_Task8/outputs/audit/2026-10-18/audit_before_merge.json)
- [Audit sau merge](SemEval2027_Task8/outputs/audit/2026-10-18/audit_after_merge.json)
- [Bản sao master cũ](SemEval2027_Task8/outputs/audit/2026-10-18/all_results.before_llm_merge.csv)

**Quyết định**
- Đóng mốc chạy/kiểm chứng 06; giữ Qwen là thí nghiệm đối chiếu fold 0.
- Ưu tiên sửa tính hợp lệ và bảng báo cáo trước khi mở thêm thí nghiệm GPU.
- Chọn comment-only làm hướng nền hiện tại; tiếp tục đánh giá multi-task và ensemble sau khi loại đầu vào hỏng.
- Kế hoạch tiếp theo P1 → P2 → P3 → P4 → P5 nằm trong pipeline; chưa đánh dấu các bước đó là hoàn thành.

**Chưa thực hiện:** train model mới; chạy lại 07/08; sửa DeBERTa; sửa `results.py`/`decision.py`;
viết thay kết luận notebook; gán nhãn lỗi thủ công; xác minh lịch BTC mới; nộp cuộc thi.

## 18/10/2026 — Gộp tài liệu và hoàn thành P1
**Yêu cầu:** gộp pipeline/roadmap, tiếp tục công việc, chỉ rõ vị trí hiện tại và tốc độ so kế hoạch.

**Đã thực hiện**
- Gộp mục tiêu, sơ đồ pipeline, bản đồ notebook, kết quả, mốc P0–P5, tiêu chí và thông tin BTC vào ROADMAP.md.
- PIPELINE.md chỉ giữ liên kết chuyển tiếp; sao lưu nguyên bản kế hoạch/handoff/worklog trước sửa.
- Thêm validation OOF: kiểm tra ID, nhãn, gold/fold/video chuẩn, xác suất hữu hạn/tổng hàng/argmax.
- Sửa average: bỏ trọng số 0 trước nhân; từ chối NaN/Inf, trọng số sai hoặc thứ tự ID/gold/fold/video khác nhau.
- Thêm khóa experiment/evaluation_scope cho logger; không làm mất lịch sử failed khi log kết quả mới.
- Phục hồi 12 dòng 05 từ OOF; master 38 dòng (34 valid, 2 failed DeBERTa, 2 recipe cũ pending).
- Chỉnh 05/07/08 dùng logic mới; xóa output phụ thuộc code thay đổi và lưu bản notebook đã chạy trước sửa.
- Điều chỉnh test import Qwen để không phụ thuộc master chưa từng có kết quả 06.
- Chạy **35 test: đều đạt**. Smoke greedy với 9 ứng viên hợp lệ/task: A 0,745465, B 0,595965.
- Audit lại: Qwen vẫn hợp lệ; 47 hash artifact gốc không đổi; không train model hay thay recipe.

**Bằng chứng:** [báo cáo P1](SemEval2027_Task8/outputs/audit/2026-10-18-p1/P1_REVIEW.md),
[audit sau P1](SemEval2027_Task8/outputs/audit/2026-10-18-p1/audit_after_p1.json),
[smoke ensemble](SemEval2027_Task8/outputs/audit/2026-10-18-p1/ensemble_smoke.json).

**Quyết định/trạng thái:** P0/P1 xong, chuyển **P2** đọc lỗi và viết kết luận.
P3 vẫn cần so lại đầy đủ single/top-3/greedy/bias và cập nhật recipe/summary/07;
smoke P1 không thay cho việc chọn hệ thống cuối. P4/P5 còn checkpoint và báo cáo.

**Tốc độ:** phần chạy thí nghiệm đi trước lịch gốc (bước 3–6 vốn dự kiến đến tháng 11);
chưa thể đánh giá dư/thiếu thời gian so hạn môn khi chưa có ngày nộp.
Không quy “8/8 notebook đã từng chạy” thành tỷ lệ hoàn thành đồ án.

## 18/10/2026 — Kiểm tra phần đọc tay và tổng hợp P2
**Yêu cầu:** người dùng đã đọc xong sample_30_B, kiểm tra công việc và cập nhật tiến độ.

**Đã kiểm tra/thực hiện**
- 30/30 ghi chú không rỗng, 30 ID riêng biệt; title/comment khớp TSV, gold/video/prediction khớp split/OOF/errors_B. Không có sai lệch trường nguồn.
- Xác nhận ba nhóm 10 mẫu; đây là mẫu chọn theo nhóm lỗi, không dùng để ước lượng tỷ lệ lỗi toàn tập.
- Đọc nhận xét và đối chiếu định nghĩa B trong guideline cục bộ trang 2–3. Góp ý riêng: model comment-only không nhận title; nguyên nhân lỗi cần trình bày như giả thuyết; không quy implicit/explicit chỉ bằng từ khóa hoặc mức gay gắt.
- Tổng hợp 8 ví dụ tiêu biểu và kết luận context/MT/LLM trong P2_REVIEW.md; chưa chép vào notebook/báo cáo môn.
- Đối chiếu 7 ID thuộc fold 0: Qwen zero-shot đúng 2, few-shot đúng 4. Các mẫu chọn từ lỗi MT, không dùng tỷ lệ này để xếp hạng hệ thống.
- Sao lưu nguyên trạng sample_30_B.csv và tài liệu; lưu audit JSON có hash nguồn. Không sửa ghi chú, nhãn, dự đoán, recipe hay code model.
- Phát hiện cell xuất lỗi 07 ghi đè kieu_kho thành rỗng khi chạy lại; ghi yêu cầu bảo toàn ghi chú theo nguồn trong handoff/P3, chưa sửa cell.
- Đồng bộ ROADMAP và HANDOFF: P2 đọc/kiểm tra/tổng hợp xong, P3 là bước kế tiếp, chưa thực hiện lựa chọn mới.

**Bằng chứng:** [báo cáo P2](SemEval2027_Task8/outputs/error_analysis/P2_REVIEW.md),
[audit mẫu và hash](SemEval2027_Task8/outputs/audit/2026-10-18-p2/sample_review.json).

**Còn lại:** P3 so single/top-3/greedy/bias và đồng bộ 07/summary; P4 save/reload/checkpoint;
P5 đưa phần tổng hợp vào báo cáo và xác minh yêu cầu BTC trước khi nộp. Không train mới trong lượt này.

## 18/10/2026 — Kiểm tra lại P2 và hoàn thành phần kỹ thuật P3
**Yêu cầu:** kiểm tra bản Excel/CSV mới; nếu ổn chuyển P3 và hướng dẫn những việc cần làm.

**Đã thực hiện/kiểm chứng**
- 30/30 ghi chú, 9 ghi chú góp ý đã sửa; trường nguồn khớp. Giữ nguyên byte CSV người dùng.
- Thêm selection.py và script CPU run_p3_selection.py. So 6 chiến lược trên 9 ứng viên hợp lệ/task; loại DeBERTa NaN.
- Sửa cách fit bias: cùng thành phần/trọng số của lượt đang xét, fit trên 4 fold ngoài fold được chấm. Ghi rõ chưa là nested CV độc lập toàn pipeline.
- Theo quy tắc gain ≥0,005, chọn A greedy+bias **0,746020 ±0,010278**, B top-3 mean+bias **0,607455 ±0,012700**. B implicit F1 0,4313 →0,4631.
- Lưu 12 dòng chiến lược, 60 điểm theo fold, 60 quyết định fit, OOF được chọn, recipe có hash/config/label order; cập nhật master/validated/summary.
- Master 40 dòng =36 valid +2 failed +2 superseded. 12 dòng 05 và 4 dòng Qwen giữ nguyên.
- 38 test đạt, gồm đổi held-out gold không đổi tham số/dự đoán fit của fold đó và bảo toàn ghi chú.
- Notebook 08 đã chạy 9 code cell, 07 chạy 12 code cell, không lỗi. Đã xem hình confusion matrix mới.
- 07 xuất lỗi P3 vào thư mục riêng theo hash nguồn; chạy lại giữ ghi chú cùng nguồn, từ chối nguồn khác. Không ghi đè 30 ghi chú P2.
- 54 artifact nguồn không đổi. 08 train/test mặc định tắt; không chạy training, checkpoint hay submission.
- Viết hướng dẫn P3, đoạn kết luận mẫu; cập nhật ROADMAP/HANDOFF. P3 kỹ thuật xong, P4 chưa chạy.

**Bằng chứng:** [báo cáo và hướng dẫn P3](SemEval2027_Task8/outputs/selection/P3_REVIEW.md),
[validation](SemEval2027_Task8/outputs/audit/2026-10-18-p3/validation.json),
[manifest](SemEval2027_Task8/outputs/selection/manifest.json).

**Việc tiếp theo:** người dùng đọc mục 1–5 notebook 08, bảng lựa chọn và giải thích đánh đổi. P4 bổ sung save/reload theo recipe rồi kiểm tra suy luận phiên mới; P5 viết báo cáo. Điểm P3 là phát triển, không phải điểm test độc lập.

## Mẫu ghi cho mốc tiếp theo
- Ngày và mục tiêu:
- Dữ liệu/cấu hình sử dụng:
- Thay đổi thực tế:
- Kiểm tra đã chạy và kết quả:
- Artifact/bằng chứng:
- Kết luận hoặc quyết định:
- Việc còn mở và mốc tiếp theo:

