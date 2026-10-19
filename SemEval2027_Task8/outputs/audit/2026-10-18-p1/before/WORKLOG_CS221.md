# Worklog — Đồ án CS221 / StereoQueerEval
Nhật ký bắt đầu ngày **18/10/2026**. Các mốc cũ dưới đây được dựng lại từ artifact/handoff,
không phải nhật ký đã ghi tại thời điểm diễn ra.

Nguồn trạng thái hiện hành: [handoff](HANDOFF_CS221.md), [roadmap](SemEval2027_Task8/ROADMAP.md),
[pipeline](SemEval2027_Task8/PIPELINE.md).

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

## Mẫu ghi cho mốc tiếp theo
- Ngày và mục tiêu:
- Dữ liệu/cấu hình sử dụng:
- Thay đổi thực tế:
- Kiểm tra đã chạy và kết quả:
- Artifact/bằng chứng:
- Kết luận hoặc quyết định:
- Việc còn mở và mốc tiếp theo:

