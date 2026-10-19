# P1 — sửa tính hợp lệ và phục hồi bảng kết quả
Ngày: 18/10/2026. Tiếp theo audit P0 cùng ngày. Không train lại model, không thay dự đoán gốc.

## Thay đổi
- Thêm `src/sqe/validation.py`: kiểm tra ID, nhãn, fold, video, xác suất và thứ tự giữa các hệ thống.
- `results.load_oof` kiểm tra với dữ liệu chuẩn trước trả về; `make_oof` chặn NaN trước argmax.
- `decision.average` bỏ trọng số 0 trước nhân; thành phần có trọng số dương hỏng hoặc lệch ID thì dừng.
  Greedy/bias cũng kiểm tra đầu vào, không tính điểm trên xác suất NaN.
- Khóa logger thêm experiment/evaluation_scope; đọc bảng mặc định loại failed/pending khỏi chọn model.
- Phục hồi 12 dòng notebook 05 từ OOF hiện có, thêm tên/hash nguồn; giữ 4 dòng Qwen.
- Đánh dấu 2 dòng DeBERTa failed_nonfinite và xóa điểm giả majority trong master;
  2 dòng lựa chọn 08 cũ pending_revalidation.
- Notebook 05 dùng experiment riêng; 07/08 bỏ qua OOF không hợp lệ và in lý do.
  Các output phụ thuộc code thay đổi được xóa, bản cũ đã sao lưu. Chưa chạy lại notebook 05/07/08.
- Gộp roadmap/pipeline thành ROADMAP.md; PIPELINE.md chỉ còn liên kết chuyển tiếp.

## Kiểm tra
- **35 test đạt**: xử lý NaN/Inf, trọng số 0/âm/sai chiều, ID/gold/fold/video lệch,
  xác suất sai tổng/argmax, kiểm tra dữ liệu chuẩn, bảo toàn các experiment và lịch sử failed,
  cộng các test cache/export/import Qwen.
- Smoke greedy trên 9 ứng viên hợp lệ/task, bỏ DeBERTa: A **0,745464806**,
  B **0,595964715** macro-F1 mean 5 fold; không còn xác suất NaN.
- Đây chỉ là smoke xác nhận P1; chưa so lại đầy đủ các chiến lược để chốt recipe ở P3.
- 47 hash của OOF/cache/prediction gốc giữ nguyên.
- Audit sau sửa: Qwen vẫn hợp lệ 4×599, bảng so fold 0 khớp;
  OOF vật lý vẫn 30 hợp lệ +2 failed được giữ làm bằng chứng.
- Master hiện **38 dòng**: 34 completed_valid, 2 failed_nonfinite, 2 pending_revalidation.
- `nb05_restored.csv` có 12 dòng, không còn ghi đè biến thể.
- `validated_results.csv` có 34 dòng; đây là các kết quả đã qua kiểm tra dữ liệu,
  vẫn phải đọc evaluation_scope để không trộn điểm Qwen fold 0 với CV 5 fold.

## Bằng chứng
- [Audit sau P1](audit_after_p1.json)
- [Smoke greedy](ensemble_smoke.json)
- [Hash dự đoán gốc](raw_artifacts_sha256.json)
- Bản trước sửa: `before/`, có code, notebook, bảng master, roadmap, pipeline, handoff và worklog.

## Việc tiếp theo
P2: 30 mẫu lỗi B còn 0/30 ghi chú, viết kết luận có dẫn chứng.
P3: chạy lại lựa chọn đầy đủ và cập nhật recipe/summary/07; chưa cần train toàn bộ.
Xem [roadmap hiện hành](../../../ROADMAP.md#hien-tai).

