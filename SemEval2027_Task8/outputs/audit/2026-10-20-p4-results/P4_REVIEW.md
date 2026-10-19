# Kiểm chứng P4 và tổng quan tiến độ — 20/10/2026

**Kết luận: P4 hoàn thành. Cả smoke, dryrun và final đã chạy trên Tesla T4; ba ZIP tải về đạt kiểm tra. P0–P4 đã xong, bước tiếp theo là P5: báo cáo và bàn giao.**
Phạm vi giữ nguyên: Subtask A + B, English. Ngày 20/10 là ngày audit tại máy; không suy thời điểm từng lượt train từ ngày sửa ZIP.

## 1. Bạn đã làm được gì?

1. **Hiểu và chuẩn bị bài toán:** nhận comment YouTube, dự đoán A có stereotype hay không và B thuộc no / hate speech implicit / explicit. Có 2.989 mẫu EN, chia 5 fold theo 243 video để hạn chế trùng ngữ cảnh giữa train và validation.
2. **Tạo mốc so sánh:** notebook 01–03 kiểm tra dữ liệu, chạy baseline đơn giản/TF-IDF và fine-tune transformer. Có dự đoán ngoài fold (OOF) để so sánh trên cùng split.
3. **Thử các hướng cải tiến:** notebook 04–06 thử title/description, encoder, học chung A/B, class weight và Qwen zero/few-shot. Qwen chỉ đánh giá 599 mẫu fold 0; DeBERTa có NaN nên bị loại, không tính là một kết quả hợp lệ.
4. **Kiểm tra và giải thích kết quả:** P0/P1 phục hồi bảng kết quả, chặn xác suất hỏng; P2 đọc 30 lỗi B, tổng hợp ví dụ và kết luận. Đã có phân tích những loại lỗi còn khó.
5. **Chọn hệ thống cuối:** P3 so single / top-3 / greedy, có/không bias. Chọn A greedy+bias và B top-3+bias, lưu thành phần, trọng số, bias, config và nguồn/hash trong recipe.
6. **Biến cấu hình thành model dùng lại được:** P4 train 7 thành phần, lưu model/tokenizer/config, nạp offline trong tiến trình mới, kiểm tra dự đoán trước/sau lưu và xuất TSV. Đã có cả bộ checkpoint train toàn bộ 2.989 mẫu.

P4 xác nhận hệ thống đã chọn có thể lưu và nạp lại để suy luận. Nó không chứng minh đây là hệ thống tốt nhất cuộc thi; chưa có đánh giá test độc lập hoặc lần nộp chính thức trong các artifact này.

## 2. Ba lượt Kaggle đã kiểm chứng

| Lượt | Mẫu train | Mẫu kiểm tra mỗi task | Epoch mỗi transformer | Checkpoint | Kết quả |
|---|---:|---:|---:|---:|---|
| smoke | 64 | 16 | 1 | 7 | Đạt thử train/save/reload |
| dryrun | 2.390 từ fold 1–4 | 599 từ fold 0 | 4 | 7 | Đạt toàn bộ pipeline, A/B đúng ID và nhãn |
| final | 2.989 toàn bộ EN | 32 lấy từ chính train | 4 | 7 | Đạt save/reload bộ model toàn dữ liệu; không chấm chất lượng trên 32 mẫu này |

Mỗi lượt có 5 transformer + 2 TF-IDF. Hai transformer multi-task có hai đầu A/B, nên 7 checkpoint tương ứng **9 phép so sánh dự đoán trước/sau nạp**. Cả 27 phép so sánh trong ba báo cáo Kaggle đều có nhãn giống hệt, `max_abs_diff=0.0`, `fresh_process=true`, `offline_reload=true`; PID tiến trình nạp khác PID tiến trình gọi.

Môi trường ghi trong cả ba gói: Python 3.13.15, PyTorch 2.11.0+cu128, Transformers 5.16.1, scikit-learn 1.6.1, CUDA 12.8, Tesla T4.

Tổng `training_minutes` của các component: smoke **0,71 phút**, dryrun **29,59 phút**, final **36,59 phút**. Đây là thời gian các bước train/dự đoán tham chiếu/lưu component được đo trong code, **không phải thời gian toàn phiên**, chưa gồm toàn bộ setup, reload, đóng ZIP hoặc tải về.

## 3. Kiểm tra thực hiện tại máy

- Đọc đủ **180 ZIP member** (60/gói), kiểm tra CRC trong quá trình giải nén luồng; lưu SHA-256 từng member và toàn bộ ba ZIP.
- Kiểm tra **66 hash payload checkpoint** (22/gói), 21 checkpoint metadata, 21 file xác suất tham chiếu; không có file báo lỗi hoặc checkpoint partial.
- Đối chiếu code đóng gói với code hiện tại, recipe P3, config gốc, run fingerprint, checkpoint identity, môi trường và thứ tự nhãn.
- Tái tạo đúng ID train/probe từ dữ liệu gốc và split: hash dữ liệu khớp; smoke/dryrun không trùng video train–probe; final dùng toàn train đúng thiết kế.
- Kiểm tra input suy luận không chứa nhãn, nội dung comment/title/description và gold khớp nguồn.
- Lịch sử transformer đủ 1 hoặc 4 epoch; mọi train loss hữu hạn. Xác suất tham chiếu đúng kích thước, hữu hạn và chuẩn hóa.
- Tự kết hợp lại xác suất theo trọng số/bias của recipe: cả **6 TSV** khớp hoàn toàn về ID/thứ tự/nhãn; tính lại macro-F1 dryrun khớp báo cáo.
- **62 artifact hiện có không thay đổi**, gồm OOF, bảng kết quả, lựa chọn P3, recipe, split và ghi chú P2.

**Phạm vi kiểm chứng:** đối chiếu độc lập file/hash/dữ liệu/dự đoán và bằng chứng reload đã lưu từ Kaggle. Không chạy lại forward pass của các model lớn trên CPU tại máy; môi trường local khác Kaggle. 43 unit test là kết quả đã đạt ngày 18/10 khi chuẩn bị P4, không phải lượt test mới ngày 20/10. Phiên này không sửa code train/inference hay train thêm model.

## 4. Cách đọc điểm cho đúng

| Bằng chứng | A macro-F1 | B macro-F1 | Dùng để làm gì? |
|---|---:|---:|---|
| P3: mean ± std trên 5 fold | 0,746020 ± 0,010278 | 0,607455 ± 0,012700 | Kết quả lựa chọn hệ thống trên dữ liệu phát triển; cross-fit tầng quyết định |
| P4 dryrun: 599 mẫu fold 0 | 0,754877 | 0,614572 | Kiểm tra pipeline với recipe đã chốt |
| P4 final: 32 mẫu probe từ train | Không chấm | Không chấm | Chỉ xác nhận lưu/nạp bộ checkpoint toàn dữ liệu |

**Không so trực tiếp hai dòng P3/P4 để kết luận model tăng điểm:** khác phạm vi và cách chọn recipe; fold 0 đã tham gia phát triển P3. Chưa phải nested CV độc lập toàn pipeline hoặc test của BTC.
F1 từng lớp B trong dryrun: no **0,742115**, implicit **0,480519**, explicit **0,621083**. Implicit vẫn là lớp khó trong lượt kiểm tra này; đây chưa phải kết quả test mới.
Giữ nguyên master/summary P3, không đưa điểm dryrun vào bảng như một kết quả CV mới.

## 5. File nằm ở đâu và cần giữ gì?

Các model đầy đủ nằm trong ba ZIP tại [kaggle/results](../../../../kaggle/results):

- `sqe_p4_final.zip`: **bộ checkpoint toàn dữ liệu cần giữ để dùng suy luận về sau**.
- `sqe_p4_dryrun.zip`: bằng chứng kiểm tra pipeline trên fold 0 cần giữ.
- `sqe_p4_smoke.zip`: bằng chứng lượt thử nhỏ.

Trong từng ZIP: `SemEval2027_Task8/outputs/p4/<mode>/checkpoints/`, cùng `plan.json`, `reload_report.json`, `reference/` và `predictions/`.
Chưa giải nén các trọng số lớn vào `outputs/p4/` của dự án. Chỉ chép bằng chứng nhẹ vào [evidence](evidence) để xem nhanh; **đây không phải bản checkpoint đầy đủ**. Giữ ZIP nguồn, không thay thế toàn bộ dự án bằng bản code trong ZIP.

- [validation.json](validation.json): kết luận audit và số liệu.
- [smoke_audit.json](smoke_audit.json), [dryrun_audit.json](dryrun_audit.json), [final_audit.json](final_audit.json): hash và kết quả chi tiết.
- [verify_downloads.py](verify_downloads.py): script tái kiểm chứng, không train hoặc thực thi trọng số model.
- [evidence/dryrun/reload_report.json](evidence/dryrun/reload_report.json): báo cáo Kaggle gốc.
- [evidence/final/reload_report.json](evidence/final/reload_report.json): bằng chứng nạp bộ final.

## 6. Việc tiếp theo — P5

1. **Chốt nội dung báo cáo môn:** bài toán/nhãn, dữ liệu và cách chia theo video, phương pháp, kết quả, phân tích lỗi, giới hạn và kết luận. Dùng template/rubric của giảng viên khi có.
2. **Lấy kết quả có sẵn vào báo cáo:** supervised mean ± std 5 fold; Qwen trong bảng riêng cùng fold 0; lựa chọn P3 có lý do; 8 ví dụ P2; P4 mô tả khả năng lưu/nạp và tính tái lập.
3. **Chuẩn bị bàn giao và trình bày:** hướng dẫn chạy, vị trí checkpoint final, môi trường thư viện, sơ đồ pipeline và giải thích được vì sao chọn hệ thống này.
4. **Khi có dữ liệu/yêu cầu test chính thức:** xác minh schema/format BTC; bổ sung lệnh suy luận nhận file mới, nạp checkpoint final, chạy A/B, kiểm tra TSV rồi mới nộp. Runner P4 hiện train/verify/package trên probe cố định, chưa có lệnh suy luận test tùy ý hoặc bước tự nộp.

Ưu tiên trước mắt là viết báo cáo từ bằng chứng đã có. Chưa cần thêm một lượt Kaggle hoặc chạy lại 01–08. P5 chưa hoàn thành; chưa xác minh lịch BTC trong phiên này và hạn môn chưa được cung cấp. So với lịch thí nghiệm gốc trong roadmap, phần thực nghiệm đi trước; chưa thể suy ra dư/thiếu thời gian cho toàn bộ đồ án.
