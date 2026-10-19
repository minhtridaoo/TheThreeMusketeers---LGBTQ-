# Chuẩn bị P4 — 18/10/2026

**Trạng thái: chuẩn bị xong, chưa chạy Kaggle/T4. Không đánh dấu P4 hoàn thành.**

- Thêm checkpoint.py, p4.py và scripts/run_p4.py: train một lần/component, lưu trọng số/tokenizer/config/label map; TF-IDF lưu cả pipeline.
- train.py hỗ trợ dựng kiến trúc từ config cục bộ rồi nạp state_dict strict; dừng nếu loss không hữu hạn.
- Nạp lại bằng tiến trình mới với cache Hugging Face rỗng/offline; so nhãn và xác suất trước/sau, kiểm tra TSV.
- Resume theo model đã hoàn thành có hash và tham chiếu; chưa resume giữa epoch.
- Có smoke 64/16, dryrun 2390/599, final 2989/32 probe. Smoke mặc định, final không tự chạy.
- 43 test đạt trên CPU, gồm model transformer nhỏ được train/lưu/nạp thật, hai head A/B, TF-IDF, corrupted checkpoint, chia dữ liệu và integration subprocess/resume.
- Kaggle notebook 5 ô code, JSON/syntax hợp lệ; ZIP 23 file đã kiểm tra CRC và SHA256, khoảng 0,66 MiB.
- Dữ liệu/kết quả P3 và 30 ghi chú P2 giữ nguyên. Không có model thật được train trong lượt chuẩn bị.

[Hướng dẫn Kaggle P4](../../../../kaggle/HUONG_DAN_KAGGLE_P4.md) hướng dẫn upload Private, GPU/Internet, chạy hai ô đầu, smoke rồi dryrun. [validation.json](validation.json) ghi bằng chứng chuẩn bị. Bản file trước sửa ở before/.

P4 hoàn thành sau khi có output dryrun Kaggle thật, đủ 7 checkpoint, fresh offline reload đạt và 599 ID đúng/task, được tải về kiểm chứng. Final train toàn bộ là lượt riêng khi cần.
