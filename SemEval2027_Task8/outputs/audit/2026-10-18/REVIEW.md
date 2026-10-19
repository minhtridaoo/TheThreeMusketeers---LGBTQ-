# Kiểm tra kết quả và tiến độ — 18/10/2026
Nguồn: artifact hiện có trong dự án, được kiểm tra trực tiếp; không chỉ dựa vào execution count.
[Audit trước merge](audit_before_merge.json) và [sau merge](audit_after_merge.json) lưu kiểm tra chi tiết,
hash notebook/prediction/OOF và điểm tính lại. Script: [audit_artifacts.py](../../../scripts/audit_artifacts.py).

## 1. Kết luận về notebook 06
**Đạt kiểm tra: đã hoàn thành full.** 5/5 code cell có execution count, không có error output.
4 cấu hình ×599 mẫu, 4 journal mỗi journal 599 bản ghi; không có failed attempt.
Status `full/complete`; model Qwen2.5-7B-Instruct Q4_K_M, Ollama 0.40.3, context 4096.

Đã kiểm tra ID duy nhất/đủ 599, gold/fold/video khớp dữ liệu EN; fingerprint metadata và cache,
nội dung prompt hiện tại, nhãn JSON gốc và nhãn CSV; few-shot không trùng ID/video với tập chấm.
Toàn bộ điểm macro-F1 và F1 lớp tính lại khớp CSV. 14 dòng bảng so sánh fold 0 cũng khớp.

Thời gian journal: **14/10/2026 05:52:09–06:07:03 UTC**, khoảng 14 phút 55 giây giữa bản ghi đầu/cuối.
Tổng thời gian request khoảng 14 phút 38 giây. Không dùng con số này làm thời gian setup/tải model
hay cam kết tốc độ cho GPU khác. Ngày 18/10 là ngày kiểm tra và cập nhật tài liệu.

| Cấu hình | A macro-F1 | B macro-F1 | F1 implicit B | F1 explicit B |
|---|---:|---:|---:|---:|
| Zero-shot | 0,607411 | 0,461242 | 0,161137 | 0,536232 |
| Few-shot | 0,692715 | 0,518695 | 0,340176 | 0,513208 |
| Chênh lệch few − zero | +0,085305 | +0,057453 | +0,179039 | −0,023024 |

Few-shot giúp nhận lớp ít mẫu hơn nhưng có đánh đổi:
- A: recall stereotype `yes` tăng 36,84% → 62,03%; recall `no` giảm 87,09% → 76,28%.
- B: implicit bị đoán thành `no` giảm **117/155 → 78/155**; implicit đúng tăng **17/155 → 58/155**.
- B: explicit bị đoán thành implicit tăng **20/162 → 70/162**.
- Tổng mẫu B đoán đúng giảm 335 → 334, nhưng macro-F1 tăng vì kết quả được phân bố cân bằng hơn
  giữa các lớp. Đây là ví dụ vì sao không chọn model chỉ theo accuracy hay ô `no/no`.

## 2. So sánh công bằng về tập chấm
Tất cả số sau chấm cùng **599 mẫu fold 0**:
| Hệ thống | A | B |
|---|---:|---:|
| TF-IDF comment | 0,705628 | 0,524467 |
| RoBERTa comment | 0,731372 | 0,617079 |
| Twitter-RoBERTa-hate comment | 0,739700 | 0,609509 |
| MT Twitter-RoBERTa comment | 0,743059 | 0,611117 |
| Qwen zero-shot | 0,607411 | 0,461242 |
| Qwen few-shot | 0,692715 | 0,518695 |

Qwen few-shot chưa vượt TF-IDF (A −0,012913; B −0,005772) hay transformer trên fold này.
Không suy rộng thành mọi LLM đều thua; đây là một model lượng tử hóa, một bộ prompt,
một cách chọn few-shot và một fold. Qwen dùng title/description/comment, baseline dùng comment.
Giữ Qwen như đối chứng, tiếp tục hướng supervised cho hệ thống chính.

## 3. Tình trạng toàn bộ notebook
Cả 8 notebook đều có đủ execution count và không lưu error output.
Tuy nhiên trạng thái tính toán hợp lệ khác nhau:
- **01–03:** có dữ liệu/split và baseline dùng được.
- **04:** các cấu hình RoBERTa/Twitter-RoBERTa có OOF hợp lệ; DeBERTa A/B failed.
- **05:** OOF các biến thể còn đầy đủ; bảng chung chỉ còn 2 dòng do key ghi kết quả trùng.
- **06:** hoàn tất/kiểm chứng như trên.
- **07:** đã xuất lỗi/biểu đồ; 30 mẫu lỗi B chưa điền kiểu lỗi.
- **08:** có recipe và dry-run, nhưng nhánh greedy lỗi nên cần chọn lại trên đầu vào sạch.

## 4. Các vấn đề đã tái xác nhận
### DeBERTa không phải một baseline thành công
32 file OOF được quét, **30 hợp lệ, 2 không hợp lệ**:
- `deberta_comment_A.csv`: 5.978 ô xác suất không hữu hạn.
- `deberta_comment_B.csv`: 8.967 ô xác suất không hữu hạn.
- Mỗi lịch sử train có 20 bản ghi epoch với loss NaN (5 fold ×4 epoch).

Nhãn argmax rơi vào `no` làm điểm trông giống majority. Không dùng điểm đó để đánh giá
chất lượng DeBERTa. Nguyên nhân gốc của mất ổn định số chưa được điều tra trong lượt kiểm tra này.

### Greedy ensemble có thể hỏng dù trọng số DeBERTa bằng 0
Đã tái hiện với code hiện tại: `average([OOF_tốt, OOF_NaN], weights=[1,0])` cho toàn NaN
ở cả A và B, vì 0×NaN vẫn là NaN. Kết quả greedy cũ không dùng để kết luận phương pháp kém.
Recipe A top-3/B single không chứa DeBERTa, nhưng lựa chọn chiến lược vẫn phải xét lại
khi một đối thủ so sánh đã bị tính sai.

### Bảng notebook 05 mất phân biệt biến thể
`src/sqe/results.py` dùng khóa notebook/model/input/subtask; các lượt MT, MT+cw và bias
dùng cùng khóa nên ghi đè nhau. 2 dòng còn lại đang là MT+cw+bias, không đại diện toàn bộ kết quả 05.
OOF còn đầy đủ để phục hồi bảng **12 cấu hình-task** (single/MT/MT+cw × raw/bias × A/B) mà không train lại.

### Sẵn sàng suy luận và báo cáo
- `sample_30_B.csv`: 0/30 kiểu lỗi được điền.
- Mục kết luận notebook 04–08 còn nội dung chờ điền.
- Không thấy checkpoint model dưới đuôi `.pt/.pth/.safetensors/.bin` trong dự án được kiểm tra.
- Dry-run A/B đủ 599 ID mỗi file, nhãn hợp lệ; điểm A 0,741446, B 0,611117 chỉ kiểm tra quy trình.
- Các ước lượng từ model selection/postprocessing OOF là kết quả phát triển, không chứng minh
  toàn pipeline đã được đánh giá bằng nested CV độc lập.

## 5. Đồng bộ đã thực hiện
Trước merge: master 24 dòng, chưa có Qwen. Sau merge: **28 dòng**, giữ nguyên nội dung 24 dòng cũ
và thêm đúng 4 dòng Qwen có experiment/scope/fingerprint. Đã tạo [bản sao trước merge](all_results.before_llm_merge.csv).
Các NaN ở cột F1 không áp dụng cho task khác hoặc std Qwen là ô trống có chủ đích;
chúng khác lỗi NaN trong ma trận xác suất DeBERTa.

Chưa sửa code train/ensemble/logger hoặc chạy lại notebook; không ghi đè dự đoán gốc.
`summary_table.csv` và recipe cũ vẫn cần tái tạo trong P1/P3.
Kế hoạch hành động nằm ở [PIPELINE.md](../../../PIPELINE.md).

