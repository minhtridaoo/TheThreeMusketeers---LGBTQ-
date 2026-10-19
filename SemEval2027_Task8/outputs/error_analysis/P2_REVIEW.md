# P2 — Kiểm tra 30 mẫu lỗi B và tổng hợp kết luận

Ngày kiểm tra: **18/10/2026**. Người dùng xác nhận đã đọc xong 30 mẫu; trợ lý đối chiếu dữ liệu và tổng hợp nhận xét. Nguồn: [sample_30_B.csv](sample_30_B.csv). Ghi chú gốc được giữ nguyên.

## 1. Kết quả và trạng thái

- **30/30** mẫu có ghi chú, 30 ID khác nhau, tất cả thực sự là dự đoán sai.
- Title/comment khớp TSV gốc; video/gold/prediction khớp split, `errors_B.csv` và `mt_twhate_comment_B.csv`. Không phát hiện sửa lệch trường nguồn.
- Ba nhóm có 10 mẫu mỗi nhóm: báo nhầm hate, bỏ sót hate, nhầm implicit/explicit. Mẫu được chọn cân bằng theo nhóm lỗi, **không phản ánh tỷ lệ lỗi trong toàn bộ dữ liệu**.
- Model: **multi-task Twitter-RoBERTa, comment-only, max_length=128, không class weight/bias**. Title hiển thị để người đọc tham khảo; model này không nhận title.
- Đã có 8 ví dụ tiêu biểu, đối chiếu 7 mẫu với Qwen và kết luận context/MT/LLM bên dưới. **P2 hoàn thành phần đọc, kiểm tra và tổng hợp; tiếp theo là P3 chọn lại hệ thống.** Nội dung tổng hợp chưa được chép vào kết luận từng notebook/báo cáo môn.

Bằng chứng và hash nguồn: [sample_review.json](../audit/2026-10-18-p2/sample_review.json). Bản sao nguyên trạng CSV và tài liệu trước cập nhật: `../audit/2026-10-18-p2/before/`.

**Kiểm tra lại trước P3:** người dùng đã sửa đủ 9 ghi chú được góp ý; 30/30 trường nguồn tiếp tục khớp. Mục 2 dưới đây giữ lịch sử góp ý ban đầu, không phải danh sách lỗi còn mở. Xem [kết quả P3](../selection/P3_REVIEW.md).

## 2. Góp ý ban đầu cho ghi chú (đã được sửa)

Ghi chú đã nêu được dấu hiệu trong câu thay vì chỉ nhắc lại nhãn sai. Những chỗ đánh dấu mơ hồ/cần ngữ cảnh phù hợp mục tiêu phân tích lỗi; không dùng chúng để tự sửa gold. Số mẫu dưới đây không tính header; dùng ID nếu đã sắp xếp lại Excel.

| Mẫu / ID | Điểm cần chỉnh | Cách diễn giải nên dùng |
|---|---|---|
| 1 — `training_EN_1733` | “Bị nhiễu ... tiêu đề coming out” gán cho model thông tin không được cung cấp. | Bình luận chỉ trích hành vi cá nhân, gold=no nhưng model dự đoán explicit. Có thể nhầm đích công kích; chưa thể quy lỗi cho tiêu đề vì model chỉ nhận comment. |
| 16 — `training_EN_0620` | “Không liên kết ... với tiêu đề” dễ gây hiểu nhầm model đã nhận title. | Chỉ từ comment khó xác định đối tượng của đại từ/lời xúc phạm. Thiếu ngữ cảnh là giả thuyết cần kiểm tra; chưa chứng minh thêm title sửa được mẫu này. |
| 12 — `training_EN_1993` | “Mỉa mai” chưa mô tả chính xác nhất dấu hiệu quan sát được. | Bệnh lý hóa kết hợp lời lẽ tích cực: vừa khen sự tiếp đón vừa dùng sick/sickness để mô tả nhóm được nói tới. |
| 14 — `training_EN_0263` | Nhóm “Phủ định hoặc trích dẫn” quá rộng. | Trích dẫn có đồng tình: I agree cho thấy tác giả ủng hộ nội dung dẫn lại; cần xét lập trường người viết. |
| 20 — `training_EN_2613` | “Văn bản dài làm loãng ngữ cảnh” mới là giả thuyết. | Cần kiểm tra phần token thực sự được đưa vào model trước khi kết luận về cắt ngắn ở 128 token. Dấu hiệu hạ thấp đã có ngay đầu câu. |
| 21–23, 28 — `training_EN_1675`, `training_EN_0311`, `training_EN_2024`, `training_EN_0373` | Không biến gold của một mẫu thành quy tắc: cứ ngoặc kép/tôn giáo/công bằng thể thao/so sánh là implicit. Mức độ gay gắt không tự quyết định explicit. | Gold của mẫu này là implicit, model dự đoán explicit; câu có dấu hiệu gián tiếp và trực diện, làm nổi bật ranh giới nhãn cần xét theo toàn bộ phát ngôn. |

Áp dụng chung: viết “model dự đoán X trong câu có dấu hiệu Y” cho điều đã quan sát; viết “có thể/giả thuyết” cho nguyên nhân. Chưa có thí nghiệm đối chứng riêng cho từng từ để khẳng định từ đó gây lỗi. Ở mẫu 25, giữ sự thận trọng với từ viết lệch `prdos`, chưa xác nhận một nghĩa duy nhất.

[Guideline cục bộ](../../docs/Guidelines_for_Participants_v1.pdf), trang 2–3, yêu cầu diễn giải comment cùng ngữ cảnh và mô tả explicit là công kích trực tiếp, implicit là biểu đạt gián tiếp/hàm ý/mã hóa. Không có quy tắc rằng tôn giáo hoặc ngoặc kép luôn ứng với một nhãn. Đây không phải xác minh bản cập nhật mới từ BTC.

## 3. Tám ví dụ tiêu biểu cho báo cáo

N = `no`, I = `yes_implicit`, E = `yes_explicit`; mũi tên là gold → dự đoán MT.

| ID | Hướng sai | Dấu hiệu và điều rút ra |
|---|---|---|
| `training_EN_1250` | N → E | Comment một từ “impeach”; title nhắc Trump. Comment ngắn thiếu đối tượng, không đủ để tự suy ra công kích LGBTQIA+. |
| `training_EN_1582` | N → E | Phê phán sự mâu thuẫn của chính trị gia. Cần phân biệt lời lẽ gay gắt với hate nhắm vào nhóm mục tiêu của bài toán. |
| `training_EN_1993` | E → N | Bệnh lý hóa xuất hiện cùng lời khen lòng nhân đạo. Cần xét nội dung đánh giá nhóm người, không chỉ sắc thái tích cực bề mặt. |
| `training_EN_0263` | I → N | Đồng tình trước khi trích dẫn quan điểm. Lập trường đối với lời dẫn lại là một phần của nội dung cần diễn giải. |
| `training_EN_0335` | I → N | Đại từ và câu nghi vấn về bản dạng cần liên hệ nhân vật trong video. Minh họa giả thuyết thiếu ngữ cảnh, chưa chứng minh context model tốt hơn. |
| `training_EN_1675` | I → E | Chê ngoại hình trực tiếp kết hợp ngoặc kép ở từ chỉ giới. Khó giải thích ranh giới nhãn bằng một từ khóa. |
| `training_EN_0370` | E → I | Từ viết lệch khó xác định nghĩa. Ghi nhận bất định, không tự mở rộng thành một từ chắc chắn. |
| `training_EN_0939` | I → E | Tán thành loại trừ một người kết hợp khẩu hiệu chính trị; sự kiện trong ngữ cảnh giúp diễn giải thái độ. |

## 4. Đối chiếu Qwen trên đúng ID thuộc fold 0

Chỉ **7/30 mẫu** thuộc fold 0 có dự đoán Qwen để đối chiếu. Các ô là nhãn đã lưu, không có lượt chạy mới.

| ID | Gold | MT | Qwen zero-shot | Qwen few-shot |
|---|---|---|---|---|
| `training_EN_1733` | N | E | N | N |
| `training_EN_1090` | N | I | N | N |
| `training_EN_1993` | E | N | N | N |
| `training_EN_1675` | I | E | E | E |
| `training_EN_0311` | I | E | N | I |
| `training_EN_0373` | I | E | N | N |
| `training_EN_0939` | I | E | N | I |

Zero-shot đúng 2/7, few-shot đúng 4/7. Đây là **mẫu chọn từ lỗi MT**, nên MT sai 7/7 theo thiết kế. Không dùng tỷ lệ này để xếp hạng model hoặc suy ra Qwen tốt hơn trên toàn bộ dữ liệu. Hai mẫu `0311` và `0939` minh họa few-shot sửa lỗi implicit của zero-shot; `1993` và `1675` vẫn sai ở cả ba cấu hình.

Qwen nhận title+description+comment còn MT nhận comment; so sánh này không cô lập tác động kiến trúc hoặc ngữ cảnh.

## 5. Kết luận thực nghiệm để đưa vào báo cáo

**Context:** Một số lỗi cho thấy người đọc cần biết đối tượng/sự kiện của video. Tuy nhiên, B với RoBERTa comment-only đạt 0,5856 macro-F1 trung bình 5 fold; thêm title đạt 0,5808, thêm title+description đạt 0,5692, ghép thẳng đạt 0,5506. Các cách đưa context đã thử chưa cải thiện B. Không suy ra context luôn vô ích hoặc chỉ cần thêm title là sửa được các lỗi này.

**Multi-task:** Cùng encoder Twitter-RoBERTa và input comment, MT không class weight/bias đạt A 0,7354 so với single 0,7335; B 0,6006 so với 0,5870. B tăng khoảng 0,0136 macro-F1 trung bình 5 fold, nhưng F1 implicit của MT vẫn chỉ 0,4313, thấp hơn explicit 0,6204. Cùng các ví dụ, kết quả cho thấy ranh giới nhãn và biểu đạt gián tiếp vẫn khó. Không dùng 30 mẫu chọn theo nhóm để ước lượng mức phổ biến của từng nguyên nhân; không tuyên bố ý nghĩa thống kê từ chênh lệch này.

**LLM:** Trên toàn bộ 599 mẫu fold 0, Qwen B tăng 0,4612 → 0,5187 khi dùng few-shot; F1 implicit tăng 0,1611 → 0,3402, explicit giảm 0,5362 → 0,5132. Các ví dụ minh họa cả cải thiện lẫn lỗi còn lại. MT cùng fold đạt 0,6111, nên giữ Qwen là đối chiếu; chưa có căn cứ thay hệ thống supervised. Không so điểm Qwen fold 0 trực tiếp với trung bình 5 fold để chọn model.

Nguồn: [validated_results.csv](../results/validated_results.csv), [llm_comparison_fold0.csv](../results/llm_comparison_fold0.csv), OOF và dự đoán Qwen có hash trong audit. Đây là kết quả phát triển, không phải điểm test độc lập.

## 6. Bước tiếp theo

**P3:** so lại single/top-3/greedy/bias từ OOF hợp lệ, xem từng fold/từng lớp, rồi cập nhật recipe/summary và phân tích tương ứng. Không cần chạy lại toàn bộ 01–08 hoặc train mới chỉ để hoàn thành P3.

**Đã xử lý ở P3:** notebook 07 xuất lỗi vào thư mục riêng theo hash nguồn, giữ nguyên CSV P2. Chạy lại cùng nguồn giữ nguyên ghi chú; không tự gắn giải thích lỗi MT cho chiến lược mới. Bản sao P2 vẫn được giữ để truy vết.
