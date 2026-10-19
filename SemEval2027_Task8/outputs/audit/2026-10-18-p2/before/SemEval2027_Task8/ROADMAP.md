# Roadmap và pipeline — CS221 / StereoQueerEval
**Một tài liệu kế hoạch duy nhất — cập nhật 18/10/2026 sau khi hoàn thành P1.**
Phạm vi: **Subtask A + B, English**.
[Handoff](../HANDOFF_CS221.md) là bản bàn giao trạng thái; [worklog](../WORKLOG_CS221.md) là lịch sử công việc.
Nội dung PIPELINE.md đã được gộp vào file này.

<a id="hien-tai"></a>
## 1. Bạn đang ở đâu, nhanh hay chậm?
**Vị trí hiện tại: P2 — đọc lỗi và viết kết luận. P0 và P1 đã hoàn thành.**
Đã có kết quả thực nghiệm; đang chuyển sang giải thích kết quả, chọn lại hệ thống và hoàn thiện sản phẩm.

| Phần công việc | Trạng thái hiện tại |
|---|---|
| Hiểu dữ liệu và thiết lập đánh giá | Xong: EDA, 2.989 mẫu EN, 243 nhóm video, 5 fold |
| Baseline và model chính | Đã chạy TF-IDF, RoBERTa; có OOF hợp lệ |
| Context, multi-task, Qwen | Đã chạy; Qwen 4×599 dự đoán đã kiểm chứng; DeBERTa được ghi nhận là failed |
| Kiểm tra và sửa dữ liệu kết quả — P0/P1 | **Xong:** phục hồi bảng 05, chặn OOF hỏng, sửa NaN ensemble, 35 test đạt |
| Đọc lỗi và viết kết luận — P2 | **Đang đến bước này:** 0/30 mẫu lỗi B có ghi chú |
| Chọn lại hệ thống cuối — P3 | Chưa xong; đã có smoke kiểm tra greedy hoạt động, chưa chọn recipe mới |
| Lưu/nạp model và đóng gói suy luận — P4 | Chưa xong; chưa thấy checkpoint lưu trong dự án |
| Báo cáo môn và nộp cuộc thi — P5 | Chưa xong; lịch/hạn môn cần xác nhận |

**Đánh giá tốc độ:** so với lịch gốc, phần chạy thí nghiệm đang đi trước.
Lịch gốc dành 12–25/10 cho bước 2–3, 26/10–08/11 cho context và 09–22/11 cho multi-task/LLM;
artifact hiện có cho thấy các lượt này đã chạy trong 11–14/10 và được rà soát/sửa kết quả ngày 18/10.
Đây là so sánh mốc công việc, không có nghĩa đã hoàn thành báo cáo hoặc hệ thống nộp bài.

**Chưa thể kết luận dư hay thiếu thời gian theo hạn môn**, vì ngày nộp chưa được xác nhận.
Không dùng “8/8 notebook” làm phần trăm hoàn thành đồ án.
Điểm cần tập trung lúc này là 30 mẫu lỗi, kết luận có dẫn chứng, lựa chọn hệ thống và checkpoint;
chưa cần thêm model hay chạy lại toàn bộ 01–08.

**Cách đọc số thứ tự:** notebook 01–08 là các file thí nghiệm. P0–P5 là các mốc hoàn thiện
sau khi đã có kết quả; **P2 không có nghĩa quay lại notebook 02**.

## 2. Bài toán và sản phẩm cần bàn giao
Input là comment YouTube, có thể kèm tiêu đề và mô tả video.
- **A:** có stereotype về LGBTQIA+ không → `no / yes`.
- **B:** có hate speech không, thể hiện thế nào → `no / yes_implicit / yes_explicit`.
- Metric chính: macro-F1, kèm F1/recall từng lớp; không chọn model chỉ vì ô `no/no` lớn.
- Train EN hiện có: 2.989 mẫu / 243 nhóm video; dùng nguyên `data/splits/folds_en.csv`.
- Scope hiện tại không mở rộng sang C/IT/NL/Persian. 100 mẫu Persian development là nhánh riêng,
  không nhập vào EN folds hay tính chung với metric EN.

Sản phẩm: thí nghiệm có thể kiểm chứng, bảng kết quả đúng phạm vi đánh giá, phân tích lỗi,
recipe có lý do lựa chọn, model/tokenizer/config có thể nạp lại, TSV dự đoán hợp lệ và báo cáo.
Hạn nộp môn **chưa xác nhận**. Lịch BTC ghi ở phần cuối là thông tin cũ, cần kiểm tra trước lúc nộp.

<a id="pipeline"></a>
## 3. Pipeline từ dữ liệu đến hệ thống
```mermaid
flowchart TD
    D["Dữ liệu EN và guideline"] --> E["01: audit, EDA, chia fold theo video"]
    E --> B["02: random / majority / TF-IDF"]
    E --> T["03: RoBERTa comment"]
    T --> C["04: context và encoder"]
    C --> M["05: single / multi-task / class weight / bias"]
    E --> L["06: Qwen zero/few-shot, fold 0"]
    B --> V["P0/P1: kiểm tra OOF và sửa bảng kết quả - ĐÃ XONG"]
    T --> V
    C --> V
    M --> V
    V --> X["P2 / 07: đọc lỗi, viết kết luận - HIỆN TẠI"]
    V --> S["P3 / 08: chọn lại hệ thống từ OOF sạch"]
    L --> R["Bảng đối chiếu riêng trên cùng 599 mẫu"]
    X --> W["P5: báo cáo có dẫn chứng và giới hạn"]
    R --> W
    S --> W
    S --> F["P4: train cấu hình cuối, lưu và nạp checkpoint"]
    F --> Q["Dry-run: dự đoán, xuất TSV, kiểm tra ID/nhãn"]
    Q --> U["Test chính thức và nộp A/B khi có"]
```

Qwen hiện chỉ có nhãn trên fold 0; không đưa vào ensemble xác suất của notebook 08.
Few-shot lấy ví dụ từ fold 1–4, không trùng ID/video của tập đang chấm.

## 4. Bản đồ 8 notebook
| Notebook | Câu hỏi cần trả lời | Đầu ra / trạng thái |
|---|---|---|
| 01 — EDA | Dữ liệu có gì, nhãn nào khó, cần chia thế nào? | Dữ liệu/folds đã kiểm tra; đọc guideline và ví dụ trước khi giải thích lỗi |
| 02 — classical | Cách đơn giản làm được tới đâu? | Random/majority/TF-IDF, OOF và mốc so sánh |
| 03 — transformer | Fine-tune RoBERTa có hơn TF-IDF không? | Có OOF hợp lệ A/B, chưa thấy checkpoint lưu |
| 04 — context | Title/description giúp hay gây nhiễu? | RoBERTa/Twitter-RoBERTa hợp lệ; DeBERTa failed, không dùng điểm giả majority |
| 05 — multi-task | Học A/B chung, class weight và bias có ích không? | **Bảng 12 dòng đã phục hồi** từ OOF; code logger đã sửa |
| 06 — LLM | Không fine-tune, Qwen zero/few-shot làm được tới đâu? | Full 4×599, 0 failed attempts; đã kiểm chứng và merge |
| 07 — error analysis | Model sai ở đâu và vì sao? | Có bảng/hình lỗi; **0/30** mẫu B được điền kiểu khó |
| 08 — ensemble/submission | Chọn hệ thống nào và xuất dự đoán thế nào? | Code đã chặn OOF hỏng; cần chạy lại lựa chọn/summary ở P3; dry-run cũ có 599 ID hợp lệ/task |

Cả 8 notebook từng chạy đầy đủ trước P1. Sau sửa code 05/07/08, output phụ thuộc thay đổi đã được
xóa để không trộn kết quả cũ với code mới; chưa chạy lại các notebook này. Bản đã chạy cũ được giữ
ở `outputs/audit/2026-10-18-p1/before/SemEval2027_Task8/notebooks/`.
Notebook 06 và dự đoán gốc vẫn giữ nguyên.

## 5. Kết quả và hướng nghiên cứu hiện tại
### Supervised — mean ± std 5 fold theo video
| Cấu hình | A | B |
|---|---:|---:|
| TF-IDF comment | 0,7002 ± 0,0209 | 0,5249 ± 0,0138 |
| RoBERTa comment | 0,7274 ± 0,0098 | 0,5856 ± 0,0179 |
| Twitter-RoBERTa-hate comment | 0,7335 ± 0,0115 | 0,5870 ± 0,0114 |
| Multi-task cùng encoder, không cw/bias | 0,7354 ± 0,0111 | 0,6006 ± 0,0259 |

Các số này là điểm phát triển/CV, không phải test độc lập của BTC.
Bảng các biến thể 05: [nb05_restored.csv](outputs/results/nb05_restored.csv).

Context chưa cải thiện ổn định: A thêm title+description +0,0014 nhưng thắng 3/5 fold;
B thêm title −0,0048, title+description −0,0164, ghép thẳng −0,0350 so với comment-only.
Không kết luận ngữ cảnh luôn vô ích; chỉ mô tả kết quả của các cách đưa ngữ cảnh đã thử.

### LLM — cùng fold 0, 599 mẫu, không có std 5 fold
| Hệ thống | A | B |
|---|---:|---:|
| Qwen zero-shot | 0,6074 | 0,4612 |
| Qwen few-shot | 0,6927 | 0,5187 |
| TF-IDF comment | 0,7056 | 0,5245 |
| RoBERTa comment | 0,7314 | 0,6171 |

Qwen Q4_K_M dùng title+description+comment; baseline trong bảng dùng comment.
Few-shot cải thiện F1 implicit B 0,1611 → 0,3402 nhưng F1 explicit giảm 0,5362 → 0,5132.
Giữ Qwen như thí nghiệm đối chiếu; **supervised comment-only / multi-task là hướng chính hiện tại**.

### Sau sửa P1
- Master: **38 dòng** = 34 `completed_valid`, 2 `failed_nonfinite`, 2 `pending_revalidation`.
- `load_results()` mặc định chỉ trả kết quả completed_valid; `include_invalid=True` dùng xem lịch sử.
- 30/32 file OOF hợp lệ; 2 file DeBERTa được giữ làm bằng chứng failed và bị chặn khỏi lựa chọn.
- Smoke greedy trên 9 ứng viên hợp lệ/task cho A **0,745465**, B **0,595965**, không còn NaN.
  Đây là kiểm tra hồi quy P1, **chưa chọn recipe mới**, chưa so lại đủ mọi chiến lược/bias ở P3.
- Công thức cũ A top-3/B single vẫn được giữ lịch sử, đánh dấu cần tái xác nhận.

<a id="ke-hoach"></a>
## 6. Kế hoạch còn lại và tiêu chí hoàn thành
| Mốc | Việc phải làm | Xong khi | Ước lượng / trạng thái |
|---|---|---|---|
| **P0** | Audit 06/OOF và merge Qwen | 4×599 khớp nhãn/cache/điểm, có bản sao | **Xong 18/10** |
| **P1** | Chặn OOF lỗi, sửa NaN ensemble, định danh thí nghiệm, phục hồi bảng 05 | 12 dòng 05, 4 dòng Qwen giữ nguyên, 35 test đạt, smoke greedy hợp lệ | **Xong 18/10, không train mới** |
| **P2 — hiện tại** | Đọc 30 lỗi B, đối chiếu một số lỗi Qwen, viết kết luận context/MT/LLM | 30/30 có kiểu lỗi, 5–10 ví dụ tiêu biểu có ID và giải thích; không sửa gold theo dự đoán | 1–2 buổi |
| **P3** | Chạy lại phần chọn chiến lược 08 trên OOF sạch; so single/top-3/greedy/bias và từng fold; đồng bộ 07/summary | Recipe có nguồn/hash, metric hợp lệ, lý do chọn; bảng cuối đúng scope | 1 buổi chủ yếu CPU |
| **P4** | Bổ sung save/reload; dry-run ở phiên mới; train cấu hình chốt trên toàn bộ train khi cần | Model/tokenizer/config/label map; nạp lại dự đoán nhất quán; TSV đúng đủ ID/nhãn | GPU ở bước train cần thiết, đo runtime thực tế |
| **P5** | Viết báo cáo; xác minh lịch/format BTC khi đến bước nộp | Báo cáo môn, kết quả nộp thực tế nếu có test; ghi worklog | 2–3 buổi viết, lịch BTC xác nhận riêng |

Ước lượng dùng để sắp xếp công việc, không phải cam kết ngày hoàn thành.
Có thể viết phần dữ liệu/phương pháp ngay trong P2; P3 phải xong trước khi train cấu hình cuối ở P4.
Nếu hạn gấp: ưu tiên kết quả sạch, phân tích lỗi, bảng so sánh và báo cáo CV;
không bắt buộc chạy lại DeBERTa hay mở thêm LLM/ngôn ngữ.

**Việc kế tiếp cụ thể:** mở `outputs/error_analysis/sample_30_B.csv`, đọc 10 mẫu đầu,
ghi `kieu_kho` (cần context, mỉa mai, phủ định/trích dẫn, nhầm implicit/explicit, nghi nhãn mơ hồ…).
Đối chiếu guideline trước khi giải thích. User đọc/ghi nhận; trợ lý hỗ trợ kiểm tra và hệ thống hóa,
không coi nhãn lỗi tự gán của trợ lý là kết quả đọc tay đã được user xác nhận.

## 7. Điều kiện hợp lệ và cách lưu kết quả
1. Dùng nguyên split theo video; fit TF-IDF/class weight/mô hình chỉ từ train fold.
2. OOF phải đủ đúng ID, gold/fold/video khớp; không thiếu/trùng; xác suất hữu hạn trong [0,1],
   tổng hàng ≈1, nhãn dự đoán khớp argmax. Các model trong ensemble phải cùng thứ tự ID/gold/fold/video.
3. Loại trọng số 0 trước phép nhân; thành phần có trọng số dương không hợp lệ phải dừng.
4. Khóa kết quả hiện tại: `notebook + model + input + subtask + experiment + evaluation_scope`.
   Biến thể raw/bias và single/MT/cw có experiment riêng, không đè nhau.
5. Supervised báo cáo mean ± std 5 fold (ddof=0), kèm F1 lớp. Qwen fold 0 có n=599, std để trống.
6. Chọn model/bias/ensemble tách dữ liệu chọn và chấm. Hậu xử lý trên OOF hiện có là cross-fitting tầng quyết định,
   chưa phải nested CV độc lập toàn pipeline. Muốn đánh giá toàn pipeline nghiêm ngặt cần outer split và
   tạo lại inner OOF chỉ từ outer-train.
7. Điểm dry-run fold 0 đã tham gia phát triển chỉ kiểm tra quy trình, không gọi là điểm test.
8. Không kết luận ý nghĩa thống kê chỉ vì thắng 4/5 fold; số fold thắng là mô tả.
9. Seed 42 và lưu phiên bản/precision thực tế. Không mở/sửa TSV gốc bằng Excel.
10. Sau mỗi mốc kiểm chứng: thêm worklog, đồng bộ handoff và file roadmap này.

## 8. File và thao tác cần biết
- [Audit notebook 06 và kết quả cũ](outputs/audit/2026-10-18/REVIEW.md)
- [Báo cáo sửa P1](outputs/audit/2026-10-18-p1/P1_REVIEW.md)
- [Master có trạng thái](outputs/results/all_results.csv) / [chỉ kết quả hợp lệ](outputs/results/validated_results.csv)
- [Bảng 12 biến thể notebook 05](outputs/results/nb05_restored.csv)
- `outputs/llm_predictions/`, `outputs/llm_cache/`: 4×599 nhãn và bằng chứng Qwen.
- `outputs/submissions/recipe.json`, `outputs/results/summary_table.csv`: bản cũ chờ P3.
- `scripts/audit_artifacts.py`: kiểm tra lại output; `scripts/repair_p1_results.py`: phục hồi bảng 05/đánh dấu trạng thái.
- `tests/`: 35 kiểm tra hiện tại; đã chạy sau sửa P1.
- `kaggle/merge_nb06_results.py` ở thư mục cha: nhập output Qwen sau tải về; **không chạy trên Kaggle**.
- Bundle Kaggle cũ chưa được đóng gói lại với sửa P1. Không chạy lại bundle cũ để ghi đè kết quả trên máy.
- Không chạy `../kaggle/nb06_tools/build_bundle.py` tùy tiện: hiện script có thể sinh đè notebook 06 đã chạy.
  Sao lưu trước nếu cần sửa/đóng gói lại.

## 9. Lịch và phạm vi nộp
Hạn môn chưa xác nhận. Tài liệu BTC đang lưu mô tả test EN 743 mẫu từ video khác, dự kiến tháng 01/2027;
kế hoạch cũ ghi đánh giá 10–31/01/2027 và system paper 02/2027. **Chưa xác minh lại các mốc này trong phiên 18/10.**
Trước khi nộp phải kiểm tra nguồn chính thức, quy cách header/tên file và giới hạn lượt nộp.
Nộp A và B riêng; không có yêu cầu tự suy ra rằng phải làm toàn bộ subtask/ngôn ngữ.

## 10. Ghi chép BTC từ roadmap trước
Giữ nguyên nội dung ghi chép gốc dưới đây để truy vết; không dùng cụm “2 tuần nữa” làm mốc hiện tại.



## Cập nhật tin tức từ BTC 
    - Thông tin về cuộc thi sắp tới: Thank you for your interest in our shared task. We are currently working on the evaluation/scoring script and on getting the three Codabench platforms ready (it will be three different pages for the three subtasks). We will make them available in approximately 2 weeks. As soon as they are ready we will announce it on the Google Group.
    - Release test triplets: Dear StereoQueerEval participants,

    We are happy to announce that we have decided to release 100 Persian (Farsi) text triplets (title, description, comments) as part of the development set. The remaining Persian data will be released as part of the test set in January, when the evaluation period starts.


    The archive containing the training data is password-protected. You can open it with the following password.

    Password Farsi Development Data: Rainbow2027!

    Once again, the new file is distributed as a password-protected RAR archive. We recommend extracting it with WinRAR on Windows; on macOS you can use the command-line version of WinRAR or a free tool that supports encrypted RAR archives, such as The Unarchiver or Keka; on Linux, unrar x archive.rar.

    Happy experimenting!

    Best,
    The StereoQueerEval Organizers 🌈