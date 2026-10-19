# P4 trên Kaggle — lưu/nạp model và kiểm tra pipeline

Trạng thái chuẩn bị ngày **18/10/2026**: đã có code/notebook/bundle và kiểm thử nhỏ trên CPU. **Chưa chạy model thật trên Kaggle, chưa hoàn thành P4.** Giữ nguyên recipe P3, kết quả cũ và 30 ghi chú P2.

## 1. P4 cần làm gì?

P3 đã chọn cách kết hợp model, nhưng chưa có trọng số model lưu trên đĩa. P4 train các thành phần đã chọn, lưu trọng số/tokenizer/config, mở tiến trình mới để nạp lại offline, rồi kiểm tra dự đoán và TSV.

Cần **7 checkpoint**: 5 transformer (MT, MT có class weight, RoBERTa+title, RoBERTa+title+description, Twitter-RoBERTa B) và 2 TF-IDF A. Hai model MT có cả đầu A/B nên chỉ train một lần mỗi model. Không chạy lại 5-fold CV, notebook 01–08, Qwen hoặc DeBERTa.

| Chế độ | Train | Dự đoán để kiểm tra | Epoch transformer | Ý nghĩa |
|---|---:|---:|---:|---|
| `smoke` (mặc định) | 64 mẫu từ fold 1–4 | 16 mẫu fold 0 | 1 | Thử code, GPU, save/reload; không đánh giá chất lượng |
| `dryrun` | 2.390 mẫu fold 1–4 | 599 mẫu fold 0 | 4 | Kiểm tra đầy đủ pipeline với recipe đã chốt |
| `final` (lượt riêng khi cần) | 2.989 mẫu | 32 mẫu đã có trong train | 4 | Lưu model train toàn bộ; probe chỉ kiểm tra save/reload |

Điểm dryrun không phải test độc lập: recipe P3 đã được chọn trên dữ liệu phát triển có fold 0. Không so nó với điểm CV rồi kết luận chất lượng tăng/giảm chỉ vì khác nhau. Final chưa có test BTC nên không báo cáo điểm chất lượng trên probe train.

## 2. Hai file cần upload

- [sqe_p4_bundle.zip](sqe_p4_bundle.zip): dataset đầu vào, khoảng 0,66 MiB; chứa EN, split, code, recipe và các config nguồn cần thiết.
- [kaggle_runner_p4.ipynb](kaggle_runner_p4.ipynb): notebook có 5 ô code, chạy theo thứ tự.

File `build_p4_bundle.py` chỉ dùng trên máy để đóng gói lại khi có sửa code, không chạy trên Kaggle. Không dùng bundle 04–08 hoặc 06 cũ cho P4.

## 3. Tạo notebook và chạy hai ô đầu

1. Kaggle → Datasets → New Dataset: upload **sqe_p4_bundle.zip**, đặt tên ví dụ `sqe-p4-bundle`, để **Private** vì có dữ liệu BTC thuộc thỏa thuận sử dụng.
2. Tạo notebook Python, Import Notebook → **kaggle_runner_p4.ipynb**. Giữ notebook Private.
3. Add Input → gắn dataset `sqe-p4-bundle`.
4. Settings → Accelerator → GPU T4 nếu có; nếu được cấp T4×2, code chỉ dùng `cuda:0`. Internet → On để tải pretrained model.
5. **Chưa bấm Run All. Chạy ô code 1, rồi ô code 2.**

Ô 1 phải in tên GPU, Torch/CUDA và xác nhận PyTorch nhận GPU. Nếu không có `nvidia-smi`, notebook dừng với chỉ dẫn bật GPU. Ô 2 phải in số mẫu EN **2989** và **Bundle hash: OK**. Có lỗi thì dừng ở đây, gửi log để xử lý trước khi train.

## 4. Smoke trước, dryrun sau

Khi hai ô đầu đạt:

1. Ô 3: giữ `MODE="smoke"`, `RESUME_RUN=""`. Kế hoạch phải có 7 component, train=64, probe=16.
2. Ô 4: train nhỏ từng component. Sau khi lưu, script tự mở **tiến trình Python mới**, dùng cache Hugging Face rỗng và offline để nạp model/tokenizer đã lưu.
3. Ô 5: kiểm tra `status="complete"`, `fresh_process=true`, `offline_reload=true`; cả 9 kiểm tra đầu A/B của 7 component phải có `labels_equal=true` và `probabilities_close=true`. Notebook tạo **sqe_p4_smoke.zip**.
4. **Chỉ khi smoke đạt**, đổi ô 3 thành `MODE="dryrun"`. Có thể Save Version → Save & Run All để chạy lượt này. Đầu ra đặt riêng, không ghi đè smoke.
5. Kết thúc dryrun: phải có 7 checkpoint, báo cáo reload đạt và 2 TSV, mỗi file **599 ID**. Tải **sqe_p4_dryrun.zip** trong Output.

Không dùng điểm smoke để chọn cấu hình; smoke chỉ chạy 1 epoch và 64 mẫu. Chưa đo runtime P4 thực tế trên T4; xem thời gian từng component được in trong log. Lượt đầu có thêm thời gian tải pretrained model.

## 5. Khi cần train full

Sau khi dryrun đã được kiểm chứng, đổi `MODE="final"` trong một lượt riêng. Giữ nguyên recipe, 4 epoch và seed 42.

Kết quả **sqe_p4_final.zip** chứa 7 checkpoint train trên toàn bộ EN. Hai file final_probe chỉ kiểm tra nạp lại, **không phải file nộp BTC**. Có thể hoãn lượt final đến khi cần dùng model; việc chạy nhiều lần không tự làm kết quả tốt hơn.

Phần dự đoán test chính thức và định dạng BTC sẽ được kiểm tra khi có dữ liệu/yêu cầu thật. Notebook P4 không tự nộp lên Codabench.

## 6. Output cần giữ và gửi lại

Mỗi ZIP chứa `SemEval2027_Task8/outputs/p4/<mode>/` với:

- `status.json`, `reload_report.json`: trạng thái và so sánh trước/sau nạp.
- `plan.json`: recipe, cấu hình, ID train/probe, hash code/dữ liệu và phiên bản thư viện.
- `checkpoints/`: transformer có model.pt + encoder_config + tokenizer + checkpoint.json; TF-IDF có pipeline.joblib + checkpoint.json.
- `reference/`: xác suất trước khi lưu để đối chiếu.
- `predictions/`: TSV đã kiểm tra ID, thứ tự và nhãn.
- Code đi kèm để tái tạo thao tác nạp lại.

Giải nén ZIP vào **thư mục riêng**, ví dụ Downloads/ket_qua_p4_dryrun, rồi đưa đường dẫn để kiểm tra. Không replace toàn bộ dự án hoặc chép bảng master cũ đè lên P3. Bản ZIP cũng cần được giữ Private.

## 7. Chạy tiếp khi ngắt phiên

Script chỉ bỏ qua model đã có checkpoint hoàn chỉnh, hash/identity khớp và có xác suất tham chiếu trước lưu. Model đang train dở sẽ bắt đầu lại; chưa lưu optimizer để resume giữa epoch.

- Nếu còn cùng session và output, chạy lại ô 4.
- Nếu có output của phiên trước: thêm nó dưới dạng input riêng, điền `RESUME_RUN` trỏ tới thư mục có `plan.json`, chọn đúng MODE, rồi chạy ô 3–5.
- Nếu báo khác code/config/environment, dừng và gửi log; không sửa hash để ép bỏ qua kiểm tra. Phiên bản thư viện và GPU phải phù hợp cho phép so sánh trước/sau nạp nghiêm ngặt.
- Nếu GPU OOM hoặc loss không hữu hạn, gửi dòng log lỗi. Chưa tự đổi batch/epoch: chúng thuộc cấu hình đã chốt, và mọi điều chỉnh cần được ghi nhận.

## 8. Khi nào đóng P4?

Đã có dryrun Kaggle thật với đủ checkpoint, nạp offline trong tiến trình mới đạt, xác suất gần nhau (`atol=1e-5`, `rtol=1e-4`), nhãn giống hệt và 2 TSV đúng 599 ID; output được tải về và kiểm chứng. Final train toàn bộ là mốc riêng khi cần. Chuẩn bị notebook hoặc smoke thành công chưa đủ để đóng P4.

Tài liệu kỹ thuật đã đối chiếu: [Kaggle notebooks](https://www.kaggle.com/docs/notebooks), [PyTorch save/load state_dict](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html), [Hugging Face local model loading](https://huggingface.co/docs/transformers/en/main_classes/model).
