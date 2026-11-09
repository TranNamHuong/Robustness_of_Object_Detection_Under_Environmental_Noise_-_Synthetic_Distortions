# Kế hoạch và trạng thái dự án

## Trạng thái hiện tại

Giai đoạn benchmark và báo cáo cốt lõi đã hoàn tất. COCO val2017 subset 500 ảnh được chạy đủ 25 điều kiện raw/recovered qua ba seed noise với YOLOv8n; full val2017 có sweep raw 12 điều kiện, recovery đã chọn lọc cho YOLOv8n và SSDLite, cùng so sánh hai detector bằng evaluator AP chung. YOLOv8s sau đó được so sánh trên cùng COCO full-val và hiện là model mặc định của desktop MVP `app.py`, vì mAP tăng 9.12 pp với độ trễ tăng khoảng 0.57 ms/ảnh trên RTX 3060. App mở ảnh, xem biến dạng/recovery, chạy detection và lưu kết quả. Chưa fine-tune; cần ảnh mục tiêu có nhãn để quyết định cho domain sản phẩm. Xem `TARGET_DOMAIN_EVALUATION.md` để chuẩn bị và chạy baseline.

## Thiết kế đề xuất

| Thành phần | Đề xuất ban đầu | Lý do |
|---|---|---|
| Nhiệm vụ | Object detection, đánh giá zero-shot | Đo trực tiếp mức suy giảm của detector pretrained khi ảnh đầu vào bị biến dạng |
| Dataset | COCO val2017, lấy mẫu cố định 500 ảnh | Nhãn COCO phù hợp trực tiếp với detector pretrained COCO; 500 ảnh giúp phép đo khả thi hơn toàn bộ validation set |
| Model chính | YOLOv8n pretrained trên COCO | Nhẹ, dễ tái lập, phù hợp mục tiêu phân tích trung cấp |
| Model đối chiếu (nếu còn thời gian) | SSD-MobileNet pretrained trên COCO | So sánh liệu xu hướng có phụ thuộc kiến trúc hay không |
| Điều kiện | Clean; Gaussian; Salt-and-Pepper; fog; low-light | Bao phủ hai loại noise số và hai điều kiện môi trường tổng hợp |
| Mức độ | 3 mức cho mỗi loại biến dạng | Đủ để vẽ xu hướng intensity–performance mà không làm số lượt đánh giá quá lớn |
| Phục hồi | Median cho Salt-and-Pepper; Non-Local Means cho Gaussian; CLAHE cho low-light; CLAHE làm baseline cho fog | Ghép phương pháp với loại nhiễu, đồng thời giữ cùng detector và ground truth |
| Chỉ số | mAP@0.5, mAP@0.5:0.95, precision/recall, FP/FN và confusion matrix | Vừa đo chất lượng tổng quát vừa cho thấy kiểu lỗi |
| Lặp lại | Chọn subset và seed cố định; lặp 3 seed nhiễu | Báo cáo trung bình và độ lệch chuẩn thay vì phụ thuộc một lần tạo nhiễu |

Các giá trị 500 ảnh và 3 seed là cấu hình khởi điểm, không phải yêu cầu cứng. Nếu tài nguyên cho phép, tăng cỡ mẫu; nếu thời gian hạn chế, giảm số ảnh nhưng giữ subset cố định và báo cáo giới hạn đó.

## Định nghĩa câu hỏi và biến đo

**Câu hỏi chính:** Khi tăng cường độ của từng biến dạng, mAP của YOLOv8n pretrained trên COCO thay đổi ra sao so với ảnh clean, và tiền xử lý có khôi phục được bao nhiêu hiệu năng?

- **Biến độc lập:** loại biến dạng, mức cường độ, và trạng thái phục hồi (raw/recovered).
- **Biến phụ thuộc chính:** mAP@0.5 và mAP@0.5:0.95.
- **Biến phụ thuộc phụ:** precision/recall, FP và FN tại một ngưỡng confidence được cố định trước.
- **Biến kiểm soát:** cùng ảnh, nhãn, checkpoint, kích thước đầu vào, NMS, thiết bị, ngưỡng đo và tập con giữa các điều kiện.
- **Mức phục hồi:** `mAP_recovered − mAP_raw` tại cùng ảnh, loại nhiễu và cường độ.
- **Suy giảm tương đối:** `100 × (mAP_clean − mAP_condition) / mAP_clean`.

## Các giai đoạn thực hiện

### 1. Chốt dữ liệu và phạm vi

- Chọn COCO val2017 làm benchmark chính hoặc thống nhất một dataset khác.
- Tạo manifest cố định gồm danh sách ảnh và annotation; ghi rõ cách lấy mẫu.
- Chọn số ảnh theo tài nguyên máy và xác định class subset nếu cần.
- Kiểm tra license, dung lượng, và cách tải dataset trước khi tải dữ liệu lớn.

**Hoàn thành khi:** biết chính xác dataset/split, số ảnh, đường dẫn annotation và quy tắc subset.

### 2. Chuẩn hóa dữ liệu và baseline clean

- Chuyển annotation sang định dạng mà pipeline đánh giá sử dụng hoặc bổ sung bộ đọc COCO JSON trực tiếp.
- Chạy detector trên ảnh clean.
- Lưu checkpoint, cấu hình, log phiên bản và metrics baseline.

**Hoàn thành khi:** pipeline đọc được ảnh/annotation và clean mAP tái lập được.

### 3. Sinh và kiểm tra biến dạng

- Tạo các mức Gaussian, Salt-and-Pepper, fog và low-light theo định nghĩa trong README.
- Xem trực quan một nhóm ảnh cho từng mức để phát hiện biến đổi quá nhẹ/quá mạnh.
- Giữ nguyên annotation và gắn seed/metadata vào manifest.

**Hoàn thành khi:** mỗi mức có ảnh đầu ra, nhãn tương ứng, định nghĩa rõ và kiểm tra trực quan đạt.

### 4. Đánh giá raw và recovered

- Chạy cùng checkpoint trên mọi điều kiện raw và recovered.
- Ghi mAP, precision/recall, FP/FN, confusion matrix và thời gian suy luận.
- Lặp lại nhiễu với các seed đã định trước.

**Hoàn thành khi:** có bảng dữ liệu thô đầy đủ để tính trung bình/độ lệch chuẩn.

### 5. Phân tích và viết báo cáo

- Vẽ mAP và FP/FN theo cường độ, so sánh raw/recovered.
- Phân tích theo class và loại nhiễu; nêu rõ các trường hợp phục hồi làm giảm hiệu năng.
- Viết giới hạn: biến dạng tổng hợp, detector/dataset cụ thể, ngưỡng confidence, và cỡ mẫu.

**Hoàn thành khi:** mỗi kết luận đều liên kết tới số liệu/biểu đồ và giới hạn được nêu rõ.

## Việc còn lại để nâng chất lượng nghiên cứu

Các đầu ra benchmark cốt lõi đã hoàn thành. Các bước tiếp theo được xếp theo giá trị khoa học:

1. **Giữ riêng test split:** chọn phương pháp/siêu tham số khử nhiễu trên validation, sau đó báo cáo kết quả cuối trên split độc lập để giảm rò rỉ lựa chọn.
2. **Đo biến thiên theo ảnh, kích thước, lớp và ngưỡng confidence:** đã chạy 9 điều kiện nặng trên ba subset COCO rời nhau; size-stratified recall cho đủ 9 điều kiện, class-wise recall cho cả bốn họ raw/recovered, và FP/FN tại ba confidence threshold. Có thêm lặp seed42/43/44 cho Gaussian/S&P. Full 25-condition curve vẫn chỉ có trên subset gốc; hiện không mở rộng thêm theo phạm vi đã chốt. SD theo subset đo biến thiên do chọn ảnh trong cùng COCO val2017; SD theo seed đo biến thiên do noise trên subset cố định.
3. **Bổ sung Gaussian-NLM full-val nếu cần đường recovery đầy đủ:** NLM đã có cả ba mức trên subset 500 ảnh và mức 0.06 trên full val; bilateral đã có cả ba mức full-val cho hai detector. Hai mức NLM full-val nhẹ hơn bị giới hạn bởi thời gian xử lý CPU.
4. **Kiểm tra module denoising trên dữ liệu COCO hiện có:** đã tạo panel clean/raw/recovered cho bốn họ biến dạng trên cùng ảnh và thêm policy chọn preprocessing theo level benchmark đã đo. Policy được tinh chỉnh trên validation, cần xác nhận độc lập trước khi khái quát hóa.
5. **Đóng gói theo yêu cầu môn học:** nguồn tham khảo chính đã được thêm vào báo cáo Markdown; desktop MVP đã có. Chuyển báo cáo sang định dạng nộp và chuẩn bị slide/poster nếu rubric yêu cầu.

## Rủi ro phương pháp cần tránh

- Không dùng tập train để đánh giá robustness.
- Không so raw và recovered trên hai subset khác nhau.
- Không thay đổi checkpoint hoặc ngưỡng confidence giữa các điều kiện.
- Không gọi fog đồng nhất/low-light tổng hợp là đánh giá ngoài trời hoặc ban đêm thực tế.
- Không đưa ra kết luận chung cho mọi detector từ chỉ một model.
- Nếu chọn phương pháp phục hồi dựa trên kết quả validation, cần có test split riêng để xác nhận.
