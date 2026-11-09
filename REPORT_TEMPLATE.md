# Báo cáo thực nghiệm: độ bền vững của object detector

## Tóm tắt

Nêu model, tập dữ liệu, các biến dạng được kiểm tra và phát hiện chính. Chỉ điền sau khi chạy thực nghiệm; báo cáo mAP clean/noisy và mức giảm tương đối.

## 1. Câu hỏi nghiên cứu

Làm rõ tác động của Gaussian noise, Salt-and-Pepper, fog tổng hợp và low-light tổng hợp lên mAP và false positive của detector. Đánh giá xem tiền xử lý tự động có khôi phục hiệu năng không.

## 2. Thiết lập

| Thuộc tính | Giá trị cần ghi |
|---|---|
| Dataset / phiên bản / split | |
| Số ảnh và số instance | |
| Model / checkpoint | |
| Ultralytics / PyTorch / CUDA | |
| Input size / batch / device | |
| Confidence cho confusion matrix | |
| Seed(s) | |
| Ngưỡng ghép IoU | |

## 3. Định nghĩa biến dạng và phục hồi

Mô tả chính xác mức noise theo README. Nêu công thức/giá trị level, seed, và preprocessing được áp dụng cho từng loại. Nếu thay tham số, cập nhật cả README và báo cáo.

## 4. Kết quả

### 4.1 mAP theo mức biến dạng

Chèn `results/map_vs_intensity.png`. Bảng tóm tắt:

| Điều kiện | Level | mAP@0.5 | mAP@0.5:0.95 | Δ mAP@0.5 so clean | Δ% |
|---|---:|---:|---:|---:|---:|
| Clean | 0 | | | 0 | 0 |

### 4.2 False positive và confusion matrix

Chèn `results/false_positives_vs_intensity.png`. Nêu ngưỡng confidence/IoU; chỉ ra lớp có FP/FN hoặc nhầm lẫn lớn nhất. Thêm các ma trận đại diện từ `results/confusion_<condition>.png`.

### 4.3 Khả năng phục hồi

So raw với recovered tại cùng loại và level. Ghi thay đổi mAP, FP, FN; lưu ý preprocessing có thể giúp loại noise này nhưng làm giảm chi tiết/hiệu năng ở loại khác.

## 5. Thảo luận

Giải thích khác biệt theo loại nhiễu, độ nhạy theo cường độ và lớp đối tượng. Phân biệt quan sát đo được với nguyên nhân suy đoán. Nêu giới hạn của fog đồng nhất/low-light tổng hợp và giá trị của ảnh thực địa.

## 6. Kết luận

Trả lời câu hỏi nghiên cứu theo số liệu thực nghiệm, nêu preprocessing hữu ích nhất và điều kiện mà nó không giúp.

## 7. Tái lập

Ghi lệnh thực tế, hash/đường dẫn checkpoint, phiên bản package, seed và vị trí `metrics.csv`, manifest cùng biểu đồ.
