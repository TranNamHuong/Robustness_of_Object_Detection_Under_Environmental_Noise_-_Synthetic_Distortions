> Cập nhật 2026-11-08: tập ảnh người dùng `target_eval/` đã bị xóa theo yêu cầu. Tập external hiện có là `datasets/open_images_v7/` (100 ảnh, 193 box, 8 lớp trùng); đã có pilot mAP@0.5 YOLOv8n 65.8%, YOLOv8s 65.4%, SSD-MobileNet 61.9% ở imgsz 320. Đây là tập nhỏ theo domain ảnh Open Images, chưa đại diện đầy đủ cho camera sản phẩm. Xem README của tập để biết metric và giới hạn.

# Hướng dẫn đánh giá trên ảnh mục tiêu

## Mục đích

Đo xem YOLO pretrained COCO có nhận diện đủ tốt trong môi trường sản phẩm dự kiến hay cần fine-tune. Ảnh ngoài COCO có thể chạy qua model, nhưng muốn đo độ chính xác phải có ground-truth boxes và class labels. Confidence score trên ảnh không nhãn không phải precision/accuracy. Ứng dụng hiện mặc định dùng YOLOv8s; YOLOv8n giữ làm model nhẹ để đối chiếu.

## Dữ liệu cần có

Chuẩn bị ảnh thực tế đại diện cho camera, góc nhìn, ánh sáng và khoảng cách mà sản phẩm sẽ gặp. Đặt ảnh và YOLO labels theo cấu trúc:

```text
target_eval/
  images/val/
  labels/val/
  data.yaml
```

Mỗi ảnh cần file `.txt` cùng tên trong `labels/val`. Mỗi dòng nhãn theo dạng `class_id x_center y_center width height`, tọa độ chuẩn hóa 0–1. Với các lớp thuộc COCO, giữ nguyên ID 0–79; sao chép nguyên danh sách `names` COCO từ `datasets/coco/full_val2017/data.yaml`. Không đánh lại ID theo subset lớp.

Nếu bạn có ảnh nhưng chưa có nhãn, có thể tạo nhãn nháp bằng YOLOv8s để giảm thao tác vẽ box. Chép ảnh vào `target_eval/images/val`, rồi chạy:

```powershell
.\.venv\Scripts\python.exe draft_auto_labels.py --source target_eval/images/val --out runs/target_auto_label_drafts
```

Mở các ảnh trong `runs/target_auto_label_drafts/previews` và kiểm tra từng box/lớp. File nháp nằm riêng trong `runs/target_auto_label_drafts/draft_labels`; sửa chúng bằng công cụ gán nhãn YOLO rồi đặt các nhãn đã kiểm tra vào `target_eval/labels/val`. Không dùng nhãn nháp chưa xác minh để tính mAP: đó là dự đoán của chính model, không phải ground truth độc lập.

Có thể sửa trực tiếp bằng giao diện gán nhãn cục bộ đã thêm:

```powershell
.\.venv\Scripts\python.exe annotate_target_images.py
```

Chọn box để đổi class, xóa box, hoặc bật **Thêm box** rồi kéo chuột để vẽ một box mới. Các lớp `bed` và `couch` vẫn tách biệt. Nút lưu ghi nhãn hiện tại vào `target_eval/labels/val`; chỉ lưu sau khi đã kiểm tra từng box và thêm các vật thể bị model bỏ sót.

Giữ các ảnh đánh giá riêng, không dùng để fine-tune. Một pilot nhỏ giúp xem ví dụ lỗi, nhưng kết luận định lượng cần đủ instances cho từng lớp mục tiêu; ghi số ảnh và support mỗi lớp.

Trước khi đánh giá hoặc huấn luyện, chạy kiểm tra cặp ảnh/nhãn, tọa độ bbox, class IDs và trùng ảnh giữa các split:

```powershell
.\.venv\Scripts\python.exe validate_yolo_dataset.py --data D:/path/to/target_eval/data.yaml
```

Nếu giữ nguyên ID COCO 0–79, thêm `--require-coco80`. Script chấp nhận split validation-only khai báo cùng thư mục ở `train` và `val`, đồng thời báo rõ đây là alias; không dùng cấu hình alias đó để train.

## Chạy baseline pretrained

Sửa `path` trong `data.yaml` trỏ tới thư mục `target_eval`. Với validation-only dataset, dùng `train: images/val` theo schema Ultralytics; lệnh dưới đây chỉ chạy đánh giá, không huấn luyện.

```powershell
.\.venv\Scripts\python.exe robustness_eval.py `
  --data D:/path/to/target_eval/data.yaml `
  --model D:/path/to/project/yolov8s.pt `
  --only-clean --no-recovery `
  --out runs/target_domain_eval `
  --imgsz 640 --batch 16 --device 0 --fp-conf 0.25 --match-iou 0.5
```

CPU fallback: replace `--device 0` with `--device cpu`. Kết quả chính nằm trong `runs/target_domain_eval/metrics.csv`, confusion matrix và thư mục `ultralytics/`.

Muốn đo trade-off trên domain mục tiêu, lặp cùng lệnh với `--model D:/path/to/project/yolov8n.pt --out runs/target_domain_eval_yolov8n` rồi so sánh mAP/FP/FN và tốc độ.

## Quyết định fine-tune

- Nếu cần lớp ngoài 80 lớp COCO, pretrained COCO không thể cung cấp detector có nhãn đúng cho lớp đó; cần dữ liệu gán nhãn và fine-tune/custom train.
- Nếu lớp đã thuộc COCO, xem mAP, recall và confusion matrix theo lớp trên ảnh mục tiêu. Chỉ fine-tune khi chất lượng không đạt yêu cầu sản phẩm đã đặt ra hoặc lỗi tập trung ở domain cần cải thiện.
- Sau fine-tune, đánh giá lại trên một tập ảnh giữ riêng chưa dùng chọn checkpoint hay tham số. So sánh pretrained và fine-tuned cùng ảnh, cùng ngưỡng và cùng evaluator.

## Trạng thái hiện tại

Benchmark COCO val2017 vẫn là baseline tổng quát: ở input 640, YOLOv8n mAP@0.5 0.5185 / mAP@0.5:0.95 0.3671; YOLOv8s 0.6096 / 0.4423. Tập Open Images cung cấp một phép đo external bổ sung; chưa thể kết luận độ chính xác trên chính môi trường camera sản phẩm hoặc quyết định fine-tune chỉ từ 100 ảnh này.

