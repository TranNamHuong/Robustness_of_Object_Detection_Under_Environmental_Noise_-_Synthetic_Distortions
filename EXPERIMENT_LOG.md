# Nhật ký thực nghiệm

## Baseline clean — CPU smoke run — 2026-11-07

| Thuộc tính | Giá trị |
|---|---|
| Dataset | COCO val2017 subset, 500 ảnh, seed 42 |
| Nhãn | 3.799 box; 79/80 lớp xuất hiện; 6 ảnh không có box |
| Model | YOLOv8n pretrained COCO (`yolov8n.pt`) |
| Ultralytics | 8.4.204 |
| PyTorch | 2.10.0+cpu |
| Phần cứng | Intel Core i5-12400F; RTX 3060 hiện có nhưng chưa được dùng do PyTorch CPU-only |
| Kích thước ảnh | 640 |
| Ngưỡng confidence / NMS cho validation | 0.001 / IoU 0.7 |
| Ngưỡng confusion matrix thủ công | Confidence 0.25, IoU match 0.5 |

### Kết quả

| Chỉ số | Giá trị |
|---|---:|
| mAP@0.5 | 0.5442 |
| mAP@0.5:0.95 | 0.3894 |
| Precision (Ultralytics) | 0.6255 |
| Recall (Ultralytics) | 0.5000 |
| TP tại IoU 0.5, confidence 0.25 | 1.885 |
| FP tại IoU 0.5, confidence 0.25 | 881 |
| FN tại IoU 0.5, confidence 0.25 | 1.914 |

Kết quả này là lần chạy ban đầu trên subset ngẫu nhiên 500 ảnh, không phải COCO val2017 đầy đủ và không đại diện mAP COCO chính thức. Một lớp không có ground truth trong subset nên không được đo ở per-class AP. Các điều kiện nhiễu chưa được chạy.

### Artefak

- `runs/baseline_clean_cpu/metrics.csv`
- `runs/baseline_clean_cpu/confusion_clean.csv`
- `runs/baseline_clean_cpu/confusion_clean.png`
- `runs/baseline_clean_cpu/ultralytics/clean/`

## Baseline clean — GPU run — 2026-11-07

Giữ nguyên dataset, checkpoint, ảnh đầu vào và ngưỡng như CPU run; chuyển sang PyTorch CUDA 12.8 trên RTX 3060. Model inference chạy khoảng 3.4 ms/ảnh theo log Ultralytics.

| Chỉ số | Giá trị |
|---|---:|
| mAP@0.5 | 0.5440 |
| mAP@0.5:0.95 | 0.3894 |
| Precision (Ultralytics) | 0.6202 |
| Recall (Ultralytics) | 0.4966 |
| TP tại IoU 0.5, confidence 0.25 | 1.896 |
| FP tại IoU 0.5, confidence 0.25 | 894 |
| FN tại IoU 0.5, confidence 0.25 | 1.903 |

CPU/GPU mAP chênh rất nhỏ; precision và bộ đếm TP/FP/FN khác nhẹ. Dùng cùng device GPU cho các điều kiện tiếp theo để giảm sai khác do backend.

Artefak: `runs/baseline_clean_gpu/metrics.csv`, `runs/baseline_clean_gpu/confusion_clean.csv`, `runs/baseline_clean_gpu/confusion_clean.png`, và `runs/baseline_clean_gpu/ultralytics/clean/`.

## Đánh giá độ bền vững — sweep GPU — 2026-11-07

| Thuộc tính | Giá trị |
|---|---|
| Dataset | Cùng COCO val2017 subset 500 ảnh, seed 42; 3.799 box |
| Model / checkpoint | YOLOv8n pretrained COCO (`yolov8n.pt`), không fine-tune |
| Phần cứng | NVIDIA RTX 3060 12 GB, CUDA 12.8; PyTorch 2.10.0+cu128; Ultralytics 8.4.204 |
| Kích thước / batch | 640 / 16 |
| Điều kiện | 1 clean + 4 loại biến dạng × 3 mức × raw/recovered = 25 hàng |
| Khôi phục | Gaussian: Fast NLM; Salt-and-Pepper: median 3×3; fog/low-light: CLAHE trên Lab-L |
| Đếm FP/FN | Confidence 0.25, ghép greedy IoU 0.5; mAP validation dùng cấu hình Ultralytics |

### Kết quả chính

Baseline clean trên GPU: **mAP@0.5 = 0.5440**, **mAP@0.5:0.95 = 0.3894**, 894 FP và 1.903 FN tại ngưỡng đếm nêu trên.

- **Salt-and-Pepper gây suy giảm mạnh nhất:** mAP@0.5 lần lượt 0.4618, 0.3673, 0.2757 tại mức 0.01, 0.03, 0.06 (giảm tối đa 49.3% tương đối so với clean). Median filter nâng các mức đó lên 0.5220, 0.5152, 0.5202. Ở mức 0.06, FP tăng từ 521 lên 816 sau lọc; lọc phục hồi nhiều TP nhưng cũng đưa thêm dự đoán qua ngưỡng confidence.
- **Gaussian:** raw mAP@0.5 là 0.5426, 0.5279, 0.4621 ở sigma/255 = 0.01, 0.03, 0.06. NLM không cải thiện hai mức thấp (0.5160 và 0.5101), chỉ tăng nhẹ mức cao lên 0.4654.
- **Fog đồng nhất:** raw mAP@0.5 là 0.5346, 0.5266, 0.4928 với alpha 0.15, 0.30, 0.50. CLAHE giảm nhẹ ở mức 0.15, gần như trung tính ở 0.30, và tăng lên 0.5228 ở mức 0.50.
- **Low-light:** level là hệ số nhân độ sáng (level thấp hơn nghĩa là tối hơn). Raw mAP@0.5 là 0.5425 ở 0.60, 0.5260 ở 0.40 và 0.4798 ở 0.25. CLAHE đạt 0.5331, 0.5283, 0.5105 tương ứng; cải thiện lớn nhất ở mức tối nhất là +0.0308 mAP.

Kết quả sơ bộ cho thấy hiệu quả khôi phục phụ thuộc loại và mức biến dạng: median lọc tốt Salt-and-Pepper, CLAHE giúp fog/low-light nặng, còn NLM cấu hình hiện tại không phù hợp với Gaussian nhẹ-vừa. FP tại ngưỡng cố định không tăng đồng đều theo mức nhiễu; cần xem cùng recall/FN và mAP, không dùng FP đơn lẻ làm kết luận.

### Artefak

- `runs/noise_sweep_seed42/metrics.csv` — đủ 25 điều kiện, kết quả cuối.
- `runs/noise_sweep_seed42/map_vs_intensity.png` — đường cong mAP raw/recovered.
- `runs/noise_sweep_seed42/false_positives_vs_intensity.png` — FP theo mức biến dạng.
- `runs/noise_sweep_seed42/confusion_<condition>.png` và `.csv` — confusion matrix cho từng điều kiện.
- `runs/noise_sweep_seed42/datasets/` — ảnh biến dạng và ảnh sau khôi phục dùng cho lần đánh giá này.

### Giới hạn và bước tiếp theo

Đây là kết quả thăm dò trên một subset ngẫu nhiên 500 ảnh và một seed, không phải mAP COCO chính thức. Subset không có toothbrush trong ground truth (79/80 lớp), nên chưa đại diện đủ lớp. Các điều kiện là biến dạng tổng hợp; fog đồng nhất không dùng depth, và low-light dùng phép biến đổi gamma đơn giản. Để củng cố kết luận, chạy thêm seed/subset hoặc toàn bộ val2017, cố định cách chọn ảnh giữa các lần lặp, báo trung bình ± độ lệch chuẩn và xem AP theo lớp. Không huấn luyện lại detector trong sweep này.

## Lặp lại với seed 43 và thống kê gộp — 2026-11-07

Đã chạy lại đủ 25 điều kiện trên cùng 500 ảnh, cùng YOLOv8n, RTX 3060, imgsz 640 và batch 16; chỉ đổi seed tạo Gaussian/Salt-and-Pepper từ 42 sang 43. Hai lần chạy có 50 dòng tổng cộng. Fog và low-light là phép biến đổi tất định, nên cùng dữ liệu tạo cùng chỉ số ở hai lần chạy; độ lệch chuẩn 0 ở các điều kiện này không biểu thị độ bất định trên nhiều ảnh/subset.

Trên hai seed, mAP@0.5 trung bình ở Salt-and-Pepper 0.06 là **0.2737 ± 0.0029**; median recovery đạt **0.5186 ± 0.0023**. Gaussian 0.03 raw đạt **0.5238 ± 0.0058**, sau NLM **0.5088 ± 0.0019**. Clean giữ nguyên 0.5440 ở cả hai lần vì không có nhiễu ngẫu nhiên.

- Lần chạy seed 43: `runs/repeat_seed43/metrics.csv`, `runs/repeat_seed43/confusion_*.png/.csv`.
- Bảng theo từng seed: `runs/repeat_analysis_seed42_43/metrics_by_seed.csv`.
- Trung bình và độ lệch chuẩn mẫu: `runs/repeat_analysis_seed42_43/metrics_summary.csv`.
- Biểu đồ mean ± SD: `runs/repeat_analysis_seed42_43/map_vs_intensity_mean_std.png` và `false_positives_mean_std.png`.
- Lệnh gộp có thể chạy lại: `aggregate_repeats.py`.

Hai seed trên cùng một subset giúp kiểm tra độ nhạy với mẫu noise ngẫu nhiên, nhưng chưa thay thế việc lặp trên nhiều subset hoặc benchmark toàn bộ val2017.

## Mở rộng toàn bộ COCO val2017 — raw only — 2026-11-08

Chuyển từ subset lên đủ 5.000 ảnh val2017, đủ 80 lớp và 36.335 instances; cùng YOLOv8n, imgsz 640, batch 16, RTX 3060, seed 42. Đã chạy clean cùng 12 điều kiện raw; mỗi điều kiện có confusion matrix. Phần full-val này chưa chạy recovery.

Clean: mAP@0.5 **0.5185**, mAP@0.5:0.95 **0.3671**, TP 17,921, FP 8,794, FN 18,414.

| Điều kiện | Level | mAP@0.5 | mAP@0.5:0.95 | Δ% so clean | FP | FN |
|---|---:|---:|---:|---:|---:|---:|
| gaussian | 0.01 | 0.5161 | 0.3651 | -0.5% | 8,639 | 18,597 |
| gaussian | 0.03 | 0.4927 | 0.3449 | -5.0% | 8,094 | 19,469 |
| gaussian | 0.06 | 0.4387 | 0.3037 | -15.4% | 7,098 | 21,476 |
| salt_pepper | 0.01 | 0.4337 | 0.3021 | -16.3% | 7,440 | 21,223 |
| salt_pepper | 0.03 | 0.3464 | 0.2344 | -33.2% | 6,237 | 24,100 |
| salt_pepper | 0.06 | 0.2459 | 0.1636 | -52.6% | 5,212 | 27,102 |
| fog | 0.15 | 0.5108 | 0.3615 | -1.5% | 8,592 | 18,743 |
| fog | 0.3 | 0.4985 | 0.3521 | -3.9% | 8,327 | 19,181 |
| fog | 0.5 | 0.4712 | 0.3320 | -9.1% | 7,705 | 20,339 |
| low_light | 0.25 | 0.4591 | 0.3224 | -11.4% | 6,233 | 21,338 |
| low_light | 0.4 | 0.4965 | 0.3509 | -4.2% | 7,513 | 19,751 |
| low_light | 0.6 | 0.5120 | 0.3639 | -1.2% | 8,091 | 18,945 |

Suy giảm lớn nhất là Salt-and-Pepper 0.06: mAP@0.5 0.2459 (-52.6% tương đối), mAP@0.5:0.95 0.1636. Gaussian 0.06 giảm 15.4%; fog 0.50 giảm 9.1%; low-light 0.25 giảm 11.4%. FP giảm ở các điều kiện nặng đồng thời FN tăng, do nhiều vật thể bị bỏ sót.

Tệp: `datasets/coco/full_val2017/data.yaml`, `datasets/coco/full_val2017/manifest.json`, `runs/full_val2017_raw_seed42/metrics.csv`, `map_vs_intensity.png`, `false_positives_vs_intensity.png`, và `confusion_<condition>.png/.csv`.

Lệnh tái lập raw-only: `python robustness_eval.py --data datasets/coco/full_val2017/data.yaml --model yolov8n.pt --out runs/full_val2017_raw_seed42 --device 0 --imgsz 640 --batch 16 --seed 42 --no-recovery`.

## Recovery chọn lọc trên COCO val2017 đầy đủ — 2026-11-08

Chạy 9 recovered-only điều kiện trên 5.000 ảnh cho salt_pepper, fog và low_light (3 level mỗi họ), cùng nhãn và seed 42. Sau đó chạy thêm Gaussian-NLM ở full-val level 0.06; chi tiết ở mục tiếp theo.

| Condition | Raw mAP50 | Recovered mAP50 | Δ mAP50 | FP raw→rec | FN raw→rec |
|---|---:|---:|---:|---:|---:|
| salt_pepper_0.01 | 0.4337 | 0.4997 | +0.0659 | 7440→8334 | 21223→19188 |
| salt_pepper_0.03 | 0.3464 | 0.4985 | +0.1521 | 6237→8255 | 24100→19281 |
| salt_pepper_0.06 | 0.2459 | 0.4936 | +0.2478 | 5212→8168 | 27102→19438 |
| fog_0.15 | 0.5108 | 0.5079 | -0.0029 | 8592→8615 | 18743→18739 |
| fog_0.3 | 0.4985 | 0.5072 | +0.0087 | 8327→8472 | 19181→18842 |
| fog_0.5 | 0.4712 | 0.4998 | +0.0286 | 7705→8144 | 20339→19224 |
| low_light_0.6 | 0.5120 | 0.5087 | -0.0033 | 8091→8271 | 18945→18969 |
| low_light_0.4 | 0.4965 | 0.5026 | +0.0062 | 7513→7872 | 19751→19350 |
| low_light_0.25 | 0.4591 | 0.4848 | +0.0257 | 6233→7155 | 21338→20336 |

Median ở Salt-and-Pepper 0.06 tăng mAP50 +0.2478 (0.2459→0.4936), FN 27.102→19.438. CLAHE fog 0.50 tăng +0.0286; CLAHE low-light 0.25 tăng +0.0257. Tập hợp raw/recovered: `runs/full_val2017_comparison_seed42/raw_vs_recovered.csv`; biểu đồ `map_raw_vs_recovered.png`.

Lệnh recovery: `python robustness_eval.py --data datasets/coco/full_val2017/data.yaml --model yolov8n.pt --out runs/full_val2017_recovered_seed42 --device 0 --imgsz 640 --batch 16 --seed 42 --families salt_pepper fog low_light --recovered-only`.

## Gaussian-NLM full val2017 — 2026-11-08

Xác nhận Gaussian noise level 0.06 sau Fast Non-Local Means trên đủ 5.000 ảnh (seed 42). NLM tạo ảnh CPU-intensive nên chỉ chạy mức nặng nhất trên full split; level 0.01/0.03 recovery đã có trên subset 500 ảnh.

| Điều kiện | Raw mAP@0.5 | Recovered mAP@0.5 | Δ mAP@0.5 | Raw mAP@0.5:0.95 | Recovered mAP@0.5:0.95 | FP raw→rec | FN raw→rec |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian 0.06 | 0.4387 | 0.4460 | +0.0073 | 0.3037 | 0.3072 | 7098→7396 | 21476→21193 |

NLM chỉ phục hồi một phần hiệu năng; recovered mAP50 vẫn thấp hơn clean 0.5185. Dataset/metrics/confusion matrix ở `runs/full_val2017_gaussian_recovered_seed42/`; đã cập nhật `aggregate_full_val.py`, `raw_vs_recovered.csv` và biểu đồ comparison.

## Phân tích class-wise confusion — 2026-11-08

Gộp confusion matrices full-val cho 23 điều kiện thành TP/FP/FN, precision/recall theo class tại conf 0.25 và IoU 0.5. So với clean ở Salt-and-Pepper 0.06, FP tăng nhiều nhất ở bench (+129; support 411), umbrella (+113; support 407), tie (+111; support 252), surfboard (+81; support 267). FN person tăng 2.178 trên support 10.777, trong khi FP person giảm 633 — detector bỏ sót người nhiều hơn, không phải sinh thêm false person detections.

Artefak: `runs/full_val2017_classwise/classwise_confusion_metrics.csv`, `classwise_deltas_vs_clean.csv`, `salt_pepper_0.06_class_errors.png`; script tái lập `analyze_class_errors.py`.

## So sánh detector SSDLite trên COCO val2017 đầy đủ — 2026-11-08

Đánh giá torchvision SSDLite320-MobileNetV3-Large COCO_V1 trên cùng 5.000 ảnh val2017, cùng nhãn và 9 điều kiện: clean, Gaussian 0.06 raw/NLM, Salt-and-Pepper 0.06 raw/median, fog 0.50 raw/CLAHE, low-light 0.25 raw/CLAHE. Checkpoint pretrained được lưu tại `runs/torch_hub/checkpoints/`; suy luận chạy trên RTX 3060. AP được tính bằng nội suy 101 điểm, IoU 0.50–0.95; confusion counts dùng conf 0.25, IoU 0.5.

Clean đạt mAP50 0.3386 / mAP50-95 0.2104. S&P 0.06 làm mAP50 giảm xuống 0.2254 (-33.4% tương đối); median phục hồi lên 0.3298 (+0.1044 so raw). Gaussian-NLM làm mAP giảm từ 0.3184 xuống 0.3065; CLAHE tăng fog 0.50 từ 0.3005 lên 0.3223 và low-light 0.25 từ 0.2872 lên 0.3050. Các điều kiện S&P/fog/low-light có FP giảm ở raw đi cùng FN tăng, nên cần diễn giải counts cùng mAP.

Gộp với YOLOv8n và tạo hai biểu đồ bằng `compare_detectors.py`. Bảng: `runs/detector_comparison_fullval/metrics.csv`; charts: `map50_detector_comparison.png`, `recovery_delta_detector_comparison.png`. Kết quả và caveat so sánh được bổ sung vào `RESULTS_REPORT.md`.

Lưu ý: SSDLite dùng transform pretrained 320 px, YOLO dùng imgsz 640; cách tính AP cũng qua hai implementation khác nhau (script custom 101 điểm và validator Ultralytics). Vì vậy xu hướng biến đổi trong từng detector đáng tin hơn so sánh trị tuyệt đối giữa detector. Đây vẫn là benchmark trên nhiễu tổng hợp, không phải dữ liệu thời tiết thực.

Tiếp tục phân tích ma trận nhầm lẫn SSDLite thành metrics theo lớp cho đủ 9 điều kiện. Ở Salt-and-Pepper 0.06 raw so với clean, FP tăng nhiều nhất ở bench (+167), teddy bear (+134), person (+126), cake (+99), train (+86); FN person tăng 546 trên support 10.777. Artefak ở `runs/ssdlite_fullval_seed42/classwise/`; biểu đồ `salt_pepper_0.06_class_errors.png`; script `analyze_ssdlite_class_errors.py`.

## Chuẩn hóa phép so sánh YOLOv8n và SSDLite — 2026-11-08

Chạy lại YOLOv8n trên đủ 5.000 ảnh ở imgsz 320 cho cùng 9 điều kiện mà SSDLite đã đánh giá. Tính AP bằng cùng hàm nội suy 101 điểm, IoU 0.50–0.95; đếm confusion ở conf 0.25, IoU 0.5. Clean YOLO đạt mAP50 0.4037, mAP50-95 0.2784. Salt-and-Pepper 0.06 raw/recovered lần lượt đạt 0.2612/0.3951; fog 0.50 0.3576/0.3859; low-light 0.25 0.3482/0.3728; Gaussian 0.06 0.3766/0.3813.

Bảng ghép cùng evaluator: `runs/normalized_detector_comparison_fullval/metrics.csv`; charts `map50_normalized_detector_comparison.png`, `recovery_delta_normalized_detector_comparison.png`. Script chạy YOLO `yolo_custom_eval.py`; script ghép `compare_detectors_normalized.py`. Chạy trên GPU RTX 3060. Dù cùng AP evaluator và input mục tiêu 320, model vẫn giữ post-processing/NMS và quy trình resize nội bộ riêng, nên không xem đây là benchmark tuyệt đối đồng nhất.

Mở rộng để so sánh đủ đường cong raw: chạy bổ sung 8 mức nhiễu còn thiếu ở cả hai detector, đủ 12 mức (3 mức × 4 họ) trên 5.000 ảnh. SSDLite hoàn tất 17 điều kiện tổng cộng (clean + 12 raw + 4 recovered mức mạnh); YOLO custom AP cũng có 17 điều kiện. Kết quả raw được ghép thành 24 dòng ở `runs/normalized_detector_comparison_fullval/raw_intensity_metrics.csv`; biểu đồ 2×2 tại `raw_mAP_vs_intensity_both_detectors.png`. Tái tạo biểu đồ bằng `plot_detector_intensity_curves.py`.

Xu hướng mAP50 raw tăng theo mức Gaussian/Salt-and-Pepper/fog giảm cường độ; S&P gây suy giảm mạnh nhất ở 0.06 (YOLO 0.2612, SSDLite 0.2254). Ảnh low-light có hệ số sáng thấp hơn tối hơn; mAP giảm từ factor 0.60 xuống 0.25 (YOLO 0.3984→0.3482, SSDLite 0.3262→0.2872). Hai detector dùng cùng evaluator AP 101 điểm và input mục tiêu 320 nhưng giữ NMS/postprocess riêng.

## Mở rộng đường cong recovery trên full val — 2026-11-08

Đánh giá thêm SSDLite trên sáu recovery còn thiếu: median S&P 0.01/0.03, CLAHE fog 0.15/0.30, CLAHE low-light 0.40/0.60. Sau đó chạy YOLOv8n-320 bằng cùng custom AP evaluator trên đúng sáu condition. Hiện cả hai detector có clean + đủ 12 raw + 10 recovered = 23 điều kiện; Gaussian-NLM chỉ recovered ở level 0.06 full-val.

Recovery delta mAP50 YOLO/SSDLite: S&P 0.01 +0.0295/+0.0356; 0.03 +0.0782/+0.0705; 0.06 +0.1339/+0.1044. Fog 0.15 -0.0052/-0.0019; 0.30 +0.0080/+0.0036; 0.50 +0.0283/+0.0218. Low-light 0.25 +0.0247/+0.0179; 0.40 +0.0045/+0.0078; 0.60 -0.0053/+0.0033. Gaussian-NLM 0.06 +0.0047/-0.0119.

Tệp ghép 20 cặp raw/recovered `runs/normalized_detector_comparison_fullval/recovery_intensity_metrics.csv`; plot `recovery_curves_both_detectors.png`; script `plot_recovery_curves.py`. SSD metrics/confusion matrices được bổ sung trong `runs/ssdlite_fullval_seed42/`, YOLO ở `runs/yolo_custom_fullval_seed42/`. Chạy lại phân tích class-wise SSDLite để bao gồm cả 23 condition.

Đã thăm dò tạo Gaussian-NLM recovered full-val ở mức 0.01/0.03 để hoàn thiện recovery curve. Tốc độ đo được khoảng 455/5.000 ảnh sau vài phút; dự báo hơn một giờ mỗi level. Dừng trước khi benchmark do chi phí thời gian cao; xóa dataset tạm chưa hoàn tất. Không có metrics nào từ lượt dở dang được dùng. Giữ Gaussian-NLM full-val ở mức 0.06 và ba mức Gaussian-NLM trên subset 500 ảnh; giới hạn này được nêu trong báo cáo.

## Gaussian bilateral recovery full-val — 2026-11-08

Để có thêm baseline khôi phục Gaussian full-val mà không chờ NLM nhiều giờ, tạo riêng ba condition `gaussian_{0.01,0.03,0.06}_bilateral` bằng `cv2.bilateralFilter(d=7, sigmaColor=50, sigmaSpace=50)`. Đánh giá trên cùng 5.000 ảnh bằng YOLOv8n-320 và SSDLite-320, cùng evaluator 101 điểm. Không ghi đè các condition NLM `_recovered`.

YOLO bilateral mAP50 các level 0.01/0.03/0.06: 0.3990/0.3990/0.3906 (raw: 0.4032/0.3959/0.3766). SSDLite bilateral: 0.3300/0.3296/0.3248 (raw: 0.3391/0.3330/0.3184). Tại 0.06, NLM đạt 0.3813 YOLO và 0.3065 SSDLite; bilateral cao hơn tương ứng 0.0093 và 0.0184.

Bảng so sánh: `runs/normalized_detector_comparison_fullval/gaussian_denoiser_comparison.csv`; chart: `gaussian_denoiser_comparison.png`; script ghép: `compare_gaussian_denoisers.py`. Dataset bilateral/Ultralytics logs/confusion: `runs/full_val2017_gaussian_bilateral_seed42/`. Thêm `--recovery-tag` vào `robustness_eval.py` để lưu condition theo tên phương pháp, và thêm bilateral roots vào evaluator scripts.

## Lặp các điều kiện ngẫu nhiên nặng với seed 43 — full val — 2026-11-08

Tạo lại Gaussian 0.06 raw, S&P 0.06 raw và S&P 0.06 median trên cùng 5.000 ảnh COCO val2017 bằng seed 43. Chấm SSDLite và YOLOv8n-320 với cùng AP evaluator. So sánh seed 42/43 (mean ± sample SD, n=2) mAP50: YOLO Gaussian 0.37706 ± 0.00068, S&P raw 0.26121 ± 0.00003, S&P median 0.39516 ± 0.00006; SSD Gaussian 0.31886 ± 0.00064, S&P raw 0.22415 ± 0.00178, S&P median 0.33035 ± 0.00077.

Seed43 dataset và native YOLO metrics/confusion ở `runs/full_val2017_seed43_selected/`; custom shared-evaluator metrics có hậu tố `_seed43` trong `runs/yolo_custom_fullval_seed42/metrics.csv` và `runs/ssdlite_fullval_seed42/metrics.csv`. Bảng gộp: `runs/normalized_detector_comparison_fullval/seed_repeat_summary.csv`; plot: `seed_repeat_map50.png`; script: `aggregate_fullval_seeds.py`. Độ lệch nhỏ chỉ phản ánh 2 mẫu noise trên cùng split; chưa thay thế lặp nhiều subset/dataset.

## Mở rộng lên ba seed full-val — seed 44 — 2026-11-08

Lặp lại Gaussian 0.06 raw, S&P 0.06 raw và median với seed 44, tạo dataset/validation ở `runs/full_val2017_seed44_selected/`. Cả hai model được chấm bằng shared evaluator 320. Cập nhật summary lên n=3: YOLO mAP50 mean ± SD — Gaussian 0.37694 ± 0.00052, S&P raw 0.26222 ± 0.00175, S&P median 0.39495 ± 0.00037; SSDLite — Gaussian 0.31885 ± 0.00045, S&P raw 0.22351 ± 0.00169, S&P median 0.33012 ± 0.00067.

Đã cập nhật `aggregate_fullval_seeds.py`, `seed_repeat_summary.csv`, `seed_repeat_map50.png`, `RESULTS_REPORT.md` và README. Ba seed giảm nguy cơ kết luận theo mẫu noise đơn lẻ nhưng vẫn là cùng một validation split và chưa đo biến thiên theo tập ảnh.

## Module denoising độc lập

Tách các phép khôi phục thành `preprocessing.py` (`recover_image`) và dùng lại từ `robustness_eval.py`. Thêm `denoise.py` để áp dụng method auto/median/NLM/CLAHE/bilateral cho file đơn hoặc thư mục, giữ relative paths, không ghi đè nếu thiếu `--overwrite`. README và report có ví dụ lệnh. Không thay đổi tham số xử lý đã dùng trong thí nghiệm.

## Lặp đầy đủ subset với seed 44 — 2026-11-08

Đã chạy đủ 25 điều kiện trên cùng COCO val2017 subset 500 ảnh, YOLOv8n pretrained, RTX 3060, imgsz 640, batch 16. Seed 44 hoàn tất trên CUDA; `runs/repeat_seed44/metrics.csv` có 25 dòng metrics, cùng các confusion matrices và dữ liệu biến đổi. Gộp cùng seed 42 và 43 tạo 75 dòng theo seed, 25 dòng tổng hợp mean ± sample SD trong `runs/repeat_analysis_seed42_43_44/`.

Với n=3, Salt-and-Pepper 0.06 raw đạt mAP@0.5 **0.2692 ± 0.0080**, median 3×3 phục hồi lên **0.5169 ± 0.0034**. Gaussian 0.03 raw đạt **0.5213 ± 0.0059**, NLM giảm còn **0.5084 ± 0.0015**; Gaussian 0.06 raw **0.4607 ± 0.0023**, NLM **0.4672 ± 0.0048**. Fog và low-light lặp lại cùng giá trị vì phép biến đổi tất định; SD=0 không phản ánh bất định do chọn ảnh. Ba lần chạy chỉ lấy mẫu noise trên cùng subset, không phải ba benchmark độc lập.

- Seed44 metrics và confusion matrices: `runs/repeat_seed44/`
- Bảng theo seed và summary: `runs/repeat_analysis_seed42_43_44/metrics_by_seed.csv`, `metrics_summary.csv`
- Biểu đồ mAP và FP mean ± SD: `runs/repeat_analysis_seed42_43_44/map_vs_intensity_mean_std.png`, `false_positives_mean_std.png`
- Script tái tạo tổng hợp: `aggregate_repeats.py` (seed 42, 43, 44)

## Phân tích lỗi theo lớp cho hai detector — 2026-11-08

Tạo `analyze_normalized_class_errors.py` để phân tích confusion matrices của YOLOv8n-320 và SSDLite-320 với cùng evaluator, xuất class-wise counts/deltas và biểu đồ raw-vs-median cho S&P 0.06. S&P 0.06 raw làm FN person tăng +938 ở YOLO và +546 ở SSDLite trên support 10.777; chair FN tăng lần lượt +220 và +190 trên support 1.771. Sau median, person FN delta giảm còn +91/+73.

Trong lúc rà soát phát hiện cả ba script class-wise trước đó tính support bằng cách trừ cột FN khỏi tổng hàng confusion matrix, khiến support thấp hơn thực tế và recall per class bị phóng đại. Đã sửa `analyze_class_errors.py`, `analyze_ssdlite_class_errors.py` và script mới: support giờ là tổng toàn bộ hàng GT, gồm cả cột background/FN. Đã chạy lại các phân tích; FP/FN deltas và đồ thị không đổi, còn support/recall được cập nhật đúng.

Thêm thay đổi precision/recall theo lớp dạng percentage points và biểu đồ recall cho S&P 0.06. Recall giảm mạnh nhất ở YOLO: refrigerator −42.1 pp, mouse −39.6 pp, microwave −38.2 pp; SSDLite: cat −37.1 pp, airplane −35.0 pp, refrigerator −34.9 pp. Các lớp có support thấp nên đây là thống kê mô tả; không diễn giải như kiểm định suy luận.

## Lặp điều kiện nặng trên ba subset COCO rời nhau — 2026-11-08

Tạo thêm hai subset 500 ảnh từ phần COCO val2017 còn lại, loại trừ manifest của subset gốc và subset trước đó. Ba nhóm có 500 ảnh mỗi nhóm, không có giao nhau; số instance lần lượt 3.799, 3.828 và 3.521. Trên hai subset mới, chạy clean cùng Gaussian 0.06 raw/NLM, S&P 0.06 raw/median, fog 0.50 raw/CLAHE và low-light 0.25 raw/CLAHE. Cố định YOLOv8n, imgsz 640, batch 16 và noise seed 42.

Qua ba subset, clean đạt mAP50 0.5413 ± 0.0256. S&P 0.06 raw đạt 0.2674 ± 0.0120; median recovery đạt 0.5185 ± 0.0283. Thay đổi tương đối so clean trong từng subset: S&P raw −50.6 ± 1.2%, median −4.2 ± 0.7%; Gaussian raw −14.7 ± 0.9%, NLM −13.0 ± 1.5%; fog raw/CLAHE −7.7 ± 1.8% / −3.0 ± 1.2%; low-light raw/CLAHE −11.1 ± 1.1% / −5.9 ± 0.3%. SD là giữa ba subset COCO rời nhau, chưa phải độ bất định trên benchmark độc lập.

- Metrics từng subset: `runs/image_subset_set2/metrics.csv`, `runs/image_subset_set3/metrics.csv`; subset gốc: `runs/noise_sweep_seed42/metrics.csv`.
- Tổng hợp 27 dòng, mean ± SD và plot: `runs/image_subset_analysis/`.
- Manifests không giao nhau: `datasets/coco/subset_500/manifest.json`, `subset_500_image_set2/manifest.json`, `subset_500_image_set3/manifest.json`.
- Tái tạo bảng/plot: `aggregate_image_subsets.py`; hỗ trợ loại trừ manifest khi lấy mẫu: `prepare_coco.py --exclude-manifest`.

## Kiểm tra định tính denoising trên COCO — 2026-11-08

Thêm `create_qualitative_figure.py` để chọn một ảnh COCO val2017 có khoảng năm bbox rồi tạo panel 4×3 clean/raw/recovered cho Gaussian 0.06 (bilateral), Salt-and-Pepper 0.06 (median), fog 0.50 (CLAHE) và low-light 0.25 (CLAHE). Ground-truth boxes được phủ giống nhau trên mọi ảnh. Đã tạo `runs/qualitative_coco_examples/qualitative_recovery_examples.png` và metadata JSON, sau đó xem trực quan. Salt-and-Pepper median loại phần lớn pixel đơn lẻ; bilateral làm ảnh mượt hơn; CLAHE tăng tương phản cục bộ ở fog nhưng không khôi phục độ sáng tổng thể trong ví dụ low-light. Đây là quan sát của một ảnh, không thay số liệu toàn tập.

Đã thử bổ sung Gaussian-NLM 0.01 full-val, nhưng sau vài phút chỉ sinh được khoảng 400/5.000 ảnh. Dừng trước bước inference do thời lượng ước tính quá dài cho mỗi mức; xóa condition sinh dở, không ghi metrics mới. Full-val NLM hiện vẫn chỉ có mức 0.06; đủ ba mức NLM vẫn có trên subset 500 ảnh.

## Khóa môi trường tái lập và hoàn thiện tham khảo — 2026-11-08

Ghi lại Python 3.10.3, Ultralytics 8.4.204, PyTorch 2.10.0+cu128, Torchvision 0.25.0+cu128, OpenCV 4.13.0, NumPy 2.2.6, PyYAML 6.0.3, Matplotlib 3.10.8 và RTX 3060. Tạo `requirements-experiment.lock.txt` cho các package trực tiếp; thêm SHA-256 cho `yolov8n.pt` và Torchvision SSDLite COCO_V1 trong report. `pip check` báo không có dependency conflicts. Thêm nguồn COCO, Ultralytics validation, Torchvision SSDLite, OpenCV denoising và COCOeval vào phần tài liệu tham khảo.

## Phân tích recall theo kích thước vật thể — COCO subset — 2026-11-08

Thêm `analyze_size_robustness.py` và chạy YOLOv8n-640 trên 500 ảnh, clean và raw/recovered ở Gaussian 0.06, S&P 0.06, fog 0.50, low-light 0.25. Bins theo diện tích bbox pixel: small <32², medium 32²–96², large ≥96²; confidence 0.25, IoU 0.5. Ghép box class-aware một-một toàn ảnh, rồi nhóm TP theo size. Recall small/medium/large: clean 0.2239/0.5354/0.7325; Gaussian raw/NLM 0.1634/0.1677, 0.4315/0.4423, 0.6338/0.6545; S&P raw/median 0.1132/0.1677, 0.2734/0.4820, 0.3846/0.7309; fog raw/CLAHE 0.1746/0.2014, 0.4632/0.4964, 0.6624/0.7062; low-light raw/CLAHE 0.1461/0.1711, 0.4156/0.4596, 0.6497/0.6744. Lưu bảng/heatmap ở `runs/size_stratified_analysis/`; số liệu là một subset/seed, không phải mAP area ranges.

Lặp size-analysis cho clean/Gaussian 0.06 raw/NLM/S&P 0.06 raw/median qua seed42/43/44 trên cùng subset. Recall mean ± SD small/medium/large: Gaussian raw 0.1625±0.0023/0.4264±0.0059/0.6356±0.0047; NLM 0.1677±0.0052/0.4396±0.0052/0.6526±0.0032; S&P raw 0.1066±0.0061/0.2732±0.0004/0.3854±0.0014; median 0.1726±0.0043/0.4803±0.0015/0.7320±0.0009. Mean/SD heatmap và bảng ở `runs/size_stratified_seed_analysis/`; SD chỉ phản ánh ba noise draws trên cùng ảnh.

## Recall theo kích thước qua ba subset COCO — 2026-11-08

Thêm `analyze_size_across_subsets.py`; chạy YOLOv8n-640, conf 0.25, IoU match 0.5 trên 9 điều kiện (clean và raw/recovered Gaussian, S&P, fog, low-light cường độ cao) ở ba subset COCO val2017 không giao nhau, mỗi subset 500 ảnh. Seed biến dạng cố định 42. CSV ghi TP/FN/support từng subset; summary và heatmap báo mean ± SD giữa các subset.

Recall small/medium/large mean ± SD: clean 0.2115±0.0107/0.5204±0.0256/0.7410±0.0121; Gaussian raw 0.1481±0.0133/0.4108±0.0226/0.6422±0.0152, NLM 0.1513±0.0143/0.4192±0.0328/0.6676±0.0114; S&P raw 0.0975±0.0136/0.2666±0.0088/0.3894±0.0042, median 0.1679±0.0088/0.4814±0.0122/0.7362±0.0088; fog raw 0.1678±0.0060/0.4607±0.0191/0.6725±0.0099, CLAHE 0.1884±0.0113/0.4953±0.0232/0.7172±0.0121; low-light raw 0.1352±0.0094/0.4196±0.0126/0.6597±0.0089, CLAHE 0.1629±0.0074/0.4497±0.0153/0.6923±0.0171.

Artifacts: `runs/size_stratified_subsets/size_stratified_recall_by_subset.csv`, `size_stratified_recall_summary.csv`, and `size_stratified_recall_by_subset.png`. SD ở đây phản ánh khác biệt giữa các subset trong một split COCO, không phải giữa các dataset độc lập.

## Phân tích lỗi theo lớp qua ba subset COCO — 2026-11-08

Thêm `analyze_class_errors_subsets.py`, tổng hợp confusion matrices clean/S&P 0.06 raw/median từ ba subset 500 ảnh không giao nhau. Hạng theo mean thay đổi recall per-subset, chỉ xếp hạng lớp có ít nhất 30 GT cộng gộp. Raw S&P làm recall cat/mouse/airplane/refrigerator giảm trung bình −73.5/−73.1/−70.7/−67.1 pp; median giảm mức này còn −12.5/−5.4/−2.7/−3.3 pp. Support còn thấp so với lớp phổ biến, nên kết quả mô tả, không phải kiểm định per-class.

Chi tiết theo subset, summary và biểu đồ mean ± SD: `runs/classwise_subset_analysis/`. SD là biến thiên giữa ba subset trong cùng COCO val2017.

## Phân tích per-class cả bốn họ biến dạng — 2026-11-08

Thêm `analyze_all_class_errors_subsets.py`, tổng hợp confusion matrices cho clean và 8 nhánh raw/recovered mức nặng từ ba subset. Trong các lớp có ≥30 GT cộng gộp, các lớp giảm recall raw nhiều nhất: Gaussian — mouse −31.7±7.5 pp; S&P — cat −73.5±13.8 pp; fog — oven −27.0±13.8 pp; low-light — frisbee −25.9±10.2 pp. Recovery nâng lần lượt lên −20.2±10.0, −12.5±0.9, −15.0±13.2, −3.7±9.5 pp so clean. Do support theo lớp thấp và SD subset tương đối lớn, các con số là mô tả.

CSV per-subset, summary và 2×2 plot nằm trong `runs/classwise_all_conditions_subsets/`.

## Độ nhạy FP/FN theo ngưỡng confidence — 2026-11-08

Thêm `analyze_confidence_threshold_sensitivity.py`; infer một lần tại conf 0.001, sau đó đếm TP/FP/FN tại conf 0.05/0.25/0.50, IoU ghép 0.5. Chạy clean và 8 nhánh raw/recovered cường độ cao trên ba subset COCO, YOLOv8n-640. Ở S&P raw, FP/100 ảnh tại ba ngưỡng là 772.9/101.5/23.4 so với clean 1200.4/180.1/41.9; FN là 466.7/556.6/620.5 so với clean 260.3/378.9/478.4. Median đưa FP về 1096.3/166.3/39.3 và FN về 281.8/399.9/494.7. Nhiễu không làm FP tăng trong điều kiện này; FP giảm cùng lúc FN tăng do recall giảm. Kết luận được ghi rõ trong report.

CSV theo subset, summary mean ± SD và biểu đồ: `runs/confidence_threshold_analysis/`. SD là biến thiên giữa ba subset COCO val2017.

## Policy denoising chọn lọc theo mức đã benchmark — 2026-11-08

Mở rộng `denoise.py --method auto` với `--level` tùy chọn. Nếu truyền level, CLI dùng policy validation-tuned từ full-val YOLOv8n/SSDLite: Gaussian 0.01/0.03→no-op, 0.06→bilateral; S&P cả ba mức→median; fog 0.15→no-op, 0.30/0.50→CLAHE; low-light 0.25/0.40→CLAHE, 0.60→no-op. Chỉ chấp nhận các mức đã benchmark. Không truyền `--level` thì giữ mapping family-auto cũ; truyền method tường minh vẫn chọn đúng method thủ công. Đây là routing dựa trên loại/mức đã biết, không phải phát hiện nhiễu từ ảnh; policy cần kiểm chứng trên test độc lập.

## Snapshot đầy đủ môi trường Python — 2026-11-08

Tạo `environment-snapshot-windows-cu128.txt` từ `pip freeze` trong `.venv` (220 package), bổ sung transitive dependencies và package khác có sẵn trong môi trường bên cạnh `requirements-experiment.lock.txt` gọn cho dependencies trực tiếp. README và report phân biệt hai file; snapshot chỉ đại diện cấu hình Windows/Python 3.10.3/CUDA 12.8 đã chạy.

## Desktop MVP cho demo computer vision — 2026-11-08

Thêm `app.py` với giao diện Tkinter tiếng Việt: mở ảnh, chọn chế độ mô phỏng nhiễu trên ảnh sạch hoặc xử lý ảnh đầu vào đã nhiễu, chọn family/level, chọn recovery theo policy đã benchmark hoặc chọn filter thủ công, chạy detector, xem baseline, ảnh nhiễu trước recovery có detections, và ảnh sau recovery có detections. Lưu ảnh đầu vào/detection/processed/noisy cùng JSON metadata. GPU CUDA được tự chọn khi khả dụng, nếu không dùng CPU. Thêm `run_app.bat` để khởi chạy bằng một lần bấm. Đây là MVP desktop cục bộ; chưa có installer, hỗ trợ webcam/batch UI hay nhận diện loại nhiễu tự động. GUI chưa được mở smoke-test.

## So sánh checkpoint YOLOv8n với YOLOv8s — 2026-11-08

Tải checkpoint chính thức Ultralytics `yolov8s.pt` (22,573,363 bytes; SHA-256 `268E5BB54C640C96C3510224833BC2EEACAB4135C6DEB41502156E39986B562D`) sau khi GitHub asset endpoint timeout ở lần đầu; tải thành công từ kho Ultralytics trên Hugging Face. Không tải dataset mới. Đánh giá clean trên COCO val2017, 5.000 ảnh/36.335 instances, imgsz 640, RTX 3060, Ultralytics 8.4.204. YOLOv8s đạt mAP50 0.60965 và mAP50-95 0.44227 so với YOLOv8n 0.51846/0.36713; chênh +0.09119/+0.07514. Confusion threshold 0.25: TP +3.538, FN −3.538, FP +1.980.

Đo latency single-image ở imgsz640/conf0.25/NMS IoU0.7, 200 COCO images, hai lượt đảo model order, warm-up20/lượt. Mean latency n/s 9.76/10.33 ms; median 8.36/8.76; p95 12.73/15.26; peak CUDA allocation 0.032/0.077 GB. Đây là phép đo RTX3060 cục bộ. `yolov8s.pt` được đặt làm checkpoint mặc định desktop app do mAP cao hơn và latency trung bình chỉ tăng 0.57 ms; fine-tune chưa quyết định cho tới khi có dữ liệu mục tiêu có nhãn.

Artifacts: `runs/yolov8s_fullval_clean/metrics.csv`, `runs/yolo_model_size_comparison/yolov8n_vs_yolov8s_fullval.csv`, `single_image_latency.csv`, and `yolov8n_vs_yolov8s_fullval.png`. Scripts: `compare_yolo_model_sizes.py`, `benchmark_yolo_inference_latency.py`.

Thêm `classwise_recovery_effect_summary.csv`, tính recovered − raw pp cho từng lớp. Trường hợp suy giảm lớn nhất theo nhóm đủ support: Gaussian teddy bear −10.0 pp, fog carrot −7.1 pp, low-light cow −3.9 pp. Với S&P, median tăng recall mean trên tất cả lớp có ít nhất 30 GT cộng gộp; mức tăng nhỏ nhất là handbag +6.3 pp. Các kết quả class-wise này mang tính mô tả, không tối ưu preprocessing theo từng lớp.

## YOLOv8s robustness và recovery trên full COCO val2017 — 2026-11-08

Chạy YOLOv8s imgsz640 trên chín điều kiện đã có: clean, Gaussian 0.06 raw/bilateral, Salt-and-Pepper 0.06 raw/median, fog 0.50 raw/CLAHE, low-light 0.25 raw/CLAHE. Mỗi condition dùng đủ 5.000 ảnh COCO val2017, GPU RTX 3060, Ultralytics 8.4.204; không tải hoặc mở rộng dataset. mAP50/mAP50-95: clean 0.6112/0.4443; Gaussian raw 0.5390/0.3832, bilateral 0.5724/0.4087; S&P raw 0.2840/0.1923, median 0.5803/0.4190; fog raw 0.5592/0.4041, CLAHE 0.5948/0.4307; low-light raw 0.5556/0.3987, CLAHE 0.5799/0.4195.

Ghép với full-val YOLOv8n cho thấy YOLOv8s có mAP tuyệt đối cao hơn ở tất cả điều kiện nặng đối chiếu, nhưng raw S&P vẫn giảm 32.72 điểm phần trăm so với clean YOLOv8s. Median recovery tăng 29.63 pp, còn thấp hơn clean 3.10 pp. Đây là số liệu tổng hợp, không chứng minh kết quả trên ảnh camera mục tiêu. Tạo `compare_yolov8_robustness.py` và `runs/yolov8_model_robustness_comparison/` (CSV, biểu đồ). GUI chưa được smoke-test; fine-tune vẫn cần dữ liệu mục tiêu có bbox labels.

## Cải thiện demo và inference smoke check — 2026-11-08

Cập nhật desktop app để chạy detection riêng trên ảnh nhiễu trước recovery, hiển thị bounding boxes ở cả khung ảnh nhiễu và khung sau recovery, báo số lượng/class trước-sau và lưu counts vào JSON. Chạy smoke inference trên COCO `000000000139.jpg`, YOLOv8s CUDA, Gaussian 0.06 và bilateral: model 80 lớp, counts baseline/noisy/recovered 16/13/15. `py_compile` thành công cho `app.py` và `compare_yolov8_robustness.py`. Đây là kiểm tra pipeline model và cú pháp; cửa sổ Tkinter vẫn chưa được mở thử tương tác.

Tạo `generate_demo_sample.py` để tái tạo bộ ảnh minh họa trong `runs/demo_sample/`: baseline, noisy, recovered, ảnh raw/processed và `metadata.json`. Chạy trên RTX 3060 cho ra 16/13/15 detections. Ảnh này chỉ minh họa COCO pretrained pipeline, không dùng để suy ra accuracy cho người dùng.

Thêm `validate_yolo_dataset.py` làm preflight trước khi fine-tune/đánh giá dữ liệu mục tiêu. Kiểm tra cấu trúc ảnh-label, class ID, bbox normalized, background labels và split overlap; có chế độ `--require-coco80`. Chạy trên dataset COCO val hiện hữu: 5.000 ảnh, 36.335 instances, 48 ảnh nền, đủ 80 lớp, nhãn hợp lệ. `train` và `val` cùng trỏ tới val được nhận diện/báo là validation-only alias. Cập nhật `TARGET_DOMAIN_EVALUATION.md` và README để hướng dẫn chạy.

## Desktop UI smoke check — 2026-11-08

Khởi tạo Tkinter app ẩn để tránh làm gián đoạn desktop, sau đó chạy đúng Tk `mainloop()` với một ảnh COCO trên YOLOv8s CUDA, Gaussian 0.06, bilateral. UI pipeline hoàn tất và cập nhật trạng thái `Hoàn tất • GPU CUDA`; detections noisy/recovered là 13/15. Mock hộp thoại lưu để xác nhận tạo đủ baseline, noisy, processed, ảnh detection cuối và JSON metadata trong `runs/app_smoke_output/`. Đây là kiểm tra UI lifecycle/save path tự động; không thay thế việc người dùng chạy tương tác trên máy và ảnh mục tiêu.

## Draft auto-labels cho ảnh chưa gán nhãn — 2026-11-08

Thêm `draft_auto_labels.py`: tạo YOLO pseudo-labels, ảnh preview có boxes và metadata confidence trong output riêng, hỗ trợ `--limit` để chạy pilot. Chạy smoke trên một ảnh COCO val hiện có, YOLOv8s CUDA; tạo `runs/auto_label_smoke/` thành công. Nhãn dự thảo được cảnh báo rõ là cần con người kiểm tra/sửa và không được dùng như ground truth độc lập để tính mAP. Bổ sung workflow này vào README và `TARGET_DOMAIN_EVALUATION.md`.

## Pilot trên ảnh mục tiêu đầu tiên — 2026-11-08

Thấy 1 ảnh người dùng trong `target_eval/images/val/images.jpg`. Chạy YOLOv8s pretrained ở conf 0.25; không có detection trong 80 lớp COCO. Ở conf 0.001 model sinh nhiều candidate (262), nên ngưỡng thấp không tạo nhãn đáng tin cậy. COCO không có class `lamp`; nếu đây là lớp cần nhận diện, cần định nghĩa custom class và người dùng phải duyệt/gán box. Tạo draft output/preview tại `runs/target_auto_label_drafts/`; nhãn nháp trống do không có detections tại conf mặc định. Validator hiện báo thiếu label; chưa thể tính mAP từ ảnh này.

Người dùng thay ảnh tại cùng đường dẫn bằng một ảnh khác chụp người nằm trên giường cùng đồ chơi. Chạy lại YOLOv8s conf0.25: 1 person (0.77), 1 bed (0.53), 4 teddy bear (0.30–0.57) và 3 remote (0.49–0.66). Các dự đoán `remote` trên đồ vật nhỏ cần kiểm tra thủ công; chưa có nhãn ground truth nên chưa thể tính mAP. Cập nhật pseudo-labels và preview trong `runs/target_auto_label_drafts/`.

Người dùng xác nhận detection `bed` thực chất là `couch`. Tạo `reviewed_labels/images.txt` với class ID đổi 59→57 và giữ box dự đoán; `reviewed_metadata.json` ghi đây là sửa nhãn do người dùng xác nhận. Các boxes/classes còn lại vẫn là nháp, không đưa vào ground truth đánh giá.

Người dùng thêm `images (1).jpg` vào `target_eval/images/val`. Chạy YOLOv8s conf0.25: 2 couch (0.82, 0.52), 3 potted plant (0.67, 0.54, 0.39), 1 cell phone (0.67). Preview/labels nháp nằm trong `runs/target_auto_label_drafts_round2/`. Đây là kết quả nhận diện định tính; chưa có ground truth cho ảnh mới.

Thêm `test_target_image_robustness.py`, chạy 3 ảnh người dùng × 9 điều kiện (clean, Gaussian/S&P/fog/low-light mức nặng raw và recovery) trên YOLOv8s CUDA, tổng cộng 27 lượt inference. Lưu số detection/class/confidence vào `runs/target_image_robustness/metrics.csv` và ảnh preview từng điều kiện. Kết quả cho thấy S&P 0.06 làm dự đoán dao động mạnh; median phục hồi số detection trên hai ảnh nhưng không khẳng định box đúng. Không tính mAP vì chưa có ground truth. Thêm ảnh thứ ba là collage hai phòng ngủ; clean model dự đoán 2 bed, 1 chair, 1 clock, 2 cup.

So sánh inference imgsz640 và 960 trên 3 ảnh mục tiêu, conf0.25: tổng detection theo từng ảnh là 6→10, 6→8, 9→11; số box có diện tích <32² px là 1→1, 1→2, 2→2; mean confidence giảm tương ứng 0.670→0.577, 0.601→0.526, 0.510→0.403. Độ phân giải cao hơn thêm nhiều dự đoán nhưng cũng hạ confidence trung bình; do chưa có ground truth, không thể xác nhận đó là tăng recall hay false positives.

Thêm selector input size 640/960/1280 vào desktop UI, ghi imgsz vào payload/summary/metadata. Smoke test Tk mainloop trên ảnh `images (1).jpg` với imgsz960 hoàn tất ở GPU; baseline/processed đều 8 detections. Mặc định vẫn là 640. README nêu tradeoff tốc độ/VRAM và giới hạn nếu thiếu ground truth.

Thêm confidence selector 0.10/0.25/0.40 vào UI, lưu ngưỡng thực tế vào JSON metadata. So sánh confidence trên ba ảnh mục tiêu: ngưỡng 0.10 cho ra lần lượt 13/12/22 detections, 0.25 cho 6/6/9, 0.40 cho 5/5/7. Mức 0.10 tăng box nhỏ nhưng thêm candidate classes như book/bowl/cup/vase; không thể phân biệt TP/FP nếu chưa có labels. Smoke test UI ở conf0.10 hoàn tất trên GPU, 12 detections ở ảnh `images (1).jpg`. Kết quả chi tiết tại `runs/target_confidence_thresholds/metrics.csv`; đây chỉ là phân tích qualitative.

Thêm `annotate_target_images.py`, giao diện Tkinter để xem ảnh/nhãn dự thảo, chọn box, đổi class, xóa box, vẽ box mới và lưu YOLO labels. Bed (ID59) và couch (ID57) là hai class riêng. Smoke test trên 3 ảnh hiện có: đổi dự thảo bed→couch và lưu sang thư mục output thử nghiệm thành công; không ghi nhãn test vào target evaluation. Hướng dẫn chạy được thêm vào README và `TARGET_DOMAIN_EVALUATION.md`.

Người dùng cho biết đã hoàn tất nhãn. Validator đạt: 3 ảnh, 21 boxes, 10 lớp. Phát hiện label `images.jpg` vẫn ghi bed dù người dùng xác nhận đó là couch; cập nhật sang class 57 và lưu audit `target_eval/labels/val/review_corrections.json`. Chạy pilot comparison trên cùng 3 ảnh: YOLOv8n mAP50 0.7656 / mAP50-95 0.6085; YOLOv8s 0.9200 / 0.8904; SSDLite320-MobileNetV3-Large 0.4452 / 0.2903. Confusion @IoU0.5/conf0.25 (TP/FP/FN): n 12/6/9, s 20/3/1, SSD 9/8/12. Lưu `runs/target_model_comparison/summary.csv`. Hạn chế nghiêm trọng: các file target labels vẫn hầu hết giống YOLOv8s pseudo-label draft; chỉ box bed→couch được xác nhận rõ ràng. Vì vậy các mAP này đo agreement với pseudo labels, không phải accuracy độc lập; không dùng để chọn model hay tuyên bố hiệu năng đến khi xác minh đầy đủ ground truth.


## So sánh cùng input size 320 trên ảnh mục tiêu — 2026-11-08

Sau khi người dùng báo đã hoàn tất nhãn, chạy lại YOLOv8n và YOLOv8s ở imgsz=320, cùng split/labels với SSDLite320 để tránh so trực tiếp YOLO 640 với SSD 320. Lưu tổng hợp `runs/target_model_comparison/summary_imgsz320.csv`. Kết quả mAP50 / mAP50-95: YOLOv8n 0.4476 / 0.2600; YOLOv8s 0.5521 / 0.4324; SSD-MobileNet 0.4452 / 0.2903. Tập chỉ có 3 ảnh, 21 box; nhãn mục tiêu trùng phần lớn với dự đoán nháp YOLOv8s nên đây chỉ là kiểm tra pilot/độ tương hợp nhãn, chưa phải phép đo accuracy độc lập.

## External validation subset — Open Images V7 — 2026-11-08

User requested a small external dataset subset limited to project-overlapping classes. Downloaded 100 Open Images validation JPEGs and retained 193 original boxes across 8 mappings: Person→COCO person, Chair→chair, Couch→couch, Bed→bed, Remote control→remote, Mobile phone→cell phone, Clock→clock, Teddy bear→teddy bear. Omitted Cup (Open Images only has Coffee cup in the boxable class list) and potted plant (available category is Houseplant) to avoid silently changing label definitions. Kept native Open Images annotations and per-image licensing/source metadata; no mAP run yet because annotation conventions and class filtering need to be integrated into the evaluator. Details and reproducible commands: `datasets/open_images_v7/README.md`; sample manifest: `datasets/open_images_v7/subset_manifest.csv`.

Người dùng yêu cầu xóa `target_eval/` sau khi tập con external đã được tải. Đã xóa đúng thư mục đó; 100 ảnh Open Images V7, nhãn nguồn và manifest trong `datasets/open_images_v7/` được giữ nguyên. Các kết quả pilot trước đó vẫn lưu trong `runs/target_model_comparison/` nhưng không còn ảnh/nhãn nguồn `target_eval`.

## So sánh trên Open Images V7 subset — 2026-11-08

Thêm `evaluate_open_images_subset.py` và chạy YOLOv8n, YOLOv8s, SSDLite320-MobileNetV3-Large cùng GPU RTX 3060, imgsz 320 trên 100 ảnh/193 box/8 lớp. Evaluator sử dụng nhãn positive/negative cấp ảnh của Open Images, bỏ qua dự đoán ở lớp unannotated, xử lý group-of, tính AP@0.5 và confusion counts tại conf 0.25. Kết quả: YOLOv8n mAP50 65.77%, TP/FP/FN 114/50/79; YOLOv8s 65.38%, 119/69/74; SSD-MobileNet 61.88%, 105/62/88. Lớp couch có AP thấp nhất: 27.1%, 32.7%, 37.7% tương ứng (14 GT boxes). Lưu `runs/open_images_v7_eval/summary.csv`, `per_class_ap50.csv`, biểu đồ so sánh và confusion matrices. Đây là subset pilot; evaluator chỉ xử lý 8 lớp và không mở rộng hierarchy nên không so trực tiếp các con số này với mAP COCO.

## COCO-80 fine-tuning pilot — 2026-11-08

Người dùng xác nhận model sau fine-tune phải giữ đúng 80 lớp COCO như baseline. Xóa `datasets/homeobjects-3k/` theo yêu cầu vì HomeObjects-3K chỉ gán nhãn 12 lớp. Giữ `datasets/open_images_v7/` vì đã được dùng làm phép đánh giá external 100 ảnh/8 lớp và có kết quả trong báo cáo; không dùng nó để fine-tune COCO-80.

Tạo `datasets/coco/train_subset_1000/` từ annotation COCO Train2017, giữ toàn bộ box COCO-80 trên 1.000 ảnh đã chọn; tải ảnh từ endpoint chính thức và lưu manifest/config. Thêm `prepare_coco_train_subset.py`, `finetune_yolo.py`, `finetune_ssdlite.py`. RTX 3060 CUDA.

Fine-tune YOLOv8n và YOLOv8s cùng dữ liệu, 640px, tối đa 10 epoch; SSDLite320-MobileNetV3-Large 320px, 10 epoch. Đánh giá trên cùng 5.000 ảnh COCO val2017. Baseline→fine-tune mAP50: YOLOv8n 51.92%→33.65%; YOLOv8s 60.96%→26.76% (early stop); SSD 33.86%→31.13%. SSD FP tăng 6,933→10,917. Lưu `runs/finetune_coco80/comparison.csv`, checkpoints/logs tại `runs/finetune_coco80/`. Kết luận: không dùng các checkpoint này làm model sản phẩm; tập COCO 1K và cấu hình hiện tại làm giảm hiệu năng. Cần dữ liệu/chiến lược fine-tune khác trước khi so robustness sau fine-tune.

Người dùng yêu cầu đổi cách tiếp cận khi checkpoint không ổn. Rà soát xác nhận còn 3 ảnh phòng độc nhất trong cache các lượt chạy cũ, nhưng nhãn hiện tại là pseudo-label chưa xác minh và các ảnh đã dùng làm pilot evaluation. Staging 3 ảnh cùng draft labels vào `datasets/target_domain_80/review/`, thêm README với lệnh gán nhãn. Không huấn luyện lại trên nhãn nháp. Kế hoạch mới: xác minh nhãn COCO-80, bổ sung ảnh nhà đa dạng và ảnh test held-out mới; sau đó fine-tune từ checkpoint COCO gốc với replay COCO-80 cân bằng, LR thấp/ít epoch, so sánh lại clean COCO và target test. Chi tiết: `TARGET_DOMAIN_FINETUNE_PLAN.md`.

Người dùng yêu cầu xóa 3 ảnh phòng. Đã xóa thư mục review staging, các thư mục draft-label/preview cũ, bản sao images/labels trong các run target comparison, 27 ảnh preview target robustness và các ảnh val-batch có chứa ảnh phòng; giữ lại CSV/metrics của thí nghiệm. Tập ảnh người dùng không còn được dùng. Thay hướng sản phẩm sang giữ checkpoint COCO-80 gốc và thêm tùy chọn inference tiling trong `app.py`: crop 640px, overlap 25%, class-aware NMS IoU 0.55, giới hạn 300 box; mặc định tắt. README ghi rõ đây là tùy chọn thử small-object recall, chưa có ground truth chứng minh tăng accuracy. Kế hoạch sản phẩm: `PRODUCT_INFERENCE_PLAN.md`.

Thêm `detection_inference.py` dùng chung inference full-frame/tiled cho app và evaluator; thêm `evaluate_tiled_inference.py` để so mAP, FP/FN và recall small/medium/large trên COCO val2017, cập nhật hướng dẫn README. Sau đó chạy full benchmark trên 5.000 ảnh (xem kết quả ở mục kế tiếp).

Chạy benchmark tiled trên RTX 3060 với YOLOv8s, imgsz640, toàn bộ 5.000 COCO val2017. Full-frame mAP50/mAP50-95 59.79%/43.75%; tiled 60.26%/43.23%. TP/FP/FN: full 21,317/10,576/15,018; tiled 21,188/9,350/15,147. Recall small/medium/large: full 31.69%/63.46%/81.14%; tiled 31.40%/63.14%/80.74%. Tiling giảm FP nhưng tăng FN, recall small giảm nhẹ; không chứng minh cải thiện vật thể nhỏ. Giữ app toggle off by default. Lưu `runs/tiled_inference_eval/summary.csv`; cập nhật `RESULTS_REPORT.md`.

## COCO-80 fine-tuning follow-up: 5K images and longer schedule — 2026-11-08

Expanded official COCO Train2017 subset to 5,000 images, retaining all COCO-80 annotations, and trained YOLOv8n from the original COCO weights at imgsz 640, batch 8, AdamW, lr0=0.0002. The 5-epoch pilot reached best internal COCO val mAP50 47.87% / mAP50-95 33.12%. A separate run configured for up to 50 epochs with patience=8 stopped early after epoch 11; its best was epoch 3 at mAP50 43.30% / mAP50-95 29.09%, versus baseline YOLOv8n mAP50 51.92%. Therefore 50 epochs did not improve this fine-tune and the checkpoint is not suitable to replace the baseline. Since Ultralytics scales its LR schedule to the requested total epochs, the 5-epoch and 50-epoch schedules differ; do not interpret them as an isolated epoch-count ablation. CSV/checkpoints: `runs/finetune_coco80/yolov8n_coco80_5k_lr0002/results.csv` and `runs/finetune_coco80/yolov8n_coco80_5k_lr0002_50ep/results.csv`. Added `--patience` CLI option to `finetune_yolo.py`.

## Noise-aware COCO-80 fine-tuning pilot — 2026-11-08

Created `prepare_noise_aware_train.py` to build a paired train split from the existing 5K COCO Train2017 subset: 5,000 clean copies plus 5,000 corrupted copies. Corruption code matches `robustness_eval.py`; assignments are balanced across Gaussian (sigma 0.01/0.03/0.06), Salt-and-Pepper (0.01/0.03/0.06), fog (0.15/0.30/0.50), and low-light factors (0.60/0.40/0.25). All labels are copied unchanged; the 80 class IDs are validated. No new source data was downloaded.

Fine-tuned original `yolov8n.pt`, imgsz640, batch8, AdamW lr0=0.0001, maximum 10 epochs, patience4, RTX 3060. Best clean-validation epoch was 7; train log is `runs/finetune_coco80/yolov8n_noiseaware_5k_pair_lr0001/results.csv`. Evaluated the selected checkpoint against the original YOLOv8n on the same 5,000-image COCO val2017 corruption sweep (seed42), reusing pre-generated validation data. Comparison CSV and curves: `runs/noiseaware_yolov8n_fullval_sweep_seed42/comparison.csv` and `mAP50_corruption_curves.png`.

mAP50 baseline→noise-aware, in percent: clean 51.85→47.30 (−4.55 pp); Gaussian 0.01/0.03/0.06: 51.61/49.27/43.87→47.19/45.55/42.12 (−4.42/−3.72/−1.75 pp); Salt-and-Pepper 0.01/0.03/0.06: 43.37/34.64/24.59→44.80/41.33/36.78 (+1.42/+6.69/+12.20 pp); fog 0.15/0.30/0.50: 51.08/49.85/47.12→47.02/46.52/45.46 (−4.06/−3.33/−1.66 pp); low-light factors 0.60/0.40/0.25: 51.20/49.65/45.91→47.10/46.40/44.47 (−4.10/−3.24/−1.44 pp). Mean absolute mAP50 across the 12 noisy conditions changes from 45.18% to 44.56% (−0.62 pp). Mean clean-to-noisy degradation improves from 6.66 pp to 2.74 pp (+3.93 pp in relative robustness), largely because clean mAP also falls by 4.55 pp. The clear absolute gain is Salt-and-Pepper; this checkpoint is a tradeoff, not a general replacement.

At conf0.25 / IoU0.5 under Salt-and-Pepper 0.06, baseline TP/FP/FN=9,233/5,212/27,102 versus noise-aware=13,778/9,401/22,557. The mAP/recall gain comes with many more false positives. Existing baseline+median recovery on the same condition scores 49.36% mAP50 with TP/FP/FN=16,897/8,168/19,438, outperforming noise-aware training both in AP and confusion counts. Keep the COCO-pretrained baseline plus preprocessing as the product path; do not replace it with this experimental checkpoint. A Salt-and-Pepper specialist is a possible optional experiment only after confidence-threshold review and an independent repeat.

Swept confidence thresholds 0.05/0.25/0.50 on the same 5K S&P 0.06 raw validation images for baseline and noise-aware YOLOv8n, plus baseline on the median-recovered images. At conf0.50, baseline raw gives TP/FP/FN=6,254/1,264/30,081; noise-aware raw=9,609/2,223/26,726; baseline+median=12,438/1,890/23,897. Raising threshold reduces false positives but the noise-aware model still has fewer TPs and more FPs than baseline+median. Counts for all thresholds: `runs/noiseaware_yolov8n_eval_seed42/salt_pepper_confidence_thresholds.csv`; generator: `evaluate_sp_confidence_thresholds.py`.

## COCO detector comparison at 320 px and 640 px — 2026-11-09

Measured YOLOv8s at 320 px on all 5,000 COCO val2017 images for the same nine clean/raw/recovered conditions used in the normalized comparison. The same custom 101-point AP evaluator, confidence 0.25 / matching IoU 0.50 counts, and fixed condition datasets were used. Clean mAP50/mAP50:0.95 = 49.12%/35.00%; raw Salt-and-Pepper 0.06 = 32.17%/22.03%; median-recovered = 48.12%/34.26%. Rebuilt the 320 px table and charts with YOLOv8n, YOLOv8s, and SSDLite; the table contains 27 records.

For 640 px, paired already-recorded full-validation YOLOv8n and YOLOv8s runs over the same nine conditions and matched recovery methods; no duplicate 640 inference was run. Clean mAP50 is 51.85% for YOLOv8n and 61.12% for YOLOv8s. SSD-MobileNet remains a separate native-320 model and is not presented as a 640 px comparison. The 640 px table contains 18 records. Reproduction/aggregation: `compare_detectors_normalized.py`; 320 px inference: `yolo_custom_eval.py --model yolov8s.pt --model-name YOLOv8s-320 --out runs/yolo_custom_fullval_yolov8s_img320_seed42 --conditions clean gaussian_0.06 gaussian_0.06_recovered salt_pepper_0.06 salt_pepper_0.06_recovered fog_0.5 fog_0.5_recovered low_light_0.25 low_light_0.25_recovered --imgsz 320 --batch 16 --device 0`.
