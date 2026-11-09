# Robustness of Object Detection Under Environmental Noise & Synthetic Distortions

Project này đánh giá độ bền vững của detector trước nhiễu số và điều kiện môi trường tổng hợp. Pipeline hiện đã đo YOLOv8n và SSDLite320-MobileNetV3-Large pretrained trên toàn bộ 5.000 ảnh COCO val2017, tạo raw/recovered conditions, tính mAP và confusion counts, đồng thời lưu biểu đồ và bảng kết quả trong `runs/`.

## Câu hỏi và giả thuyết

**Câu hỏi:** mAP và số false positive thay đổi thế nào khi tăng Gaussian noise, Salt-and-Pepper, fog hoặc giảm độ sáng? CLAHE, median filter và Non-Local Means có phục hồi hiệu năng không?

Giả thuyết có thể kiểm tra:

1. mAP giảm khi cường độ nhiễu tăng; mức giảm khác nhau theo loại biến dạng.
2. Gaussian noise làm giảm recall/độ chính xác định vị; Salt-and-Pepper có thể tạo false positive từ các điểm sáng/tối.
3. CLAHE có thể giúp low-light, median filter giúp Salt-and-Pepper, và Non-Local Means giúp Gaussian noise. Phục hồi cũng có thể làm mất chi tiết và giảm mAP ở mức nhẹ.

Kết quả full-val hiện có trong `RESULTS_REPORT.md`; đây là benchmark tổng hợp có thể tái lập, không thay thế đánh giá trên dữ liệu thời tiết/ban đêm thực tế.

## GitHub repository

Git tracks source code, documentation, compact experiment CSVs, and report figures. COCO/Open Images source images, generated full-resolution corruption datasets, model checkpoints (`.pt`), Python environments, and frontend dependency folders are intentionally excluded because they occupy many gigabytes or are redistributable assets. A fresh clone therefore needs the datasets and YOLO checkpoints restored locally before running inference or the image gallery; follow the setup instructions below and the dataset notes in `WEBSITE_README.md`.

## Ứng dụng desktop MVP

Ứng dụng `app.py` biến pipeline thành demo dùng được trên Windows. Có hai chế độ: mô phỏng nhiễu trên ảnh sạch, hoặc đưa vào một ảnh đã nhiễu sẵn để khử nhiễu. Người dùng chọn family/level, chạy recovery tự động theo policy benchmark hoặc chọn bộ lọc thủ công, rồi xem ba khung: baseline, ảnh nhiễu trước recovery có bounding boxes, và ảnh sau recovery có bounding boxes. Nút **Lưu kết quả** ghi ảnh baseline, ảnh phát hiện, ảnh đã xử lý, ảnh nhiễu thô và file JSON metadata, gồm số detections trước/sau recovery. Kết quả trên một ảnh là số object/confidence, không phải mAP.

Có thể chọn input size 640/960/1280 và confidence 0.10/0.25/0.40 để thử ảnh có vật thể nhỏ. Input size cao hơn tốn thời gian/VRAM; confidence thấp hiện thêm box nhưng có thể tăng dự đoán sai. Tăng số box không tự động có nghĩa accuracy tăng; muốn so mAP cần nhãn ground truth.

App còn có tùy chọn **Tiling 640px • vật thể nhỏ**: chia ảnh lớn thành các ô 640px chồng lấn 25%, chạy cùng YOLOv8s COCO-80 rồi gộp box trùng theo lớp. Pilot trên 5.000 COCO val2017 cho thấy mAP@0.5 tăng 0.47 điểm phần trăm nhưng recall small giảm từ 31.69% xuống 31.40%, mAP@0.5:0.95 giảm 0.53 điểm và false positives giảm. Chưa có bằng chứng tiling giúp vật thể nhỏ trên ảnh mục tiêu; chế độ vẫn tắt mặc định.

Để đánh giá thay vì chỉ xem demo, `evaluate_tiled_inference.py` so sánh full-frame với tiled trên COCO val2017, lưu mAP@0.5/0.5:0.95, FP/FN và recall theo small/medium/large vào `runs/tiled_inference_eval/summary.csv`:

```powershell
.\.venv\Scripts\python.exe evaluate_tiled_inference.py --device 0
```

Mở `run_app.bat` hoặc chạy:

```powershell
.\.venv\Scripts\python.exe app.py
```

Ứng dụng mặc định dùng checkpoint `yolov8s.pt` trong thư mục dự án và tự chọn GPU CUDA nếu khả dụng; nếu không có GPU thì chạy CPU. YOLOv8s được chọn sau so sánh cùng COCO val2017: mAP cao hơn YOLOv8n, độ trễ một ảnh trên RTX 3060 chỉ tăng nhẹ. Policy theo mức chỉ nhận các level đã benchmark. Khi chọn **Ảnh gốc**, app chỉ chạy detector và không thêm biến dạng/khử nhiễu.

YOLOv8s đã được huấn luyện trước trên COCO nên app nhận diện được 80 lớp COCO mà không cần train lại. Trên toàn bộ COCO val2017, YOLOv8s đạt mAP@0.5 = 0.6096 và mAP@0.5:0.95 = 0.4423 ở input 640; các số này đo cả tập dữ liệu, không phải độ chính xác cam kết cho từng ảnh. Nếu sản phẩm cần lớp ngoài COCO hoặc ảnh thuộc domain khác, cần dữ liệu gán nhãn phù hợp, fine-tune riêng và đánh giá trên tập giữ lại.

Để đo khả năng tổng quát trên môi trường sản phẩm và quyết định fine-tune, xem [hướng dẫn đánh giá trên ảnh mục tiêu](TARGET_DOMAIN_EVALUATION.md). Đã thêm tập con Open Images V7 validation gồm 100 ảnh/193 box và chạy so sánh ba model ở imgsz 320; YOLOv8n đạt mAP@0.5 65.8%, YOLOv8s 65.4%, SSD-MobileNet 61.9%. Đây là pilot 8 lớp, chưa phải kết luận về camera/domain sản phẩm. Ảnh người dùng trong `target_eval/` đã được xóa theo yêu cầu; có thể tạo lại thư mục theo hướng dẫn bên dưới khi cần.

Trước khi đánh giá/train dữ liệu YOLO mới, kiểm tra cấu trúc và nhãn bằng `python validate_yolo_dataset.py --data path/to/data.yaml`; thêm `--require-coco80` nếu dùng nguyên ID COCO. Lệnh này kiểm tra ảnh/nhãn, bbox, class IDs và split overlap.

Nếu chỉ có ảnh, `python draft_auto_labels.py --source target_eval/images/val --out runs/target_auto_label_drafts` tạo nhãn dự thảo và ảnh preview bằng YOLOv8s. Cần người kiểm tra/sửa từng nhãn trước khi dùng làm ground truth hoặc tính mAP.

Mở giao diện sửa box/class bằng `python annotate_target_images.py`. Nút **Lưu nhãn đã duyệt** ghi nhãn YOLO vào `target_eval/labels/val`; kiểm tra mọi đối tượng và phân biệt đúng `bed` với `couch` trước khi lưu.

So sánh YOLOv8n và YOLOv8s trên toàn COCO val2017 (accuracy metrics, độ trễ single-image và VRAM) nằm trong `runs/yolo_model_size_comparison/`; tạo lại phần mAP bằng `python compare_yolo_model_sizes.py` và phần tốc độ bằng `python benchmark_yolo_inference_latency.py`.

Đã đánh giá thêm YOLOv8s trên clean, nhiễu mức nặng và recovery trên đủ COCO val2017. Bảng so sánh trực tiếp với YOLOv8n và biểu đồ nằm trong `runs/yolov8_model_robustness_comparison/`; tái tạo bằng `python compare_yolov8_robustness.py`. Kết quả vẫn là benchmark nhiễu tổng hợp trên COCO, chưa đo độ chính xác cho môi trường camera mục tiêu.

Ví dụ chạy detector trên một ảnh COCO, Gaussian 0.06 và bilateral recovery được lưu tại `runs/demo_sample/` (gồm baseline/noisy/recovered detections và metadata). Tạo lại bằng `python generate_demo_sample.py`; đây là minh họa luồng xử lý, không phải phép đo độ chính xác ngoài domain.

## Giao thức đánh giá

- **Detector:** checkpoint Ultralytics YOLO do người chạy cung cấp; mặc định ví dụ dùng `yolov8n.pt`. Ghi lại chính xác checkpoint và phiên bản thư viện trong báo cáo.
- **Dataset:** validation split có nhãn YOLO chuẩn. Có thể chuyển COCO/VOC sang định dạng Ultralytics trước khi chạy. Không dùng tập train để chọn mức tiền xử lý và báo cáo kết quả trên cùng validation set như thể là test độc lập.
- **Nhiễu Gaussian:** độ lệch chuẩn `sigma = level × 255`, mức 0.01, 0.03, 0.06.
- **Salt-and-Pepper:** tỷ lệ pixel bị thay bằng đen/trắng tổng cộng bằng level, phân chia đều cho hai loại; mức 0.01, 0.03, 0.06.
- **Fog:** hòa ảnh với màu khí quyển gần trắng: `I' = (1 − alpha)I + 235 alpha`, alpha 0.15, 0.30, 0.50. Đây là fog đồng nhất, không mô hình hóa độ sâu.
- **Low-light:** nhân cường độ chuẩn hóa với 0.60, 0.40, 0.25 rồi áp dụng gamma 1.15 để làm tối ảnh.
- **Phục hồi:** auto chọn median 3×3 cho Salt-and-Pepper, fast Non-Local Means cho Gaussian, và CLAHE trên kênh độ sáng Lab cho fog/low-light. Chạy cùng model, confidence, kích thước ảnh và seed với nhánh raw.
- **Metrics:** Ultralytics validation trả `mAP@0.5` và `mAP@0.5:0.95`. Bảng confusion/FP ghép dự đoán với ground truth theo IoU ≥ 0.5, cùng lớp để tính TP; một dự đoán ghép nhầm lớp được tính một FP và một FN. Dự đoán dưới `--fp-conf` bị bỏ qua. Ma trận có hàng/cột background để biểu diễn FN/FP chưa ghép.
- **Lặp lại:** seed mặc định 42 tạo cùng một nhiễu cho raw và recovered. Nên chạy thêm nhiều seed, báo trung bình và độ lệch chuẩn, rồi đánh giá cuối trên test split giữ riêng.

## Bắt đầu dự án

Python 3.10+ được khuyến nghị. Cài dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-experiment.lock.txt
```

`requirements-experiment.lock.txt` pins the direct packages used for the reported experiments, including PyTorch/Torchvision CUDA 12.8. For the complete package snapshot from the recorded Windows environment, see `environment-snapshot-windows-cu128.txt`; that file also includes unrelated packages present in the workspace. The general `requirements.txt` remains available for a CPU or flexible setup. If installing PyTorch separately, use the [official PyTorch installation selector](https://pytorch.org/get-started/previous-versions/); for the recorded Windows configuration:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
```

Kiểm tra `torch.cuda.is_available()` trả `True`, sau đó truyền `--device 0` khi chạy. Lệnh và phiên bản cần được đối chiếu với [bảng cài đặt PyTorch chính thức](https://pytorch.org/get-started/previous-versions/).

### Tải lại dataset cần cho benchmark và website

Dataset và checkpoint lớn không được lưu trong GitHub. Trên Windows, sau khi clone repo và cài Python dependencies, chạy script sau để tải COCO val2017 và hai tập Open Images V7 đã ghim đúng danh sách ảnh 100/300:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\download_project_datasets.ps1
```

Script tải từ [trang COCO chính thức](https://cocodataset.org/#download) (ảnh val2017 và annotation train/val 2017) và [Open Images V7](https://storage.googleapis.com/openimages/web/download_v7.html), giải nén/chuẩn bị nhãn COCO, dựng lại annotation Open Images và tải đúng 400 ảnh validation. Hai danh sách ID nhỏ nằm trong `benchmark_data/` để mọi người tái tạo cùng subset. Không có train images, dữ liệu nhiễu sinh ra, hay toàn bộ Open Images được tải.

COCO cần khoảng 1.3 GB tải về; sau giải nén và tạo YOLO labels nên chừa khoảng 4 GB dung lượng trống. Open Images tải thêm các bảng annotation nguồn và 400 ảnh được chọn (dung lượng thay đổi theo ảnh). Có thể bỏ qua một nguồn: `.\download_project_datasets.ps1 -SkipCoco` chỉ tải Open Images; `.\download_project_datasets.ps1 -SkipOpenImages` chỉ tải COCO. Nếu tải bị ngắt, chạy lại cùng lệnh để tiếp tục file zip dang dở. Kiểm tra manifest/license của từng ảnh trước khi chia sẻ lại ảnh Open Images.

Script tạo các đường dẫn COCO được dùng trong báo cáo: `datasets/coco/full_val2017/data.yaml` và ba subset 500 ảnh (`subset_500`, `subset_500_image_set2`, `subset_500_image_set3`). Open Images được tạo tại `datasets/open_images_v7` (100 ảnh) và `datasets/open_images_v7_test_extra` (300 ảnh, thư viện website).

Để dựng lại cache prediction cho thư viện website sau khi tải dataset và có checkpoint YOLOv8s:

```powershell
.\.venv\Scripts\python.exe evaluate_open_images_subset.py --data-root datasets/open_images_v7_test_extra --source-root datasets/open_images_v7/source --out runs/open_images_v7_test_extra_eval --device 0 --imgsz 320 --batch 16 --models yolov8s.pt
```

Để tải COCO thủ công, các link chính thức là [val2017.zip](https://images.cocodataset.org/zips/val2017.zip) và [annotations_trainval2017.zip](https://images.cocodataset.org/annotations/annotations_trainval2017.zip). Giải nén sao cho có `datasets/coco/val2017/` và `datasets/coco/annotations/instances_val2017.json`, rồi tạo subset như sau:

```powershell
python prepare_coco.py --images datasets/coco/val2017 --annotations datasets/coco/annotations/instances_val2017.json --out datasets/coco/subset_500 --num-images 500 --seed 42
```

Khi tạo bộ mới, script lưu ảnh/nhãn YOLO, `data.yaml` và `manifest.json` chứa danh sách ảnh, seed và phân bố lớp. Benchmark YOLOv8n và SSDLite, các điều kiện raw/recovered, confusion matrices, biểu đồ và báo cáo đã có sẵn; xem `RESULTS_REPORT.md` và `EXPERIMENT_LOG.md`. Script tải ở trên tái tạo các tập đánh giá, không tải COCO train2017. Muốn chạy fine-tuning cần tải riêng [train2017.zip](https://images.cocodataset.org/zips/train2017.zip); thao tác đó cần khoảng 18 GB tải về và không cần cho demo website hay benchmark validation.

Ví dụ dataset YAML (`data.yaml`):

```yaml
path: D:/datasets/my-detection-set
train: images/train
val: images/val
names:
  0: person
  1: car
```

Ảnh validation cần ở `images/val`, nhãn tương ứng ở `labels/val`, với mỗi dòng nhãn là `class_id x_center y_center width height` chuẩn hóa về 0–1.

Ultralytics yêu cầu YAML có cả `train` và `val`. Trong subset val-only của dự án, khóa `train` trỏ cùng thư mục chỉ để thỏa schema; các lệnh benchmark chỉ gọi validation và tuyệt đối không dùng cấu hình này để huấn luyện.

```powershell
python robustness_eval.py --data D:/datasets/my-detection-set/data.yaml --model yolov8n.pt --out results --device 0
```

Chạy CPU bằng `--device cpu`. Có thể điều chỉnh `--imgsz`, `--batch`, `--seed`, `--fp-conf` (mặc định 0.25) và `--match-iou` (mặc định 0.5). `--no-recovery` bỏ nhánh phục hồi. Checkpoint `yolov8n.pt` có thể được tải bởi Ultralytics nếu chưa nằm sẵn trên máy; benchmark chính thức cần ghi checkpoint/version thực tế.

## Đầu ra

- `metrics.csv`: một dòng cho từng điều kiện raw/recovered; mAP, precision/recall, TP/FP/FN và số ảnh.
- `map_vs_intensity.png`, `false_positives_vs_intensity.png`: đường cong theo từng loại biến dạng.
- `confusion_<condition>.csv` và `.png`: confusion matrix per condition; lớp background là hàng/cột cuối.
- `datasets/<condition>/`: ảnh đã tạo, nhãn giữ nguyên, `data.yaml` và `manifest.json` nguồn/điều kiện.
- `ultralytics/`: artefak validation Ultralytics.

## Cách phân tích kết quả

1. Lấy mAP clean làm mốc. Tính suy giảm tuyệt đối `mAP_clean − mAP_condition` và suy giảm tương đối `% = 100 × (mAP_clean − mAP_condition) / mAP_clean`.
2. Đọc slope của từng đường cong; slope âm lớn biểu thị nhạy với biến dạng hơn. So raw/recovered ở cùng level để đo mức phục hồi.
3. Đọc FP/FN cùng mAP: FP tăng phản ánh dự đoán giả, FN tăng phản ánh vật thể bị bỏ sót. Ma trận phân lớp cho thấy lớp nào dễ nhầm.
4. So sánh với nhiều seed và khoảng tin cậy. Không kết luận “khử nhiễu cải thiện” chỉ từ một checkpoint hoặc một lần chạy.

## Giới hạn

Fog và low-light hiện là biến đổi pixel tổng hợp, không thay thế ảnh thời tiết/ban đêm thật. Augmentation không mô phỏng thay đổi màu sắc camera, motion blur, phản xạ, phân bố độ sâu hay domain shift. Đếm FP/FN là tại một ngưỡng confidence cố định và không thay thế mAP. Nếu dùng cùng validation set để tinh chỉnh preprocessing, cần test split độc lập cho kết luận cuối. Model API và checkpoint khác nhau có thể tạo số đo khác nhau.

## COCO val2017 đầy đủ

Đã chuẩn bị `datasets/coco/full_val2017` với đủ 5.000 ảnh và 80 lớp. Kết quả raw-only nằm trong `runs/full_val2017_raw_seed42`; thư mục có bảng mAP, confusion matrices và biểu đồ theo cường độ. Recovery đã chạy trên subset 500 ảnh với ba seed; đã chạy xác nhận median/CLAHE trên toàn val2017 ở `runs/full_val2017_recovered_seed42`. Xem `RESULTS_REPORT.md` và `EXPERIMENT_LOG.md`.

Tùy chọn `--families salt_pepper fog low_light --recovered-only` chỉ chạy nhánh recovery cho các họ được chọn; thêm `--levels 0.06` để giới hạn cường độ. Gaussian-NLM full-val chỉ chạy mức 0.06 vì xử lý lâu; ba mức đều có số đo recovery trên subset 500 ảnh.

Class-wise analysis của full val2017 nằm trong `runs/full_val2017_classwise`; tái tạo bằng `analyze_class_errors.py`.

So sánh YOLOv8n với SSDLite trên clean và các điều kiện mạnh chọn lọc nằm trong `runs/detector_comparison_fullval/metrics.csv`, cùng hai biểu đồ. Chạy lại phép ghép bảng và vẽ bằng `python compare_detectors.py`. Lưu ý hai detector dùng pipeline và input size khác nhau; xem giới hạn trong báo cáo trước khi so sánh trị tuyệt đối giữa model.

Per-class FP/FN của SSDLite nằm trong `runs/ssdlite_fullval_seed42/classwise/`; tạo lại bằng `python analyze_ssdlite_class_errors.py`.

So sánh per-class cùng evaluator/input 320 cho YOLOv8n và SSDLite nằm trong `runs/normalized_detector_comparison_fullval/classwise/`; tạo lại bằng `python analyze_normalized_class_errors.py`. Bảng có support, precision/recall và thay đổi theo từng lớp. Support per class là tổng ground-truth instances, bao gồm các object bị bỏ sót.

Phép so sánh đã chuẩn hóa YOLOv8n-320 với SSDLite-320 (cùng evaluator AP 101 điểm) nằm trong `runs/normalized_detector_comparison_fullval/metrics.csv`. Chạy lại YOLO bằng `python yolo_custom_eval.py --device 0 --imgsz 320 --batch 16`, sau đó ghép biểu đồ bằng `python compare_detectors_normalized.py`. NMS/post-processing vẫn theo mặc định từng kiến trúc; xem giới hạn trong báo cáo.

Đường cong mAP raw theo đủ 12 mức của hai detector nằm trong `runs/normalized_detector_comparison_fullval/raw_intensity_metrics.csv`; dựng lại bằng `python plot_detector_intensity_curves.py`. Biểu đồ: `raw_mAP_vs_intensity_both_detectors.png`.

So sánh recovery theo cường độ (20 cặp raw/recovered mà cả hai model có dữ liệu full-val) nằm trong `runs/normalized_detector_comparison_fullval/recovery_intensity_metrics.csv`; dựng biểu đồ bằng `python plot_recovery_curves.py`. Gaussian-NLM hiện chỉ có kết quả full-val ở mức 0.06.

Gaussian có thêm bilateral recovery ở cả ba mức, đặt tách với NLM. Bảng/biểu đồ nằm trong `runs/normalized_detector_comparison_fullval/gaussian_denoiser_comparison.csv` và `gaussian_denoiser_comparison.png`; tạo lại bảng bằng `python compare_gaussian_denoisers.py`.

Thống kê lặp seed42/43/44 cho Gaussian 0.06 và S&P 0.06 raw/median nằm trong `runs/normalized_detector_comparison_fullval/seed_repeat_summary.csv`; biểu đồ `seed_repeat_map50.png`; tạo lại bằng `python aggregate_fullval_seeds.py`. Đây là n=3 trên cùng split.

Sweep đầy đủ 25 điều kiện trên subset 500 ảnh qua ba seed noise (42/43/44) nằm trong `runs/repeat_seed44/metrics.csv` và `runs/repeat_analysis_seed42_43_44/`; bảng gộp có mean ± SD, cùng biểu đồ mAP và FP. Tạo lại summary bằng `python aggregate_repeats.py`.

Độ nhạy do chọn ảnh được đo thêm trên hai subset 500 ảnh không giao nhau với subset ban đầu. Mỗi subset có clean và 8 điều kiện nặng raw/recovered; summary 3 subset nằm trong `runs/image_subset_analysis/`. Tạo subset loại trừ manifest cũ bằng `prepare_coco.py --exclude-manifest ...`; gộp bằng `python aggregate_image_subsets.py`.

Recall theo kích thước COCO object (small/medium/large), so sánh clean và raw/recovered ở Gaussian, S&P, fog, low-light mức nặng, nằm trong `runs/size_stratified_analysis/`; tái tạo bằng `python analyze_size_robustness.py`. Protocol dùng YOLOv8n imgsz 640, confidence 0.25, IoU match 0.5 trên subset 500 ảnh seed42.

Độ lệch theo seed cho clean/Gaussian/S&P được tổng hợp ở `runs/size_stratified_seed_analysis/`; ví dụ chạy lại bằng `python analyze_size_robustness.py --seeds 42 43 44 --conditions clean gaussian_0.06 gaussian_0.06_recovered salt_pepper_0.06 salt_pepper_0.06_recovered --out runs/size_stratified_seed_analysis`.

Độ nhạy recall theo kích thước vật thể trên ba subset COCO rời nhau, đủ 9 điều kiện nặng, ở `runs/size_stratified_subsets/`; tạo lại bằng `python analyze_size_across_subsets.py`. Summary báo mean ± SD giữa các subset (khác với SD giữa seed nhiễu).

Phân tích thay đổi recall theo lớp cho S&P 0.06 raw/median trên ba subset, gồm counts, delta và biểu đồ mean ± SD, nằm trong `runs/classwise_subset_analysis/`; tái tạo bằng `python analyze_class_errors_subsets.py`.

Phân tích per-class cho cả Gaussian 0.06, S&P 0.06, fog 0.50 và low-light 0.25 (raw/recovered) trên ba subset ở `runs/classwise_all_conditions_subsets/`; bảng có mean ± SD recall change và recovered − raw, tạo lại bằng `python analyze_all_class_errors_subsets.py`.

Độ nhạy FP/FN theo confidence threshold 0.05/0.25/0.50 cho bốn biến dạng nặng, raw/recovered trên ba subset COCO, nằm trong `runs/confidence_threshold_analysis/`; tạo lại bằng `python analyze_confidence_threshold_sensitivity.py`. Counts dùng IoU 0.5, tính trên 100 ảnh để dễ so sánh.

Tạo hình kiểm tra định tính trên cùng một ảnh COCO qua clean/raw/recovered cho Gaussian, Salt-and-Pepper, fog và low-light bằng `python create_qualitative_figure.py`. Hình và metadata ảnh được lưu tại `runs/qualitative_coco_examples/`; bbox màu xanh là ground truth giữ nguyên. Đây là bước quan sát trực quan, không thay thế phép đo mAP.

## Dùng denoising độc lập

`preprocessing.py` cung cấp cùng các recovery methods mà benchmark dùng; `denoise.py` chạy trên một ảnh hoặc cả thư mục và giữ cấu trúc thư mục con. `auto` chọn median cho Salt-and-Pepper, NLM cho Gaussian, CLAHE cho fog/low-light. Ví dụ:

```powershell
python denoise.py --input D:/images/noisy --output D:/images/recovered --family salt_pepper
python denoise.py --input D:/images/noisy/photo.jpg --output D:/images/recovered/photo.jpg --family gaussian --method bilateral
```

Các method: `auto`, `median`, `nlmeans`, `clahe`, `bilateral`. Dùng `--overwrite` nếu cần thay thế file đầu ra đã tồn tại.

Có thể thêm `--level` để dùng policy chọn lọc đã đối chiếu trên full COCO val2017 với YOLOv8n và SSDLite. Policy dùng no-op khi lọc không có lợi nhất quán và chọn bilateral cho Gaussian 0.06, median cho S&P, CLAHE cho fog 0.30/0.50 và low-light 0.25/0.40:

```powershell
python denoise.py --input D:/images/noisy --output D:/images/selected --family gaussian --level 0.06
python denoise.py --input D:/images/noisy --output D:/images/selected --family fog --level 0.15
```

Chỉ các mức đã benchmark được chấp nhận khi bật policy này. Đây là lựa chọn tinh chỉnh trên COCO val2017, nên cần xác nhận trên dữ liệu độc lập trước khi dùng như quy tắc tổng quát.


### Noise-aware fine-tuning pilot

To build the paired clean/corrupted COCO-80 training data from the existing 5K subset, fine-tune YOLOv8n, and compare the checkpoint across the existing full-val noise sweep:

```powershell
.venv\Scripts\python.exe prepare_noise_aware_train.py
.venv\Scripts\python.exe finetune_yolo.py --model yolov8n.pt --name yolov8n_noiseaware_5k_pair_lr0001 --data datasets/coco/train_subset_5000_noise_aware/data.yaml --epochs 10 --batch 8 --device 0 --lr0 0.0001 --patience 4
.venv\Scripts\python.exe evaluate_checkpoint_corruption_sweep.py --model runs/finetune_coco80/yolov8n_noiseaware_5k_pair_lr0001/weights/best.pt --out runs/noiseaware_yolov8n_fullval_sweep_seed42 --imgsz 640 --batch 16 --device 0
.venv\Scripts\python.exe compare_noiseaware_sweep.py
```

The current pilot improves raw Salt-and-Pepper mAP but lowers clean, Gaussian, fog, and low-light absolute mAP. For the severe Salt-and-Pepper case, original YOLOv8n plus median recovery reaches 49.36% mAP50 versus 36.78% for the noise-aware checkpoint, with fewer false positives. The desktop app already defaults to original COCO-pretrained YOLOv8s and the level-aware “Tự chọn theo benchmark” recovery policy; keep that as the product path. Do not load the noise-aware checkpoint as a global replacement. See `PRODUCT_INFERENCE_PLAN.md`, `RESULTS_REPORT.md`, and the CSV/curve in `runs/noiseaware_yolov8n_fullval_sweep_seed42/`.
