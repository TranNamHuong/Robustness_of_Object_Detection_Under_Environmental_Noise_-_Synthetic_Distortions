# Robustness of Object Detection Under Environmental Noise and Synthetic Distortions

**Final project report**  
**Project status:** Core benchmark, analysis, and desktop demonstration completed  
**Primary benchmark:** COCO 2017 validation (COCO val2017)

## Abstract

This project studies how object detectors respond to synthetic image corruption and whether classical image restoration can recover detection performance. The main benchmark uses pretrained YOLOv8n on all 5,000 COCO val2017 images and evaluates Gaussian noise, salt-and-pepper noise, synthetic fog, and low-light transformations at three severity levels. Additional experiments cover repeated noise seeds and image subsets, object-size and class-wise errors, SSDLite and YOLOv8s comparisons, a small external Open Images V7 evaluation, fine-tuning pilots, and tiled inference. A Windows desktop application demonstrates detection and optional preprocessing on individual images.

Salt-and-pepper noise caused the largest measured degradation: YOLOv8n mAP@0.5 fell from 0.5185 on clean COCO val2017 to 0.2459 at severity 0.06. A 3×3 median filter raised it to 0.4936. At the same severity, Gaussian noise reduced mAP@0.5 to 0.4387; bilateral filtering raised it to 0.4817 in the YOLOv8n/YOLOv8s robustness comparison. CLAHE produced smaller, severity-dependent gains for fog and low light. These gains were not universal: preprocessing sometimes reduced scores at mild severity, and a noise-aware fine-tuned model improved salt-and-pepper results while lowering clean and several other corrupted-condition scores. The desktop demonstration therefore keeps an original COCO-pretrained YOLOv8s checkpoint and uses a benchmark-derived, level-aware preprocessing policy. Synthetic results and the 100-image external pilot are exploratory; they do not establish performance on a specific production camera or real weather conditions.

## 1. Project goals and questions

The project set out to:

1. Measure the effect of increasing synthetic corruption severity on object detection quality.
2. Compare raw corrupted images with images processed by a corruption-matched restoration method.
3. Examine false positives, false negatives, class errors, and recall by object size alongside aggregate mAP.
4. Compare detector families and model sizes under clearly stated evaluation protocols.
5. Package the inference and restoration workflow in a usable Windows desktop demo and provide scripts and artifacts for reproduction.

The principal research question was: **How does detector performance change under synthetic Gaussian noise, salt-and-pepper noise, fog, and low light, and how much of the loss can image preprocessing recover?**

## 2. Methods

### 2.1 Datasets and models

COCO val2017 is the primary benchmark. Full-split experiments use 5,000 images, 80 classes, and 36,335 ground-truth instances. A fixed 500-image subset is used for complete multi-seed sweeps and detailed analyses. Two additional disjoint 500-image subsets were used to measure sensitivity to image selection. Models are pretrained COCO checkpoints and are not fine-tuned for the primary corruption benchmark.

The primary full-validation corruption sweep uses YOLOv8n at 640-pixel input size, batch size 16, and noise seed 42. A subset sweep repeats conditions with noise seeds 42, 43, and 44. Selected severe conditions are repeated on full validation and/or across the three disjoint subsets. SSDLite320-MobileNetV3-Large and YOLOv8s are used in additional comparisons. Since those comparisons use different model transforms and, in some experiments, different AP implementations or input sizes, results are interpreted within each stated protocol rather than treated as a single interchangeable leaderboard.

An external pilot uses 100 Open Images V7 validation images, 193 boxes, and eight classes shared with the project label set. This is a limited transfer check, not a representative deployment evaluation.

### 2.2 Synthetic corruption definitions

The same image IDs and annotations are used for clean, raw-corrupted, and recovered comparisons. Labels remain unchanged because the transformations are photometric.

| Corruption | Transformation and tested levels |
|---|---|
| Gaussian noise | Add zero-mean Gaussian noise with standard deviation `level × 255`; levels 0.01, 0.03, 0.06. |
| Salt-and-pepper | Replace a total fraction of pixels with black or white, split evenly; levels 0.01, 0.03, 0.06. |
| Fog | Blend the image toward atmospheric intensity 235 using `I' = (1 − α)I + 235α`; α = 0.15, 0.30, 0.50. This is spatially uniform and does not model scene depth. |
| Low light | Multiply normalized intensity by 0.60, 0.40, or 0.25 and apply gamma 1.15. A smaller factor means a darker image. |

The tested recovery methods are Fast Non-Local Means (NLM) or bilateral filtering for Gaussian noise, a 3×3 median filter for salt-and-pepper noise, and CLAHE on the Lab lightness channel for fog and low light. The method used is stated with each result; the Gaussian method differs between some experiments.

### 2.3 Metrics and matching

Detection quality is reported primarily as mAP@0.5 and mAP@0.5:0.95 from the evaluator specified for each experiment. Confusion counts, precision, recall, and size-stratified recall use a fixed confidence threshold of 0.25 and class-aware one-to-one matching at IoU 0.5 unless stated otherwise. A wrong-class match contributes one false positive and one false negative. These thresholded counts are distinct from the confidence sweep used by the AP evaluators. Accordingly, a lower FP count alone is not evidence of improved detection: it can result from missed objects and higher FN.

### 2.4 Reproducibility environment

The recorded experiment environment is Windows, Python 3.10.3, NVIDIA GeForce RTX 3060 12 GB, CUDA 12.8, PyTorch 2.10.0, Torchvision 0.25.0, Ultralytics 8.4.204, and OpenCV 4.13.0. Direct dependency pins are in `requirements-experiment.lock.txt`; a full environment snapshot is in `environment-snapshot-windows-cu128.txt`. Large datasets, generated corrupted images, and model weights are excluded from source control and must be restored locally before reproducing inference results.

## 3. Main results

### 3.1 YOLOv8n full-validation corruption sweep

The clean baseline on all of COCO val2017 is mAP@0.5 = **0.5185** and mAP@0.5:0.95 = **0.3671**. The table reports raw corrupted images. Percentage loss is relative to the clean mAP@0.5 baseline.

| Condition | Severity | mAP@0.5 | mAP@0.5:0.95 | Relative mAP@0.5 loss |
|---|---:|---:|---:|---:|
| Clean | 0 | 0.5185 | 0.3671 | 0.0% |
| Gaussian | 0.01 | 0.5161 | 0.3651 | 0.5% |
| Gaussian | 0.03 | 0.4927 | 0.3449 | 5.0% |
| Gaussian | 0.06 | 0.4387 | 0.3037 | 15.4% |
| Salt-and-pepper | 0.01 | 0.4337 | 0.3021 | 16.3% |
| Salt-and-pepper | 0.03 | 0.3464 | 0.2344 | 33.2% |
| Salt-and-pepper | 0.06 | 0.2459 | 0.1636 | 52.6% |
| Fog | 0.15 | 0.5108 | 0.3615 | 1.5% |
| Fog | 0.30 | 0.4985 | 0.3521 | 3.9% |
| Fog | 0.50 | 0.4712 | 0.3320 | 9.1% |
| Low light | 0.60 | 0.5120 | 0.3639 | 1.2% |
| Low light | 0.40 | 0.4965 | 0.3509 | 4.2% |
| Low light | 0.25 | 0.4591 | 0.3224 | 11.4% |

Salt-and-pepper noise is the most damaging corruption in this benchmark at all tested levels. Performance also degrades monotonically with severity for Gaussian noise and fog. Low-light severity follows the expected direction: the darkest condition (factor 0.25) produces the largest low-light loss. The FP count at confidence 0.25 decreases under severe corruption as detections are missed; for example, salt-and-pepper 0.06 has 5,212 FP versus 8,794 on clean, while FN rises from 18,414 to 27,102. Both types of error and mAP must therefore be considered together.

### 3.2 Restoration on full COCO val2017

| Corruption | Level | Raw mAP@0.5 | Recovered mAP@0.5 | Change | FP raw → recovered | FN raw → recovered |
|---|---:|---:|---:|---:|---:|---:|
| Salt-and-pepper | 0.01 | 0.4337 | 0.4997 | +0.0659 | 7,440 → 8,334 | 21,223 → 19,188 |
| Salt-and-pepper | 0.03 | 0.3464 | 0.4985 | +0.1521 | 6,237 → 8,255 | 24,100 → 19,281 |
| Salt-and-pepper | 0.06 | 0.2459 | 0.4936 | +0.2478 | 5,212 → 8,168 | 27,102 → 19,438 |
| Fog | 0.15 | 0.5108 | 0.5079 | −0.0029 | 8,592 → 8,615 | 18,743 → 18,739 |
| Fog | 0.30 | 0.4985 | 0.5072 | +0.0087 | 8,327 → 8,472 | 19,181 → 18,842 |
| Fog | 0.50 | 0.4712 | 0.4998 | +0.0286 | 7,705 → 8,144 | 20,339 → 19,224 |
| Low light | 0.60 | 0.5120 | 0.5087 | −0.0033 | 8,091 → 8,271 | 18,945 → 18,969 |
| Low light | 0.40 | 0.4965 | 0.5026 | +0.0062 | 7,513 → 7,872 | 19,751 → 19,350 |
| Low light | 0.25 | 0.4591 | 0.4848 | +0.0257 | 6,233 → 7,155 | 21,338 → 20,336 |

Median filtering recovered most of the salt-and-pepper loss, raising mAP@0.5 at level 0.06 from 0.2459 to 0.4936 and reducing FN by 7,664. It also increases FP relative to the raw corrupted image because the detector can make more predictions after image details are restored. CLAHE provides modest gains for severe fog and low light but slightly lowers mAP at the mildest tested levels. Gaussian NLM at level 0.06 raises mAP by only 0.0073 (0.4387 to 0.4460), recovering a small share of the loss. Bilateral filtering was more effective in the later YOLO robustness comparison (Section 3.6), so Gaussian restoration conclusions depend on the selected method.

### 3.3 Repeatability and image-selection sensitivity

On a fixed 500-image COCO subset, the full set of 25 clean/raw/recovered conditions was evaluated with three noise seeds. Clean mAP@0.5 was 0.5440. At salt-and-pepper 0.06, raw mAP was **0.2692 ± 0.0080** and median-recovered mAP was **0.5169 ± 0.0034** (mean ± sample standard deviation across seeds). At Gaussian 0.06, NLM did not help consistently at mild-to-moderate levels; its mean effect at 0.06 was a small positive change.

To assess sensitivity to image selection, nine severe conditions were also tested across three disjoint 500-image subsets. Salt-and-pepper 0.06 produced mean raw mAP@0.5 of **0.2674 ± 0.0120** and median-recovered mAP of **0.5185 ± 0.0283**. Gaussian 0.06 raw and NLM were 0.4616 ± 0.0175 and 0.4705 ± 0.0180. Fog 0.50 raw and CLAHE were 0.4994 ± 0.0193 and 0.5249 ± 0.0201; low light 0.25 raw and CLAHE were 0.4812 ± 0.0183 and 0.5095 ± 0.0230. These repeated-subset results support the main ranking, while showing that image composition contributes meaningful variability. The subsets all come from COCO val2017 and are not independent datasets.

### 3.4 Errors by object size, class, and confidence

Recall is lowest for small objects under clean and corrupted conditions. Across three disjoint subsets, mean small/medium/large recall for clean images is 0.212/0.520/0.741. Under salt-and-pepper 0.06 it falls to 0.098/0.267/0.389. Median filtering raises it to 0.168/0.481/0.736. The large-object recall is nearly restored, while small-object recall remains substantially below clean.

Class-wise analysis finds varying sensitivity. On the three-subset analysis, cat, mouse, airplane, and refrigerator are among the classes with the largest salt-and-pepper recall drops, although those estimates are descriptive and depend on class support. The full-val confusion analysis also shows that person false negatives increase substantially under severe salt-and-pepper corruption. In other conditions, some class-specific recovery effects are negative, reinforcing that an aggregate mAP gain does not mean every category benefits.

Confidence sweeps show the expected precision-recall trade-off: raising confidence reduces FP but increases FN. At confidence 0.25 averaged across the three subsets, salt-and-pepper 0.06 has 101.5 FP and 556.6 FN per 100 images; median recovery has 166.3 FP and 399.9 FN, compared with 180.1 FP and 378.9 FN for clean. Recovery moves both counts toward clean rather than uniformly minimizing them.

### 3.5 Detector comparisons

At 320-pixel input, YOLOv8n, YOLOv8s, and SSDLite were evaluated on the same full COCO split using a shared 101-point AP evaluator for the normalized comparison. For the selected clean and severe conditions:

| Condition | YOLOv8n | YOLOv8s | SSDLite |
|---|---:|---:|---:|
| Clean | 40.37% | 49.12% | 33.86% |
| Gaussian 0.06 raw | 37.66% | 46.40% | 31.84% |
| Gaussian recovered | 38.13% | 46.57% | 30.65% |
| Salt-and-pepper 0.06 raw | 26.12% | 32.17% | 22.54% |
| Salt-and-pepper recovered | 39.51% | 48.12% | 32.98% |
| Fog 0.50 raw | 35.76% | 44.23% | 30.05% |
| Fog recovered | 38.59% | 47.81% | 32.23% |
| Low light 0.25 raw | 34.82% | 43.80% | 28.72% |
| Low light recovered | 37.28% | 46.25% | 30.50% |

Within this protocol, YOLOv8s scores highest and salt-and-pepper remains the most harmful condition for all three detectors. At 640-pixel input, separate Ultralytics runs show YOLOv8s improving clean mAP@0.5 over YOLOv8n by roughly 9 percentage points, with mean single-image latency on an RTX 3060 increasing from 9.76 ms to 10.33 ms (200 images; measured with the specified inference settings). These 640-pixel figures are not directly interchangeable with the shared-evaluator 320-pixel table.

For the 100-image Open Images V7 pilot, mAP@0.5 on eight classes is 65.8% for YOLOv8n, 65.4% for YOLOv8s, and 61.9% for SSDLite. YOLOv8s yields more true positives at confidence 0.25 than YOLOv8n, but also more false positives. The small sample and class support (193 boxes in total) prevent a deployment-level conclusion.

### 3.6 Restoration alternatives and model adaptation

On full COCO val2017 at input 320, salt-and-pepper median filtering and CLAHE for severe fog/low light improve results for both YOLOv8n and SSDLite. Gaussian NLM is architecture-dependent: it modestly helps YOLOv8n but lowers SSDLite mAP in the selected comparison. A separate full-val Gaussian denoiser sweep found bilateral filtering preferable for the tested YOLO checkpoints at severe Gaussian noise. This supports choosing restoration by measured condition and severity rather than applying one filter universally.

Fine-tuning pilots did not produce a better general-purpose COCO detector. Fine-tuning on 1,000 COCO images lowered validation mAP for YOLOv8n, YOLOv8s, and SSDLite. A 5,000-image YOLOv8n pilot reached 47.87% best mAP@0.5; a separate longer-capped run stopped early and reached 43.30%, both below the pretrained baseline. The training schedules differed, so this was not a controlled epoch-count comparison.

Noise-aware YOLOv8n fine-tuning improved salt-and-pepper 0.06 mAP@0.5 from 24.59% to 36.78%, but clean mAP fell from 51.85% to 47.30%, and absolute mAP fell for Gaussian, fog, and low light. Its mean mAP across 12 corrupted conditions was slightly lower than baseline. Baseline YOLOv8n with median recovery reached 49.36% on salt-and-pepper 0.06 with fewer false positives than the noise-aware checkpoint. The adapted checkpoint is therefore not selected as the global product model.

### 3.7 Tiled inference

On full COCO val2017, YOLOv8s tiled inference (640-pixel tiles, 25% overlap) was compared with full-frame inference. Tiling increased mAP@0.5 from 59.79% to 60.26%, but lowered mAP@0.5:0.95 from 43.75% to 43.23%. Small-object recall changed from 31.69% to 31.40%, while false positives fell from 10,576 to 9,350. This pilot does not show improved small-object recall, so tiling remains an optional, disabled-by-default demonstration setting.

## 4. Software implementation and project deliverables

The repository contains a reusable robustness evaluation pipeline and a Windows desktop MVP.

- `robustness_eval.py`, `ssdlite_eval.py`, and `yolo_custom_eval.py` run detector evaluation and save metrics, predictions, and confusion outputs.
- `preprocessing.py` and `denoise.py` provide the shared restoration functions and a standalone image/file-tree CLI.
- `prepare_coco.py`, `prepare_open_images_subset.py`, and related scripts create datasets, labels, fixed subsets, and manifests.
- Aggregation and analysis scripts compute repeated-run summaries, confidence sensitivity, object-size recall, and class-wise errors.
- `app.py` provides the desktop demo. Users can load an image, select or simulate a corruption, run detection before and after restoration, choose model input size and confidence, and save the images and JSON metadata. The app defaults to YOLOv8s and supports full-frame or optional tiled inference.
- `server/` and `web/` contain a separate web/API implementation and visualization frontend; setup and data requirements are documented in `WEBSITE_README.md`.
- `EXPERIMENT_LOG.md` records experiment progression; `RESULTS_REPORT.md` contains detailed benchmark tables and artifact references; `TARGET_DOMAIN_EVALUATION.md` documents how to evaluate labeled target images.

The desktop application is an image-level demonstration: object counts and confidence scores on a single image are not substitutes for mAP on a labeled dataset. The app's benchmark-based restoration policy supports tested corruption families and levels; it does not automatically identify unknown real-world noise.

## 5. Discussion

The experiments show that corruption type matters as much as nominal severity. Sparse extreme pixel changes from salt-and-pepper noise strongly disrupt detection, and a small median filter removes much of that damage. Gaussian noise is less destructive at comparable configured levels, while CLAHE helps most on the strongest fog and low-light transformations. The restoration method can itself remove useful detail, explaining small negative changes at mild levels and class-specific regressions.

More model capacity improves clean COCO performance in the YOLOv8n/YOLOv8s comparison, but does not remove robustness gaps. The detector still loses substantial performance under severe salt-and-pepper noise. Fine-tuning on a small subset can trade clean accuracy for robustness to a specific synthetic corruption, and the experiments show that a tested preprocessing step can outperform that trade-off for the severe salt-and-pepper case.

The strongest recommendations supported by the current evidence are limited to the benchmarked setting: retain the pretrained YOLOv8s checkpoint for the demo; use median filtering for salt-and-pepper corruption; use CLAHE selectively for severe fog/low light; and evaluate Gaussian restoration choices per detector and severity. Product deployment decisions require labeled images from the actual target camera and environment.

## 6. Limitations and future work

1. **Synthetic-to-real gap:** Fog is uniform and depth-independent; low light is a simple intensity/gamma transformation. Neither covers realistic weather, motion blur, sensor noise, glare, compression, or camera-specific artifacts.
2. **Validation reuse:** Recovery policies and comparisons are selected using COCO validation data. A separate held-out test split is needed for unbiased final estimates.
3. **Limited independent replication:** Three noise seeds and three disjoint subsets quantify some variability, but all subsets come from the same COCO validation distribution. Several full-split conditions have only one noise seed.
4. **External pilot scale:** The Open Images evaluation has 100 images and eight shared classes. It is insufficient to represent a target domain or support fine-tuning decisions.
5. **Metric protocol differences:** Input size, model transforms, NMS, confidence handling, and AP evaluator differ across some experiments. Cross-model comparisons should only be made within a matched protocol.
6. **Restoration is not free:** Median, NLM, bilateral, and CLAHE can alter texture and fine details; results vary by class, severity, and detector.
7. **Fine-tuning scope:** The training pilots do not constitute a broad hyperparameter search. Their negative outcomes apply to the tested datasets and schedules.

Recommended next work is to collect and annotate a representative target-domain validation/test set, choose preprocessing only on the validation portion, assess the final policy on the untouched test portion, and report latency and memory on the intended deployment hardware. If further training is justified, use a larger, domain-matched training set and track both clean and corrupted performance to make trade-offs explicit.

## 7. Conclusion

The project completed its core objective: it quantified how four synthetic corruption families affect object detection, tested matched restoration methods, analyzed errors at multiple levels, compared detectors, and packaged the workflow in an interactive desktop demo. The clearest result is the severe effect of salt-and-pepper noise and the strong recovery from median filtering. Fog and low-light restoration provides smaller and severity-dependent benefits; Gaussian denoising depends on the selected filter. Larger YOLO models improve benchmark accuracy but remain vulnerable to corruption, while the tested fine-tuning runs do not justify replacing pretrained weights.

These findings are useful as a reproducible synthetic robustness study and as a prototype workflow. They do not establish real-world robustness for an unmeasured camera or application. That conclusion requires independent labeled target-domain data.

## 8. Reproduction and artifact guide

After installing dependencies and restoring the datasets/checkpoints, representative commands include:

```powershell
# Full COCO val2017 YOLOv8n corruption benchmark
.venv\Scripts\python.exe robustness_eval.py --data datasets/coco/full_val2017/data.yaml --model yolov8n.pt --out runs/full_val2017_raw_seed42 --device 0 --imgsz 640 --batch 16 --seed 42

# Compare the detector family using the normalized 320-pixel protocol
.venv\Scripts\python.exe yolo_custom_eval.py --device 0 --imgsz 320 --batch 16
.venv\Scripts\python.exe compare_detectors_normalized.py

# Launch the desktop demo
.venv\Scripts\python.exe app.py
```

Key artifacts and documentation:

- Detailed experimental record: `RESULTS_REPORT.md`
- Experiment commands and chronology: `EXPERIMENT_LOG.md`
- Project scope and status: `PROJECT_PLAN.md`
- Dataset and environment setup: `README.md`
- Target-domain evaluation procedure: `TARGET_DOMAIN_EVALUATION.md`
- Full-validation raw metrics: `runs/full_val2017_raw_seed42/metrics.csv`
- Full-validation recovered metrics: `runs/full_val2017_recovered_seed42/metrics.csv`
- Raw-versus-recovered summary: `runs/full_val2017_comparison_seed42/raw_vs_recovered.csv`
- Repeated-seed summary: `runs/repeat_analysis_seed42_43_44/metrics_summary.csv`
- Three-subset analysis: `runs/image_subset_analysis/metrics_summary_by_image_subset.csv`
- Normalized detector comparison: `runs/normalized_detector_comparison_fullval/metrics.csv`
- Open Images pilot: `runs/open_images_v7_eval/summary.csv`
- Tiled inference: `runs/tiled_inference_eval/summary.csv`

Large datasets, generated image variants, checkpoints, and local environments are not committed. The dataset download and reconstruction steps are documented in `README.md` and `download_project_datasets.ps1`.
