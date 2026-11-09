# Product inference path after fine-tuning trials

- Keep the original COCO-pretrained YOLOv8s checkpoint for the demo; it retains all 80 COCO classes.
- Run the benchmark-selected recovery before detection when the input corruption family/level is known: median for Salt-and-Pepper, bilateral for Gaussian 0.06, and CLAHE for fog 0.30/0.50 and low-light 0.25/0.40; use no-op for levels where preprocessing did not consistently help. The app exposes this as “Tự chọn theo benchmark”.
- Do not use the mixed noise-aware YOLOv8n fine-tune as a global product checkpoint. It raised severe Salt-and-Pepper mAP50 from 24.59% to 36.78%, but baseline YOLOv8n plus median recovery reached 49.36% and had fewer false positives. The fine-tuned checkpoint also lost clean mAP and regressed on Gaussian, fog, and low-light.
- Do not use the three COCO-1K fine-tuned checkpoints as product weights: each scored below its own baseline on clean COCO validation.
- The three user room photos and their pseudo-label previews were deleted at the user's request. Open Images V7 remains as an external 8-class evaluation set; it is not full COCO-80 training data.
- The desktop app now has an optional overlapping 640px tiling mode for large images and small objects. It is off by default. Its COCO val2017 pilot slightly raised mAP@0.5 but reduced small-object recall and mAP@0.5:0.95, so it is not established as a small-object improvement.
- For a later fine-tune, start from the original COCO weights and use newly provided, fully reviewed target images with complete COCO-80 labels, plus a COCO-80 replay set. Reserve separate new target images for final evaluation.
