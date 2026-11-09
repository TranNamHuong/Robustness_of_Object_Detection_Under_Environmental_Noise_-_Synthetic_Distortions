# Object Detection Robustness Lab Website Plan

## Project audit

The repository is a Windows Python project with an existing Tkinter demo in `app.py`, shared corruption/recovery logic in `robustness_eval.py` and `preprocessing.py`, YOLOv8n/s weights in the project root, and an SSDLite evaluator in `ssdlite_eval.py`. FastAPI, Uvicorn, OpenCV, Ultralytics, PyTorch, and CUDA are available in `.venv`. There is no existing React frontend, HTTP API, or Git repository.

Existing measured data includes COCO val2017 results for 5,000 images (multiple evaluators/input sizes), three 500-image subset runs with an aggregate mean and sample standard deviation, and Open Images V7 subsets of 100 and 300 images. These are separate protocols. The Open Images evaluator covers 8 classes in the 100-image pilot and 7 classes in the disjoint 300-image expansion; its mAP is not a COCO-80 result. User-provided historical figures in the specification have no confirmed condition/protocol and will be labeled as reference-only or omitted from calculated comparisons.

## Data integration rules

- Read benchmark records from the existing CSVs and manifests; do not re-run benchmark sweeps during web requests.
- Retain dataset, subset, model, image size, metric implementation, thresholds, source file, and available seed metadata with each result.
- Never combine records across incompatible protocols. Surface the protocol and sample count in tables, charts, exports, and recommendations.
- Compute mAP degradation/recovery only when a clean/noisy/restored comparison is available in the same protocol. Keep the existing three-subset aggregate as its own dataset view.
- Compute TP/FP/FN only from labeled evaluation records and their documented matching thresholds. For arbitrary uploads/webcam, return detections and latency only.
- Keep original pretrained weights. Do not train or fine-tune from the website.

## Phases and verification gates

| Phase | Deliverables | Gate before moving on |
| --- | --- | --- |
| P1 UI shell | React + TypeScript + Vite application, four routed pages, Vietnamese-first bilingual shell, responsive navigation, design tokens, empty/loading/error states, and truthful static methodology copy. | Production frontend build; automated route and accessibility smoke checks; no fabricated metrics in the shell. |
| P2 Live detection | FastAPI API, model readiness, YOLOv8n/s and SSDLite adapters, upload validation, existing corruption transforms, recovery/NLM, synchronized original/noisy/restored results, compare, overlay, JSON/image exports, request limits, and device fallback. | API/unit tests plus one real image smoke inference using the existing checkpoint; fail cleanly for unavailable weights/GPU. |
| P3 Benchmark analytics | Read-only adapters for the compatible COCO 5k protocol, 3×500 subset aggregate, Open Images pilot and expansion; filters, metrics, provenance, plots, confusion data where valid, and export. | Tests compare API rows to source CSVs and verify protocol separation, sample counts, missing-data behavior, and no fake values. |
| P4 Recommendation and methods | Deterministic evidence-based recommendations, “insufficient evidence” behavior, methodology, formulas, limitations, glossary, guided tour, and a labeled qualitative gallery using existing artifacts. | Unit tests for every criterion and missing/incompatible evidence; gallery paths and labels checked against source manifests. |
| P5 QA and delivery | Accessibility and responsive polish, webcam permission/error flow if the available browser/API setup supports it, exports, run scripts, `.env.example`, and README setup/reproducibility docs. | Full frontend build and backend checks, endpoint smoke checks, manual browser review at desktop/mobile widths, and verified start/stop commands. |

## Proposed local structure

```text
web/                    React + TypeScript + Vite client
  src/                  routed pages, shared components, API client, styles
server/                 FastAPI application and data/inference adapters
  robustness_api/       configuration, schemas, corruption, inference, data access
  tests/                unit and integration tests
artifacts/              generated website preview/export artifacts only
WEBSITE_IMPLEMENTATION_PLAN.md
```

Existing datasets, model weights, experiment outputs, and the desktop application remain in place. The website will run locally because inference relies on this project's local Python environment, CUDA device, and checkpoints; deployment is not part of these implementation phases.

## Implementation sequence

1. Build the UI shell and validate routes, responsive navigation, and empty states.
2. Add and test the API contracts and real detection pipeline.
3. Connect verified benchmark records and test data provenance and comparison boundaries.
4. Add recommendation rules, methodology content, glossary, and gallery.
5. Complete exports, accessibility/responsive review, documentation, and launch scripts.

## Implementation status

Last updated: 2026-11-08

| Phase | Status | Implemented and checked |
| --- | --- | --- |
| P1 UI shell | Complete | Four responsive routes, bilingual labels, route tests, production build. |
| P2 Live detection | Complete | Local FastAPI API; YOLOv8n/s checkpoint readiness; upload/webcam UI; measured noise and recovery; compare; overlays and JSON. Live HTTP smoke check used YOLOv8s on `cuda:0` (RTX 3060), both clean and Gaussian 0.06. Arbitrary uploaded images return no accuracy metrics. SSD remains unavailable unless its existing weights are cached. |
| P3 Benchmark analytics | Complete | Six existing measured protocol sources plus isolated unverified reference data; metrics, filters, charts, export, counts and subset SD. API smoke check returned 18 shared-protocol COCO records. |
| P4 Recommendation and methods | Complete | Six criterion rules, insufficient-evidence handling, protocol-only rankings, methodology/glossary and paginated Open Images 300 GT/prediction gallery with per-image license/source. |
| P5 QA and delivery | Complete | Windows start/stop scripts, website README, `.gitignore`, 12 backend unit tests, 5 frontend route tests and production build. Local HTTP checks returned website 200, GPU readiness, metrics/gallery/recommendation data and real inference. |

The automatic headless screenshot attempt did not produce image files in the available environment. Build, unit, HTTP/API and actual GPU inference checks passed. The user can review the running site at `http://127.0.0.1:5173`.
