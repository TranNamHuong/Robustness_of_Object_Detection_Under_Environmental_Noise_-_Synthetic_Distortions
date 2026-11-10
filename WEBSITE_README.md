# Object Detection Robustness Lab website

Local React + TypeScript frontend and FastAPI service for the existing computer-vision project. It reuses the project's YOLO checkpoints, corruption/recovery code, experiment CSVs, labeled datasets, and cached predictions. It does not train a model or rerun benchmark sweeps.

## Run locally (Windows PowerShell)

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\run_local.ps1
```

Open `http://127.0.0.1:5173`. The API docs are at `http://127.0.0.1:8000/docs`. To stop the services, run:

```powershell
.\stop_local.ps1
```

The project `.venv` and the Node.js runtime used to install frontend dependencies must exist. The frontend dependencies are in `web/`; builds use the committed `web/pnpm-lock.yaml`. The launch script uses the local Vite entry point, so it does not need pnpm on `PATH`.

## Fresh clone setup

From PowerShell, in the cloned project folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
corepack enable
Set-Location web
pnpm install --frozen-lockfile
Set-Location ..
.\run_local.ps1
```

Python 3.10+ and a supported Node.js LTS release are required. If `py -3.11` is unavailable, create the virtual environment with another supported Python version. To reproduce the recorded Windows CUDA experiment environment instead, install `requirements-experiment.lock.txt` and follow the PyTorch/CUDA instructions in the root README.

The saved benchmark CSV files are included in Git, so Comparison and Research analytics work without downloading COCO or rerunning experiments. Dataset images and model weights are deliberately excluded. Without the Open Images image subset, the gallery is empty but the rest of the website still starts. Without YOLO checkpoints or cached SSD weights, the detection controls report that those models are unavailable. Follow the root README's dataset download steps only when you need the gallery or want to reproduce benchmarks; restore checkpoints separately before using image inference.

## Pages

- **Detection Lab:** upload or webcam capture, YOLOv8n/s, measured synthetic distortions, recovery, and side-by-side detection overlays. An unlabeled image shows detections/latency only; it never reports accuracy metrics.
- **Comparison:** filters and CSV export for one selected experiment/protocol at a time, with deterministic evidence-based model recommendations.
- **Research analytics:** measured degradation, restoration, count, and subset variance charts where the source contains those metrics.
- **Methodology & gallery:** pipeline and metric definitions plus ground-truth/prediction overlays from the existing 300-image Open Images V7 external set. Individual image author/license/source links are shown.

## Data and interpretation

The API reads existing CSVs under `runs/` and serves existing labeled images under `datasets/` when that dataset is present. COCO, Open Images, different input sizes/evaluators, and the unverified user-supplied reference table remain separate. The reference table is never eligible for a ranking. Recommendation results are scoped to the selected protocol and displayed condition; “insufficient evidence” is returned if comparable metrics are missing. Gallery prediction boxes use a display confidence threshold and are not a new benchmark result.

The detection API uses checkpoints already in the project and the existing `robustness_eval.py` / `preprocessing.py` functions. Images uploaded to the local API are not persisted. SSD-MobileNet is disabled until its pretrained weights are already cached locally; the application does not download weights.

## Checks

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s server/tests -v
cd web
pnpm test
pnpm run build
```

Website runtime logs and process IDs are written under `.website_runtime/`.
