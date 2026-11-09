#!/usr/bin/env python
"""Local desktop MVP for image corruption, recovery, and YOLO detection."""
from __future__ import annotations

import json
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO

from detection_inference import predict_image
from denoise import LEVEL_AWARE_AUTO
from robustness_eval import Condition, perturb
from preprocessing import recover_image


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "yolov8s.pt"
FAMILIES = {
    "Ảnh gốc": "clean",
    "Gaussian": "gaussian",
    "Salt-and-Pepper": "salt_pepper",
    "Fog mô phỏng": "fog",
    "Low-light": "low_light",
}
LEVELS = {
    "clean": [0.0],
    "gaussian": [0.01, 0.03, 0.06],
    "salt_pepper": [0.01, 0.03, 0.06],
    "fog": [0.15, 0.30, 0.50],
    "low_light": [0.60, 0.40, 0.25],
}
METHODS = {
    "Tự chọn theo benchmark": "auto",
    "Không khử nhiễu": "none",
    "Median": "median",
    "NLM": "nlmeans",
    "CLAHE": "clahe",
    "Bilateral": "bilateral",
}


@dataclass
class DetectionView:
    image: np.ndarray
    count: int
    classes: dict[str, int]
    average_confidence: float


def annotate(image: np.ndarray, result, names) -> DetectionView:
    canvas = image.copy()
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return DetectionView(canvas, 0, {}, 0.0)
    xyxy = boxes.xyxy.detach().cpu().numpy().astype(int)
    class_ids = boxes.cls.detach().cpu().numpy().astype(int)
    scores = boxes.conf.detach().cpu().numpy()
    counts: Counter[str] = Counter()
    for box, class_id, score in zip(xyxy, class_ids, scores):
        name = str(names.get(class_id, class_id)) if isinstance(names, dict) else str(names[class_id])
        counts[name] += 1
        color_seed = int(class_id) * 37
        color = ((37 + color_seed) % 220 + 35,
                 (113 + color_seed * 2) % 220 + 35,
                 (191 + color_seed * 3) % 220 + 35)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {score:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(y1, text_h + 8)
        cv2.rectangle(canvas, (x1, top - text_h - 8), (x1 + text_w + 6, top + baseline - 3), color, -1)
        cv2.putText(canvas, label, (x1 + 3, top - 5), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return DetectionView(canvas, len(scores), dict(counts), float(np.mean(scores)))


def method_for(family: str, level: float, requested: str) -> str:
    if requested != "auto":
        return requested
    known = LEVEL_AWARE_AUTO.get(family, {})
    match = next((value for key, value in known.items() if abs(float(key) - level) < 1e-8), None)
    if match is not None:
        return match
    # The original family-only default remains available for unbenchmarked levels.
    return {"gaussian": "nlmeans", "salt_pepper": "median",
            "fog": "clahe", "low_light": "clahe"}[family]


class RobustVisionApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Robust Vision Lab")
        self.root.geometry("1580x930")
        self.root.minsize(1180, 740)
        self.model_path = DEFAULT_MODEL
        self.model = None
        self.source_path: Path | None = None
        self.source_image: np.ndarray | None = None
        self.last_result = None
        self.photo_refs = {}
        self.title_labels = {}
        self._build_ui()

    def _build_ui(self):
        self.root.configure(bg="#f3f5f8")
        header = tk.Frame(self.root, bg="#132238", padx=24, pady=16)
        header.pack(fill="x")
        tk.Label(header, text="Robust Vision Lab", font=("Segoe UI", 20, "bold"),
                 fg="white", bg="#132238").pack(anchor="w")
        tk.Label(header, text="Mô phỏng nhiễu • khử nhiễu • phát hiện vật thể",
                 font=("Segoe UI", 10), fg="#c8d4e2", bg="#132238").pack(anchor="w", pady=(3, 0))

        controls = tk.Frame(self.root, bg="white", padx=16, pady=12,
                            highlightbackground="#dce3eb", highlightthickness=1)
        controls.pack(fill="x", padx=16, pady=14)
        self.open_button = ttk.Button(controls, text="Mở ảnh", command=self.open_image)
        self.open_button.grid(row=0, column=0, padx=(0, 10), pady=4)
        self.path_var = tk.StringVar(value="Chưa chọn ảnh")
        ttk.Label(controls, textvariable=self.path_var, width=42).grid(row=0, column=1, sticky="w")

        ttk.Label(controls, text="Chế độ").grid(row=1, column=0, padx=(0, 6), pady=(8, 2), sticky="e")
        self.mode_var = tk.StringVar(value="Mô phỏng nhiễu")
        self.mode_box = ttk.Combobox(controls, textvariable=self.mode_var,
                                     values=["Mô phỏng nhiễu", "Ảnh đầu vào đã nhiễu"],
                                     state="readonly", width=25)
        self.mode_box.grid(row=1, column=1, sticky="w", pady=(8, 2))
        self.mode_box.bind("<<ComboboxSelected>>", self.on_mode_change)

        ttk.Label(controls, text="Input size").grid(row=1, column=2, padx=(18, 6), pady=(8, 2), sticky="e")
        self.imgsz_var = tk.StringVar(value="640")
        self.imgsz_box = ttk.Combobox(controls, textvariable=self.imgsz_var,
                                      values=["640", "960", "1280"],
                                      state="readonly", width=8)
        self.imgsz_box.grid(row=1, column=3, sticky="w", pady=(8, 2))

        self.tile_var = tk.BooleanVar(value=False)
        self.tile_box = ttk.Checkbutton(controls, text="Tiling 640px • vật thể nhỏ",
                                        variable=self.tile_var)
        self.tile_box.grid(row=1, column=6, columnspan=2, sticky="w", padx=(18, 4), pady=(8, 2))

        ttk.Label(controls, text="Confidence").grid(row=1, column=4, padx=(18, 6), pady=(8, 2), sticky="e")
        self.conf_var = tk.StringVar(value="0.25")
        self.conf_box = ttk.Combobox(controls, textvariable=self.conf_var,
                                     values=["0.10", "0.25", "0.40"],
                                     state="readonly", width=8)
        self.conf_box.grid(row=1, column=5, sticky="w", pady=(8, 2))

        ttk.Label(controls, text="Biến dạng").grid(row=0, column=2, padx=(18, 6))
        self.family_var = tk.StringVar(value="Ảnh gốc")
        self.family_box = ttk.Combobox(controls, textvariable=self.family_var,
                                       values=list(FAMILIES), state="readonly", width=18)
        self.family_box.grid(row=0, column=3, padx=4)
        self.family_box.bind("<<ComboboxSelected>>", self.on_family_change)

        ttk.Label(controls, text="Mức").grid(row=0, column=4, padx=(12, 5))
        self.level_var = tk.StringVar(value="0")
        self.level_box = ttk.Combobox(controls, textvariable=self.level_var,
                                      state="readonly", width=8)
        self.level_box.grid(row=0, column=5, padx=4)
        self.on_family_change()

        ttk.Label(controls, text="Khử nhiễu").grid(row=0, column=6, padx=(12, 5))
        self.method_var = tk.StringVar(value="Tự chọn theo benchmark")
        self.method_box = ttk.Combobox(controls, textvariable=self.method_var,
                                       values=list(METHODS), state="readonly", width=24)
        self.method_box.grid(row=0, column=7, padx=4)

        self.run_button = ttk.Button(controls, text="Xử lý và phát hiện", command=self.run_pipeline)
        self.run_button.grid(row=0, column=8, padx=(14, 6))
        self.save_button = ttk.Button(controls, text="Lưu kết quả", command=self.save_results, state="disabled")
        self.save_button.grid(row=0, column=9, padx=6)

        self.model_var = tk.StringVar(value="YOLOv8s pretrained trên COCO • 80 lớp")
        self.device = "0" if torch.cuda.is_available() else "cpu"
        device_label = "GPU CUDA" if self.device == "0" else "CPU"
        self.status_var = tk.StringVar(value=f"Sẵn sàng • {device_label} • conf 0.25")
        info = tk.Frame(self.root, bg="#f3f5f8")
        info.pack(fill="x", padx=20, pady=(0, 8))
        ttk.Label(info, textvariable=self.model_var).pack(side="left")
        ttk.Label(info, textvariable=self.status_var).pack(side="right")

        panels = tk.Frame(self.root, bg="#f3f5f8")
        panels.pack(fill="both", expand=True, padx=16, pady=(0, 12))
        self.image_labels = {}
        panel_titles = [("original", "Ảnh gốc • baseline detection"),
                        ("distorted", "Ảnh nhiễu • detection trước recovery"),
                        ("processed", "Kết quả sau khử nhiễu + YOLO")]
        for col, (key, title) in enumerate(panel_titles):
            card = tk.Frame(panels, bg="white", padx=10, pady=10,
                            highlightbackground="#dce3eb", highlightthickness=1)
            card.grid(row=0, column=col, sticky="nsew", padx=6)
            panels.grid_columnconfigure(col, weight=1, uniform="image")
            panels.grid_rowconfigure(0, weight=1)
            title_label = tk.Label(card, text=title, font=("Segoe UI", 11, "bold"),
                                   bg="white", fg="#20334b")
            title_label.pack(anchor="w", pady=(0, 8))
            self.title_labels[key] = title_label
            image_label = tk.Label(card, text="Mở ảnh để bắt đầu", bg="#e8edf3",
                                   fg="#69798c", font=("Segoe UI", 11),
                                   width=48, height=28, compound="center")
            image_label.pack(fill="both", expand=True)
            self.image_labels[key] = image_label

        self.summary_var = tk.StringVar(value="Kết quả phát hiện sẽ hiển thị tại đây.")
        summary = tk.Label(self.root, textvariable=self.summary_var, justify="left",
                           anchor="w", bg="white", fg="#24364b", padx=18, pady=12,
                           font=("Segoe UI", 10), relief="solid", bd=1)
        summary.pack(fill="x", padx=22, pady=(0, 14))

    def on_family_change(self, _event=None):
        family = FAMILIES[self.family_var.get()]
        values = [f"{level:g}" for level in LEVELS[family]]
        self.level_box["values"] = values
        self.level_var.set(values[-1] if family != "low_light" else values[-1])
        self.level_box.configure(state="disabled" if family == "clean" else "readonly")

    def on_mode_change(self, _event=None):
        if self.mode_var.get() == "Ảnh đầu vào đã nhiễu":
            self.title_labels["original"].configure(text="Ảnh đầu vào • detections hiện tại")
            self.title_labels["distorted"].configure(text="Trước recovery • detections")
        else:
            self.title_labels["original"].configure(text="Ảnh sạch • baseline detection")
            self.title_labels["distorted"].configure(text="Ảnh nhiễu • detection trước recovery")

    def open_image(self):
        path = filedialog.askopenfilename(
            title="Chọn ảnh cần kiểm tra",
            filetypes=[("Ảnh", "*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff"), ("Tất cả", "*.*")])
        if not path:
            return
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            messagebox.showerror("Không mở được ảnh", "OpenCV không đọc được file ảnh này.")
            return
        self.source_path = Path(path)
        self.source_image = image
        self.path_var.set(f"{self.source_path.name} • {image.shape[1]}×{image.shape[0]}")
        self._show_image("original", image)
        self._show_image("distorted", None)
        self._show_image("processed", None)
        self.summary_var.set("Ảnh đã sẵn sàng. Chọn điều kiện rồi bấm ‘Xử lý và phát hiện’.")
        self.save_button.configure(state="disabled")

    def _show_image(self, key: str, image: np.ndarray | None, boxes=None):
        label = self.image_labels[key]
        if image is None:
            label.configure(image="", text="Chưa có kết quả")
            label.image = None
            return
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        pil.thumbnail((500, 610), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        self.photo_refs[key] = photo
        label.configure(image=photo, text="")
        label.image = photo

    def run_pipeline(self):
        if self.source_image is None:
            messagebox.showinfo("Chọn ảnh", "Mở một ảnh trước khi chạy phát hiện.")
            return
        family = FAMILIES[self.family_var.get()]
        level = float(self.level_var.get() or 0)
        requested_method = METHODS[self.method_var.get()]
        input_mode = self.mode_var.get()
        method = method_for(family, level, requested_method) if family != "clean" else "none"
        source = self.source_image.copy()
        seed = 42
        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.family_box.configure(state="disabled")
        self.level_box.configure(state="disabled")
        self.method_box.configure(state="disabled")
        self.mode_box.configure(state="disabled")
        self.imgsz_box.configure(state="disabled")
        self.tile_box.configure(state="disabled")
        self.conf_box.configure(state="disabled")
        self.status_var.set("Đang xử lý ảnh và chạy YOLO…")
        family_label = self.family_var.get()
        imgsz = int(self.imgsz_var.get())
        confidence = float(self.conf_var.get())
        tiled = bool(self.tile_var.get())

        def worker():
            try:
                if input_mode == "Ảnh đầu vào đã nhiễu" or family == "clean":
                    distorted = source.copy()
                else:
                    distorted = perturb(source, Condition(f"{family}_{level:g}", family, level),
                                        np.random.default_rng(seed))
                processed = (distorted.copy() if method == "none" else
                             recover_image(distorted, family, method))
                if self.model is None:
                    if not self.model_path.exists():
                        raise FileNotFoundError(f"Không tìm thấy checkpoint: {self.model_path}")
                    self.model = YOLO(str(self.model_path))
                raw_original = predict_image(self.model, source, imgsz, confidence, self.device, tiled)
                original_view = annotate(source, raw_original, self.model.names)
                raw_distorted = predict_image(self.model, distorted, imgsz, confidence, self.device, tiled)
                raw_processed = predict_image(self.model, processed, imgsz, confidence, self.device, tiled)
                distorted_view = annotate(distorted, raw_distorted, self.model.names)
                processed_view = annotate(processed, raw_processed, self.model.names)
                payload = {"distorted": distorted, "processed": processed,
                           "original_view": original_view, "distorted_view": distorted_view,
                           "processed_view": processed_view,
                           "family": family, "family_label": family_label,
                           "level": level, "method": method, "input_mode": input_mode,
                           "device": self.device, "seed": seed, "imgsz": imgsz,
                           "confidence": confidence, "tiled": tiled}
                self.root.after(0, lambda: self._pipeline_done(payload))
            except Exception as exc:
                self.root.after(0, lambda err=exc: self._pipeline_error(err))

        threading.Thread(target=worker, daemon=True).start()

    def _pipeline_done(self, payload):
        self.last_result = payload
        self._show_image("original", payload["original_view"].image)
        self._show_image("distorted", payload["distorted_view"].image)
        self._show_image("processed", payload["processed_view"].image)
        baseline = payload["original_view"]
        raw_view = payload["distorted_view"]
        result = payload["processed_view"]
        baseline_label = "Detection trên ảnh đầu vào" if payload["input_mode"] == "Ảnh đầu vào đã nhiễu" else "Baseline ảnh sạch"
        self.summary_var.set(
            f"{baseline_label}: {baseline.count} vật thể | Trước recovery: {raw_view.count} | "
            f"Sau recovery: {result.count} vật thể | "
            f"Confidence trung bình: {result.average_confidence:.2f}\n"
            f"Class trước recovery: {', '.join(f'{name} ×{count}' for name, count in sorted(raw_view.classes.items())) or 'không có'}\n"
            f"Class sau recovery: {', '.join(f'{name} ×{count}' for name, count in sorted(result.classes.items())) or 'không có'}\n"
            f"Điều kiện: {payload['family_label']} {payload['level']:g} • "
            f"Recovery: {payload['method']} • imgsz: {payload['imgsz']} • "
            f"IoU NMS: 0.70 • conf: {payload['confidence']:.2f} • "
            f"tiling 640: {'bật' if payload['tiled'] else 'tắt'}")
        device_label = "GPU CUDA" if payload["device"] == "0" else "CPU"
        self.status_var.set(f"Hoàn tất • {device_label}")
        self.run_button.configure(state="normal")
        self.open_button.configure(state="normal")
        self.family_box.configure(state="readonly")
        self.level_box.configure(state="disabled" if FAMILIES[self.family_var.get()] == "clean" else "readonly")
        self.method_box.configure(state="readonly")
        self.mode_box.configure(state="readonly")
        self.imgsz_box.configure(state="readonly")
        self.tile_box.configure(state="normal")
        self.conf_box.configure(state="readonly")
        self.save_button.configure(state="normal")

    def _pipeline_error(self, error):
        self.status_var.set("Có lỗi khi xử lý")
        self.run_button.configure(state="normal")
        self.open_button.configure(state="normal")
        self.family_box.configure(state="readonly")
        self.level_box.configure(state="disabled" if FAMILIES[self.family_var.get()] == "clean" else "readonly")
        self.method_box.configure(state="readonly")
        self.mode_box.configure(state="readonly")
        self.imgsz_box.configure(state="readonly")
        self.tile_box.configure(state="normal")
        self.conf_box.configure(state="readonly")
        messagebox.showerror("Không xử lý được ảnh", str(error))

    def save_results(self):
        if self.last_result is None or self.source_path is None:
            return
        output = filedialog.asksaveasfilename(
            title="Lưu ảnh phát hiện",
            initialfile=f"{self.source_path.stem}_robustvision.jpg",
            defaultextension=".jpg", filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")])
        if not output:
            return
        target = Path(output)
        processed_path = target.with_name(f"{target.stem}_processed{target.suffix}")
        raw_path = target.with_name(f"{target.stem}_distorted{target.suffix}")
        payload = self.last_result
        processed_image = payload["processed_view"].image
        baseline_path = target.with_name(f"{target.stem}_baseline{target.suffix}")
        for path, image in ((target, processed_image),
                            (baseline_path, payload["original_view"].image),
                            (processed_path, payload["processed"]),
                            (raw_path, payload["distorted"])):
            if not cv2.imwrite(str(path), image):
                messagebox.showerror("Lỗi lưu file", f"Không thể lưu {path}")
                return
        result_view = payload["processed_view"]
        raw_view = payload["distorted_view"]
        metadata = {
            "source_image": str(self.source_path), "saved_detection_image": str(target),
            "saved_baseline_detection_image": str(baseline_path),
            "saved_processed_image": str(processed_path), "saved_distorted_image": str(raw_path),
            "model": str(self.model_path), "family": payload["family"], "level": payload["level"],
            "input_mode": payload["input_mode"],
            "recovery_method": payload["method"], "confidence_threshold": payload["confidence"],
            "nms_iou": 0.7, "imgsz": payload["imgsz"],
            "tiled_inference": payload["tiled"], "tile_size": 640 if payload["tiled"] else None,
            "device": payload["device"], "noise_seed": payload["seed"],
            "raw_detection_count": raw_view.count, "raw_class_counts": raw_view.classes,
            "detection_count": result_view.count, "class_counts": result_view.classes,
            "average_confidence": result_view.average_confidence,
            "saved_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        }
        metadata_path = target.with_suffix(".json")
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        messagebox.showinfo("Đã lưu", f"Ảnh baseline, ảnh phát hiện, ảnh xử lý và metadata đã lưu cạnh:\n{target}")


def main():
    root = tk.Tk()
    RobustVisionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
