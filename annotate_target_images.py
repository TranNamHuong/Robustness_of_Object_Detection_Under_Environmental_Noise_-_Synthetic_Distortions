"""Review and edit YOLO boxes/classes for target-domain images."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
CANVAS_W, CANVAS_H = 1120, 760


@dataclass
class Box:
    class_id: int
    xc: float
    yc: float
    width: float
    height: float


def read_names(yaml_path: Path) -> dict[int, str]:
    data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw = data.get("names")
    if isinstance(raw, list):
        return dict(enumerate(map(str, raw)))
    return {int(key): str(value) for key, value in raw.items()}


def load_boxes(path: Path) -> list[Box]:
    if not path.is_file():
        return []
    boxes = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Invalid YOLO label at {path}:{line_no}; expected 5 fields")
        class_id = int(fields[0])
        xc, yc, width, height = map(float, fields[1:])
        boxes.append(Box(class_id, xc, yc, width, height))
    return boxes


class LabelEditor:
    def __init__(self, root: tk.Tk, image_dir: Path, draft_dir: Path, output_dir: Path,
                 names: dict[int, str]):
        self.root = root
        self.image_dir = image_dir.resolve()
        self.draft_dir = draft_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.names = names
        self.images = sorted(p for p in self.image_dir.rglob("*")
                             if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
        if not self.images:
            raise ValueError(f"No images found in {self.image_dir}")
        self.index = 0
        self.boxes: list[Box] = []
        self.selected: int | None = None
        self.dirty = False
        self.add_mode = False
        self.drag_start: tuple[int, int] | None = None
        self.temp_rect: int | None = None
        self.image = None
        self.photo = None
        self.scale = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._build_ui()
        self.load_current()

    def _build_ui(self):
        self.root.title("Gán nhãn ảnh mục tiêu • YOLO")
        self.root.geometry("1450x920")
        toolbar = ttk.Frame(self.root, padding=8)
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="◀ Ảnh trước", command=lambda: self.move(-1)).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Ảnh sau ▶", command=lambda: self.move(1)).pack(side="left", padx=3)
        self.add_button = ttk.Button(toolbar, text="Thêm box: tắt", command=self.toggle_add)
        self.add_button.pack(side="left", padx=(18, 3))
        ttk.Button(toolbar, text="Xóa box", command=self.delete_selected).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Đổi lớp", command=self.change_class).pack(side="left", padx=3)
        ttk.Button(toolbar, text="Lưu nhãn đã duyệt", command=self.save).pack(side="left", padx=(18, 3))
        self.status = tk.StringVar()
        ttk.Label(toolbar, textvariable=self.status).pack(side="right", padx=8)

        body = ttk.Frame(self.root, padding=(8, 0, 8, 8))
        body.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(body, width=CANVAS_W, height=CANVAS_H, bg="#202733",
                                highlightthickness=0)
        self.canvas.pack(side="left", fill="both", expand=True)
        panel = ttk.Frame(body, width=260, padding=8)
        panel.pack(side="right", fill="y")
        ttk.Label(panel, text="Lớp cho box mới/đang chọn").pack(anchor="w")
        self.class_var = tk.StringVar()
        class_values = [f"{i}: {name}" for i, name in sorted(self.names.items())]
        self.class_box = ttk.Combobox(panel, textvariable=self.class_var,
                                      values=class_values, state="readonly", width=28)
        self.class_box.pack(fill="x", pady=(4, 12))
        if class_values:
            self.class_box.current(0)
        ttk.Label(panel, text="Boxes trong ảnh").pack(anchor="w")
        self.box_list = tk.Listbox(panel, width=34, height=28, exportselection=False)
        self.box_list.pack(fill="both", expand=True, pady=4)
        ttk.Label(panel, text="Chọn box để đổi lớp/xóa. Bật 'Thêm box' rồi kéo chuột để vẽ.\n\nKiểm tra từng đối tượng; nhãn tự sinh chỉ là gợi ý.",
                  wraplength=250, justify="left").pack(anchor="w", pady=8)
        self.canvas.bind("<Button-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.box_list.bind("<<ListboxSelect>>", self.on_list_select)
        self.root.bind("<Control-s>", lambda _event: self.save())
        self.root.bind("<Left>", lambda _event: self.move(-1))
        self.root.bind("<Right>", lambda _event: self.move(1))

    def _paths(self, image_path: Path) -> tuple[Path, Path]:
        relative = image_path.relative_to(self.image_dir)
        return (self.draft_dir / relative).with_suffix(".txt"), (self.output_dir / relative).with_suffix(".txt")

    def load_current(self):
        path = self.images[self.index]
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        self.image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        draft_path, final_path = self._paths(path)
        self.boxes = load_boxes(final_path if final_path.is_file() else draft_path)
        self.selected = None
        self.dirty = False
        self._draw()
        self._refresh_list()
        self.status.set(f"{self.index + 1}/{len(self.images)} • {path.name}" + (" • chưa lưu" if self.dirty else ""))

    def _draw(self):
        self.canvas.delete("all")
        h, w = self.image.shape[:2]
        self.scale = min((CANVAS_W - 24) / w, (CANVAS_H - 24) / h)
        display = Image.fromarray(self.image).resize((int(w * self.scale), int(h * self.scale)), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(display)
        self.offset_x = (CANVAS_W - display.width) // 2
        self.offset_y = (CANVAS_H - display.height) // 2
        self.canvas.create_image(self.offset_x, self.offset_y, image=self.photo, anchor="nw")
        for i, box in enumerate(self.boxes):
            x1 = self.offset_x + (box.xc - box.width / 2) * w * self.scale
            y1 = self.offset_y + (box.yc - box.height / 2) * h * self.scale
            x2 = self.offset_x + (box.xc + box.width / 2) * w * self.scale
            y2 = self.offset_y + (box.yc + box.height / 2) * h * self.scale
            color = "#27d17f" if i == self.selected else "#ff9f43"
            name = self.names.get(box.class_id, f"class {box.class_id}")
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=color, width=3 if i == self.selected else 2)
            self.canvas.create_text(x1 + 3, max(10, y1 - 8), text=f"{name} [{box.class_id}]",
                                    fill="white", anchor="sw", font=("Segoe UI", 10, "bold"))

    def _refresh_list(self):
        self.box_list.delete(0, tk.END)
        for i, box in enumerate(self.boxes):
            name = self.names.get(box.class_id, f"class {box.class_id}")
            self.box_list.insert(tk.END, f"{i + 1}. {name} (ID {box.class_id}) • {box.width:.2f}×{box.height:.2f}")
        if self.selected is not None and self.selected < len(self.boxes):
            self.box_list.selection_set(self.selected)
            self.box_list.see(self.selected)
            box = self.boxes[self.selected]
            self.class_var.set(f"{box.class_id}: {self.names.get(box.class_id, 'unknown')}")

    def on_list_select(self, _event=None):
        selected = self.box_list.curselection()
        self.selected = selected[0] if selected else None
        self._draw()

    def toggle_add(self):
        self.add_mode = not self.add_mode
        self.add_button.configure(text="Thêm box: bật" if self.add_mode else "Thêm box: tắt")
        self.canvas.configure(cursor="crosshair" if self.add_mode else "")

    def _image_point(self, x: int, y: int) -> tuple[float, float]:
        h, w = self.image.shape[:2]
        px = min(max((x - self.offset_x) / self.scale, 0), w)
        py = min(max((y - self.offset_y) / self.scale, 0), h)
        return px / w, py / h

    def on_press(self, event):
        if self.add_mode:
            self.drag_start = (event.x, event.y)
            self.temp_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y,
                                                          outline="#27d17f", width=3, dash=(5, 3))
            return
        h, w = self.image.shape[:2]
        nx, ny = self._image_point(event.x, event.y)
        candidates = []
        for i, box in enumerate(self.boxes):
            if abs(nx - box.xc) <= box.width / 2 and abs(ny - box.yc) <= box.height / 2:
                candidates.append((box.width * box.height, i))
        self.selected = min(candidates)[1] if candidates else None
        self._draw()
        self._refresh_list()

    def on_drag(self, event):
        if self.add_mode and self.drag_start and self.temp_rect:
            self.canvas.coords(self.temp_rect, self.drag_start[0], self.drag_start[1], event.x, event.y)

    def on_release(self, event):
        if not self.add_mode or not self.drag_start:
            return
        x1, y1 = self._image_point(*self.drag_start)
        x2, y2 = self._image_point(event.x, event.y)
        class_text = self.class_var.get().split(":", 1)[0]
        if abs(x2 - x1) > 0.005 and abs(y2 - y1) > 0.005:
            self.boxes.append(Box(int(class_text), (x1 + x2) / 2, (y1 + y2) / 2,
                                  abs(x2 - x1), abs(y2 - y1)))
            self.selected = len(self.boxes) - 1
            self.dirty = True
        self.drag_start = None
        self.temp_rect = None
        self._draw()
        self._refresh_list()
        self._update_status()

    def change_class(self):
        if self.selected is None:
            messagebox.showinfo("Chọn box", "Chọn một box trong danh sách hoặc trên ảnh trước.")
            return
        try:
            class_id = int(self.class_var.get().split(":", 1)[0])
        except ValueError:
            return
        self.boxes[self.selected].class_id = class_id
        self.dirty = True
        self._draw()
        self._refresh_list()
        self._update_status()

    def delete_selected(self):
        if self.selected is None:
            return
        del self.boxes[self.selected]
        self.selected = None
        self.dirty = True
        self._draw()
        self._refresh_list()
        self._update_status()

    def save(self):
        image_path = self.images[self.index]
        _, output_path = self._paths(image_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [f"{box.class_id} {box.xc:.6f} {box.yc:.6f} {box.width:.6f} {box.height:.6f}" for box in self.boxes]
        output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.dirty = False
        self._update_status()

    def _update_status(self):
        self.status.set(f"{self.index + 1}/{len(self.images)} • {self.images[self.index].name}" + (" • chưa lưu" if self.dirty else " • đã lưu"))

    def move(self, delta: int):
        if self.dirty:
            answer = messagebox.askyesnocancel("Nhãn chưa lưu", "Lưu thay đổi trước khi chuyển ảnh?")
            if answer is None:
                return
            if answer:
                self.save()
        self.index = (self.index + delta) % len(self.images)
        self.load_current()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, default=Path("target_eval/images/val"))
    parser.add_argument("--draft-labels", type=Path, default=Path("runs/target_auto_label_drafts_round3/draft_labels"))
    parser.add_argument("--out", type=Path, default=Path("target_eval/labels/val"))
    parser.add_argument("--data", type=Path, default=Path("target_eval/data.yaml"))
    args = parser.parse_args()
    root = tk.Tk()
    try:
        LabelEditor(root, args.images, args.draft_labels, args.out, read_names(args.data))
        root.mainloop()
    except Exception as exc:
        root.destroy()
        raise SystemExit(str(exc))


if __name__ == "__main__":
    main()
