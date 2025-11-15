from datasets import load_dataset
import numpy as np
from PIL import Image, ImageDraw
from typing import Union, Tuple
from typing import List, Optional
from .constants import label_names

class QuickDrawDataset:
    def __init__(
        self,
        split: str = "train",
        image_size: Tuple[int, int] = (256, 256),
        cache_dir: str = "/research/mayukh/huggingface_cache_dir",
        custom_class_names: Optional[List[str]] = None
    ):
        self.dataset = load_dataset(
            "darknoon/quickdraw",
            split=split,
            cache_dir=cache_dir
        )
        self.image_size = image_size

        self.custom_class_names = set(custom_class_names) if custom_class_names else None

        # --- FAST FILTERING: compute valid indices in a single pass ---
        if self.custom_class_names:
            # convert label names -> original integer ids
            allowed_ids = {
                k for k, v in label_names.items() if v in self.custom_class_names
            }

            # Extract the entire "word" column once (no Python loop):
            word_column = self.dataset["word"]     # this is a list, very fast

            # Build list of indices to keep:
            valid_indices = [i for i, w in enumerate(word_column) if w in allowed_ids]

            # Use select instead of filter → MUCH FASTER
            self.dataset = self.dataset.select(valid_indices)

            # Build contiguous remapping
            sorted_names = sorted(self.custom_class_names)
            self.new_label_map = {name: i for i, name in enumerate(sorted_names)}
        else:
            self.new_label_map = None

    def __len__(self):
        return len(self.dataset)
    
    def normalize_strokes(self, strokes):
        # Flatten all points
        points = [(x, y) for stroke in strokes for (x, y) in stroke]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        width = max_x - min_x
        height = max_y - min_y

        # avoid division by zero
        if width == 0: width = 1
        if height == 0: height = 1

        # scale to fit the image with a small padding
        target_w, target_h = self.image_size
        scale = min(target_w / width, target_h / height) * 0.9

        # translate + scale strokes
        norm_strokes = []
        for stroke in strokes:
            norm_strokes.append([
                ((x - min_x) * scale, (y - min_y) * scale)
                for (x, y) in stroke
            ])
        return norm_strokes

    def decode_drawing(self, drawing):
        x, y = 0, 0
        strokes = []
        stroke = []

        for dx, dy, end in drawing:
            x += dx
            y += dy
            stroke.append((x, y))
            if end == 1:
                strokes.append(stroke)
                stroke = []

        if stroke:
            strokes.append(stroke)

        return strokes

    def render_strokes(self, strokes):
        img = Image.new("L", self.image_size, color=0)
        draw = ImageDraw.Draw(img)

        for stroke in strokes:
            if len(stroke) > 1:
                draw.line(stroke, fill=255, width=3)

        return img

    def __getitem__(self, idx) -> dict[str, Union[np.ndarray, Image.Image]]:
        item = self.dataset[idx]
        drawing = item["drawing"]

        strokes = self.decode_drawing(drawing)
        strokes = self.normalize_strokes(strokes)
        img = self.render_strokes(strokes)

        orig_name = label_names[item["word"]]

        # map to new contiguous label id if using subset
        if self.new_label_map:
            new_label = self.new_label_map[orig_name]
        else:
            new_label = item["word"]

        return {
            "image": img,
            "name": orig_name,
            "label": new_label
        }
