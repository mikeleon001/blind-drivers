import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import Dataset


class TuSimpleDataset(Dataset):
    """TuSimple labelled frames and binary lane masks."""

    def __init__(self, root, image_size=(256, 512), line_width=8):
        self.root = Path(root)
        self.image_size = image_size
        self.line_width = line_width
        self.records = []

        label_files = sorted(self.root.glob("label_data_*.json"))
        if not label_files:
            raise FileNotFoundError(f"No label_data_*.json files found in {self.root}")

        for label_file in label_files:
            with label_file.open() as file:
                self.records.extend(json.loads(line) for line in file if line.strip())

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(self.root / record["raw_file"]).convert("L")
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)

        for lane in record["lanes"]:
            points = [
                (x, y)
                for x, y in zip(lane, record["h_samples"])
                if 0 <= x < image.width and 0 <= y < image.height
            ]
            if len(points) >= 2:
                draw.line(points, fill=1, width=self.line_width)

        height, width = self.image_size
        image = image.resize((width, height), Image.Resampling.BILINEAR)
        mask = mask.resize((width, height), Image.Resampling.NEAREST)

        image = torch.from_numpy(np.array(image, dtype=np.float32)).unsqueeze(0) / 255.0
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))
        return image, mask
