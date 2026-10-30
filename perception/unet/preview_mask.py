"""Preview what the U-Net sees: run the trained weights on saved camera images and
write an overlay (lane pixels painted red) next to the original.

Needs no CARLA. Save this file in perception/unet/ (next to model.py), then:

    python perception/unet/preview_mask.py <image-or-folder> [--out unet_preview]

A folder is searched for *_cam.png (the debug frames saved by
run_lane_keeping.py --detector cv --save-debug N); if none, for any .png/.jpg.
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import UNetSegmentation  # noqa: E402

IN_H, IN_W = 256, 512  # the size the model was trained at (train.py defaults)


def collect(path):
    p = Path(path)
    if p.is_file():
        return [p]
    cams = sorted(p.glob("*_cam.png"))
    if cams:
        return cams
    return sorted([*p.glob("*.png"), *p.glob("*.jpg")])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="an image, or a folder of images")
    ap.add_argument("--weights", default=str(HERE / "unet_tusimple.pt"))
    ap.add_argument("--out", default="unet_preview")
    ap.add_argument("--max", type=int, default=12, help="at most this many images")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="lane probability needed to paint a pixel (lower = more lane)")
    args = ap.parse_args()

    files = collect(args.source)[: args.max]
    if not files:
        sys.exit(f"no images found in {args.source}")

    model = UNetSegmentation()
    model.load_state_dict(torch.load(args.weights, map_location="cpu"))
    model.eval()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    for f in files:
        bgr = cv2.imread(str(f))
        if bgr is None:
            print(f"skip (cannot read): {f}")
            continue
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None]  # [1,1,H,W]
        with torch.no_grad():
            prob = torch.softmax(model(x), dim=1)[0, 1].numpy()  # P(lane) per pixel
        mask = (prob >= args.threshold).astype(np.uint8)
        mask_full = cv2.resize(mask, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)

        overlay = bgr.copy()
        overlay[mask_full > 0] = (0, 0, 255)  # red = "lane" according to the model
        blend = cv2.addWeighted(bgr, 0.5, overlay, 0.5, 0)
        cv2.imwrite(str(out_dir / f"{f.stem}_unet.png"), np.hstack([bgr, blend]))
        print(f"{f.name}: {100.0 * mask.mean():.1f}% of pixels marked as lane")

    print(f"\nSaved side-by-side images (left = camera, right = model) to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
