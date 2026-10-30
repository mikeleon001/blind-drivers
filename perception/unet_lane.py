"""
perception/unet_lane.py

Lane detector that uses the trained U-Net (perception/unet/) to find lane paint, then reuses
ClassicalLaneDetector for everything after that:

    BGR frame -> U-Net lane mask (image) -> warp to bird's-eye view (same ground-plane
              homography as the classical detector) -> line voting + curve fit -> pick the
              nearest left/right pair -> lateral offset (m), heading error (rad), lane width (m)

Only the "find the paint" step changes. The bird's-eye view only covers the road 5-25 m ahead,
so sky and building false positives in the upper image are thrown away automatically.

Needs torch (in addition to numpy + opencv). Same conventions and LaneResult as
perception/classical_lane.py.
"""
import importlib.util
import os

import cv2
import numpy as np

from perception.classical_lane import ClassicalLaneDetector, LaneResult  # noqa: F401

UNET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unet")
DEFAULT_WEIGHTS = os.path.join(UNET_DIR, "unet_tusimple.pt")
IN_H, IN_W = 256, 512        # size the model was trained at (perception/unet/train.py defaults)


def _load_model_class():
    """Import perception/unet/model.py without relying on the current directory."""
    spec = importlib.util.spec_from_file_location("unet_model", os.path.join(UNET_DIR, "model.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.UNetSegmentation


class UNetLaneDetector(ClassicalLaneDetector):
    def __init__(self, img_w=800, img_h=600, fov_deg=90.0, cam_height=2.4,
                 weights=None, threshold=0.5, min_piece_m=0.8, device=None, min_pixels=40, **kwargs):
        # min_pixels is lower than the classical detector's: after thinning there is ~1 point per row
        super().__init__(img_w, img_h, fov_deg, cam_height, min_pixels=min_pixels, **kwargs)
        import torch                                   # imported here so the rest works without torch
        self._torch = torch
        self.threshold = threshold
        self.min_piece_px = int(round(min_piece_m / self.res))
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = _load_model_class()().to(self.device)
        self.model.load_state_dict(torch.load(weights or DEFAULT_WEIGHTS, map_location=self.device))
        self.model.eval()
        self._bev_mask = None

    # ---- the only torch-dependent step -----------------------------------------------------
    def _model_mask(self, frame_bgr):
        """Full-size uint8 mask (0 / 255): 255 where the model says "lane paint"."""
        torch = self._torch
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (IN_W, IN_H), interpolation=cv2.INTER_LINEAR)
        x = torch.from_numpy(small.astype(np.float32) / 255.0)[None, None].to(self.device)
        with torch.inference_mode():
            prob = torch.softmax(self.model(x), dim=1)[0, 1].cpu().numpy()
        prob = cv2.resize(prob, (frame_bgr.shape[1], frame_bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
        return (prob >= self.threshold).astype(np.uint8) * 255

    # ---- plumbing: feed the model's mask into the classical pipeline -----------------------
    def detect(self, frame_bgr, want_debug=False):
        mask_img = self._model_mask(frame_bgr)
        bev = cv2.warpPerspective(mask_img, self.M, (self.bev_w, self.bev_h), flags=cv2.INTER_LINEAR)
        mask = (bev > 127).astype(np.uint8) * 255
        # Drop tiny specks: real paint is a stretch of at least ~0.8 m in the bird's-eye view.
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        keep = np.zeros(n, bool)
        for i in range(1, n):
            keep[i] = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]) >= self.min_piece_px
        self._bev_mask = self._centerline(keep[labels])
        return super().detect(frame_bgr, want_debug)

    @staticmethod
    def _centerline(mask_bool):
        """Thin every painted stretch to one pixel per row (its middle). The model draws lines
        several pixels wide; fitting a line to the whole band biases it, the middle does not."""
        out = np.zeros(mask_bool.shape, np.uint8)
        padded = np.zeros((mask_bool.shape[0], mask_bool.shape[1] + 2), np.int8)
        padded[:, 1:-1] = mask_bool
        d = np.diff(padded, axis=1)                       # +1 where a run starts, -1 after it ends
        r_s, c_s = np.nonzero(d == 1)
        r_e, c_e = np.nonzero(d == -1)                    # same order: one start and one end per run
        out[r_s, (c_s + c_e - 1) // 2] = 255
        return out

    def _paint_mask(self, bev_bgr):
        return self._bev_mask
