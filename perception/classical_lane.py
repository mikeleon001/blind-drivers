"""
perception/classical_lane.py

Classical (no-learning) lane detector for the CARLA front camera.

    BGR frame -> bird's-eye view (ground-plane homography) -> paint threshold (white / yellow)
              -> Hough-style line voting + curve refit on the paint pixels (in meters)
              -> pick the left/right pair that brackets the car and looks like one lane
              -> lateral offset (m), heading error (rad), lane width (m)

Because the camera geometry is known (CARLA camera: fov, height, no pitch), image pixels map
straight onto the road plane, so the output is already in METERS. That is what the PID needs.

Lines are classified left/right by where they cross the car (y at x = 0), not by image column,
so it keeps working when the car is yawed and the markings run diagonally in the bird's-eye view.

Conventions (match simulation/run_lane_keeping.py and the CSV logs):
    lateral_offset  > 0 : vehicle is RIGHT of the lane center
    heading_error   > 0 : vehicle points to the RIGHT of the lane direction

Needs only numpy + opencv. Camera assumptions (must match the CARLA sensor you spawn):
    pinhole, pitch = 0, mounted `cam_height` m above the road, centered on the car.
"""
from dataclasses import dataclass, field
import math
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore", message="Polyfit may be poorly conditioned")


@dataclass
class LaneResult:
    lateral_offset: float = 0.0       # m, + = vehicle right of lane center
    heading_error: float = 0.0        # rad, + = vehicle yaw right of lane direction
    lane_width: float = 0.0           # m (measured if both lines found, else the assumed width)
    detected: bool = False
    n_lines: int = 0                  # 0, 1 or 2 markings used
    debug: dict = field(default_factory=dict)


class ClassicalLaneDetector:
    def __init__(self, img_w=800, img_h=600, fov_deg=90.0, cam_height=2.4,
                 x_near=5.0, x_far=25.0, y_half=6.0, res=0.05,
                 assumed_lane_width=3.5, min_pixels=120,
                 paint_contrast=35, paint_kernel_m=0.65,
                 yellow_h=(18, 38), yellow_s_min=90, yellow_v_min=90,
                 fit_degree=2, max_abs_offset=2.5, max_abs_heading=0.6,
                 inlier_tol=0.12):
        self.W, self.H = img_w, img_h
        self.f = img_w / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        self.cx, self.cy = img_w / 2.0, img_h / 2.0
        self.h = cam_height
        self.x_near, self.x_far, self.y_half, self.res = x_near, x_far, y_half, res
        self.lane_w = assumed_lane_width
        self.width_est, self._width_seen = assumed_lane_width, False   # learned from 2-line frames
        self.min_pixels = min_pixels
        self.paint_contrast, self.paint_kernel_m = paint_contrast, paint_kernel_m
        self.yellow_h, self.yellow_s_min, self.yellow_v_min = yellow_h, yellow_s_min, yellow_v_min
        self.deg = fit_degree
        self.max_abs_offset, self.max_abs_heading = max_abs_offset, max_abs_heading
        self.inlier_tol = inlier_tol

        # Bird's-eye image: columns = lateral Y (left -> right), rows = forward X (far -> near).
        self.bev_w = int(round(2 * y_half / res))
        self.bev_h = int(round((x_far - x_near) / res))
        src = np.float32([self.ground_to_pixel(x_far, -y_half), self.ground_to_pixel(x_far, y_half),
                          self.ground_to_pixel(x_near, y_half), self.ground_to_pixel(x_near, -y_half)])
        dst = np.float32([[0, 0], [self.bev_w - 1, 0],
                          [self.bev_w - 1, self.bev_h - 1], [0, self.bev_h - 1]])
        self.M = cv2.getPerspectiveTransform(src, dst)

    # ---- geometry -------------------------------------------------------------------------
    def ground_to_pixel(self, x, y):
        """Ground point (x forward, y right, meters, camera frame) -> image pixel (u, v)."""
        return (self.cx + self.f * y / x, self.cy + self.f * self.h / x)

    def bev_to_ground(self, col, row):
        """BEV pixel -> (x forward, y right) in meters."""
        return self.x_far - row * self.res, -self.y_half + col * self.res

    # ---- detection ------------------------------------------------------------------------
    def _paint_mask(self, bev_bgr):
        """Thin bright stripes (lane paint) found by LOCAL contrast, so paint in shadow (gray ~120
        on ~60 asphalt) is caught while big bright areas (sunlit sidewalk tiles) are not.
        A top-hat filter keeps bright features narrower than its kernel (0.65 m by default)."""
        gray = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2GRAY)
        k = max(3, int(round(self.paint_kernel_m / self.res)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
        stripe = tophat >= self.paint_contrast

        hsv = cv2.cvtColor(bev_bgr, cv2.COLOR_BGR2HSV)
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        yellow = ((h >= self.yellow_h[0]) & (h <= self.yellow_h[1]) &
                  (s >= self.yellow_s_min) & (v >= self.yellow_v_min) & (tophat >= 15))
        mask = (stripe | yellow).astype(np.uint8) * 255

        # Drop specks (asphalt texture): keep connected pieces that are long enough to be paint.
        # Also drop solid blobs (sidewalk tiles, vehicles): paint is thin, so its bounding box is
        # either narrow (<0.3 m) or mostly empty (a diagonal stripe).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=4)
        keep = np.zeros(n, bool)
        blob_px = int(round(0.3 / self.res))
        for i in range(1, n):
            w, hh, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
            long_enough = area >= 20 and max(w, hh) >= int(0.4 / self.res)
            # blob = wide, solid and not elongated (a tile or a car), unlike a long thin stripe
            blobby = (min(w, hh) >= blob_px and area >= 0.45 * w * hh
                      and max(w, hh) < 4 * min(w, hh))
            keep[i] = long_enough and not blobby
        return (keep[labels].astype(np.uint8)) * 255

    def _final_fit(self, x, y):
        """Least-squares fit of y(x). Quadratic only if the points span most of the view
        (extrapolating a curve from a short stretch back to x = 0 is error-prone)."""
        if x.max() - x.min() < 4.0:
            return None
        deg = self.deg if (x.max() - x.min()) > 12.0 else 1
        c = np.polyfit(x, y, deg)
        return np.concatenate([np.zeros(self.deg + 1 - len(c)), c])     # always deg+1 coefficients

    def _find_lines(self, x, y, max_lines=7):
        """Hough-style voting for lines y = a + b x among the paint points (meters), strongest
        first. Deterministic, and sparse (dashed) lines are found because the vote counts
        points, not luck. Each peak is refined with a least-squares fit (quadratic if the line
        spans most of the view) and its points are removed before looking for the next line.
        Returns a list of (coeffs, inlier_x, inlier_y)."""
        lines = []
        slopes = np.linspace(-0.5, 0.5, 51)                  # +-27 deg in 0.02 steps
        bin_w, offset0 = 0.2, 25.0                           # intercept bins, shifted to be >= 0
        for _ in range(max_lines):
            if len(x) < self.min_pixels:
                break
            best_n, best_b, best_a = 0, 0.0, 0.0
            for b in slopes:
                idx = np.floor((y - b * x + offset0) / bin_w).astype(int)
                cnt = np.bincount(idx, minlength=int(2 * offset0 / bin_w) + 2)
                two = cnt[:-1] + cnt[1:]                     # a window two bins (0.4 m) wide
                j = int(two.argmax())
                if two[j] > best_n:
                    best_n, best_b, best_a = int(two[j]), b, (j + 1) * bin_w - offset0
            if best_n < self.min_pixels:
                break
            inl = np.abs(y - (best_a + best_b * x)) < 0.25
            coeffs = None
            for _ in range(2):                               # refine on the curve itself
                c = self._final_fit(x[inl], y[inl]) if inl.sum() >= 3 else None
                if c is None:
                    break
                coeffs = c
                inl = np.abs(y - np.polyval(c, x)) < self.inlier_tol
            if coeffs is not None and inl.sum() >= self.min_pixels:
                lines.append((coeffs, x[inl], y[inl]))
            x, y = x[~inl], y[~inl]
            if not inl.any():
                break
        return lines

    def _choose(self, lines):
        """The ego lane is bounded by the NEAREST marking on each side of the car: take the
        narrowest left/right pair that is a plausible lane (2-6 m wide, roughly parallel).
        Otherwise fall back to the single longest line. Returns 1-2 coeff arrays, or []."""
        # (y at the car, coeffs, n inlier pixels, road length spanned in m)
        cand = [(np.polyval(c, 0.0), c, len(ix), float(np.ptp(ix))) for c, ix, _ in lines]
        cand = [t for t in cand if t[3] >= 6.0]              # must span at least 6 m of road
        lefts = [t for t in cand if t[0] < -0.3]
        rights = [t for t in cand if t[0] > 0.3]
        best, best_w = None, 1e9
        for l in lefts:
            for r in rights:
                width = r[0] - l[0]
                slope_diff = abs(np.polyval(np.polyder(r[1]), 0.0) - np.polyval(np.polyder(l[1]), 0.0))
                if 2.0 <= width <= 6.0 and slope_diff < 0.12 and width < best_w:
                    best, best_w = (l[1], r[1]), width
        if best is not None:
            return list(best)
        singles = [t for t in cand if t[3] >= 8.0]
        if singles:
            return [max(singles, key=lambda t: t[2])[1]]
        return []

    def detect(self, frame_bgr, want_debug=False):
        bev = cv2.warpPerspective(frame_bgr, self.M, (self.bev_w, self.bev_h),
                                  flags=cv2.INTER_LINEAR)
        mask = self._paint_mask(bev)

        rows, cols = np.nonzero(mask)
        if len(rows) > 6000:                                  # keep it fast; every k-th point
            step = int(math.ceil(len(rows) / 6000))
            rows, cols = rows[::step], cols[::step]
        x, y = self.bev_to_ground(cols.astype(np.float64), rows.astype(np.float64))

        lines = self._find_lines(x, y)
        chosen = self._choose(lines)

        res = LaneResult(n_lines=len(chosen))
        if chosen:
            if len(chosen) == 2:
                center = (chosen[0] + chosen[1]) / 2.0
                width = float(np.polyval(chosen[1], 0.0) - np.polyval(chosen[0], 0.0))
            else:
                c = chosen[0]
                left_side = np.polyval(c, 0.0) < 0
                center = c.copy()
                center[-1] += self.width_est / 2.0 * (1 if left_side else -1)
                width = self.width_est
            slope = np.polyval(np.polyder(center), 0.0)
            off = float(-np.polyval(center, 0.0))
            hdg = float(-math.atan(slope))
            # Plausibility gate: a car in its lane is never this far off or this skewed.
            if abs(off) <= self.max_abs_offset and abs(hdg) <= self.max_abs_heading:
                res.lateral_offset, res.heading_error = off, hdg
                res.lane_width = width
                res.detected = True
                if len(chosen) == 2 and 2.0 <= width <= 5.0:   # remember the measured lane width
                    self.width_est = width if not self._width_seen else 0.9 * self.width_est + 0.1 * width
                    self._width_seen = True
            else:
                res.n_lines = 0

        if want_debug:
            dbg = bev.copy()
            dbg[mask > 0] = (255, 0, 255)                      # detected paint = magenta
            for coeffs, ix, iy in lines:                       # every candidate line, thin blue
                for xm in np.linspace(self.x_near, self.x_far, 40):
                    cv2.circle(dbg, (int((np.polyval(coeffs, xm) + self.y_half) / self.res),
                                     int((self.x_far - xm) / self.res)), 1, (255, 128, 0), -1)
            for coeffs in chosen:                              # the ones actually used, red
                for xm in np.linspace(self.x_near, self.x_far, 40):
                    cv2.circle(dbg, (int((np.polyval(coeffs, xm) + self.y_half) / self.res),
                                     int((self.x_far - xm) / self.res)), 2, (0, 0, 255), -1)
            cv2.line(dbg, (self.bev_w // 2, 0), (self.bev_w // 2, self.bev_h), (255, 255, 0), 1)
            txt = (f"off={res.lateral_offset:+.2f}m hdg={math.degrees(res.heading_error):+.1f}deg "
                   f"lines={res.n_lines}")
            cv2.putText(dbg, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            res.debug = {"bev": dbg, "mask": mask}
        return res
