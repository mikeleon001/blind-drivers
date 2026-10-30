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
                 inlier_tol=0.12, width_tol=0.6, near_fit_max=16.0,
                 track_gate=0.5, track_max_gap=10, track_confirm=10,
                 lane_model=True, model_xmax=15.0):
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
        self.near_fit_max = near_fit_max   # offset/heading come from the line nearer than this (m)
        self.track_gate, self.track_max_gap, self.track_confirm = track_gate, track_max_gap, track_confirm
        self.lane_model, self.model_xmax = lane_model, model_xmax
        self._m_prev = None                              # last (a, b) of the lane-model fit
        self._support = (0, 0)
        self._conf = 0                                   # consecutive steady frames seen
        self._track, self._since_track = None, 0        # last (left, right) edge positions at the car
        self.width_tol = width_tol      # a chosen pair must be within this of the lane width seen so far

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
                # Offset and heading are about the road NEXT to the car. A curve fitted over the
                # whole 5-25 m and then extended back to the car lets the bend far ahead push the
                # near position by 0.4 m or more, so use a straight fit of the near part only.
                near = x[inl] <= self.near_fit_max
                span = np.ptp(x[inl][near]) if near.any() else 0.0
                if near.sum() >= 30 and span >= 4.0:
                    c1 = np.polyfit(x[inl][near], y[inl][near], 2 if span >= 10.0 else 1)
                    coeffs = np.concatenate([np.zeros(self.deg + 1 - len(c1)), c1])
                lines.append((coeffs, x[inl], y[inl]))
            x, y = x[~inl], y[~inl]
            if not inl.any():
                break
        return lines

    def _model_lines(self, x, y):
        """Fit the whole lane at once: centre c(x) = a + b x + k x^2, edges c -/+ W/2, sharing one
        shape. A solid line on one side supports a dashed (or missing) line on the other, and a
        far-away junk line cannot win by itself. Near field only (x <= model_xmax): offset and
        heading are about the road next to the car, and a bend far ahead must not bend the fit.
        Returns [left_edge_poly, right_edge_poly] (np.polyval order) or []."""
        sel = x <= self.model_xmax
        xs, ys = x[sel], y[sel]
        if len(xs) > 1500:                                    # plenty to vote with; keeps it fast
            st = len(xs) // 1500 + 1
            xs, ys = xs[::st], ys[::st]
        if len(xs) < self.min_pixels:
            self._since_track += 1
            return []
        a_grid = np.arange(-2.0, 2.0 + 1e-9, 0.1)
        tracking = (self._m_prev is not None and self._since_track <= self.track_max_gap
                    and self._conf >= self.track_confirm)
        if tracking:
            a_grid = a_grid[np.abs(a_grid - self._m_prev[0]) <= self.track_gate]
            if len(a_grid) == 0:
                self._since_track += 1
                return []
        W0 = self.width_est
        pa_ref, pa_pen = (self._m_prev[0], 150.0) if tracking else (0.0, 0.0)   # while following, prefer answers near the last one

        def search(a_grid, bs, ks, bw, dWs=(0.0, -0.3, 0.3)):
            """Best (score, (a, b, k, W)). Edges are parallel curves a normal distance W/2 from the
            centre c(x) = a + b x + k x^2, so sideways they sit (W/2)*sqrt(1 + c'(x)^2) away: wider
            than W/2 where the road is steep in the image. Each paint point votes for the centre
            offset 'a' it would imply as a left edge and as a right edge; the best 'a' wins."""
            off0 = 12.0
            nb = int(2 * off0 / bw) + 1
            best = (0, None)
            allc = []
            for b in bs:
                for k in ks:
                    r = ys - (b * xs + k * xs * xs)
                    g = np.sqrt(1.0 + (b + 2.0 * k * xs) ** 2)
                    for dW in dWs:
                        Wc = W0 + dW
                        hl = np.bincount(np.clip(((r + 0.5 * Wc * g + off0) / bw).astype(int), 0, nb - 1), minlength=nb)
                        hr = np.bincount(np.clip(((r - 0.5 * Wc * g + off0) / bw).astype(int), 0, nb - 1), minlength=nb)
                        hs = hl[:-1] + hl[1:] + hr[:-1] + hr[1:]
                        ia = np.clip(np.round((a_grid + off0) / bw - 0.5).astype(int), 0, nb - 2)
                        sc = hs[ia] - (25 if dW else 0) - 50.0 * abs(b) - 200.0 * abs(k) - pa_pen * np.abs(a_grid - pa_ref)
                        j = int(sc.argmax())
                        if sc[j] > best[0]:
                            best = (int(sc[j]), (a_grid[j], b, k, Wc))
                        allc.append((int(sc[j]), (a_grid[j], b, k, Wc)))
            search.allc = allc
            return best

        # Stage 1: coarse over every shape up to a tight bend (k = 1/(2R): 0.04 is R = 12 m);
        # stage 2: fine around the winner. The least-squares step below then polishes it.
        search(a_grid, np.arange(-0.4, 0.4001, 0.04), np.arange(-0.04, 0.0401, 0.008), 0.1, (0.0,))
        cands = sorted(search.allc, key=lambda t: -t[0])[:2]
        best = (0, None)
        for _, (_, b0, k0, _) in cands:
            r = search(a_grid, np.arange(b0 - 0.04, b0 + 0.0401, 0.01),
                       np.arange(k0 - 0.008, k0 + 0.0081, 0.002), 0.1)
            if r[0] > best[0]:
                best = r
        if best[1] is None or best[0] < self.min_pixels:
            self._since_track += 1
            return []
        a, b, k, Wc = best[1]
        a_grid_best = a
        for _ in range(3):                                   # least-squares refinement on inliers
            c = a + b * xs + k * xs * xs
            g = np.sqrt(1.0 + (b + 2.0 * k * xs) ** 2)
            inl_l = np.abs(ys - (c - 0.5 * Wc * g)) < 0.2
            inl_r = np.abs(ys - (c + 0.5 * Wc * g)) < 0.2
            X = np.concatenate([xs[inl_l], xs[inl_r]])
            if len(X) < self.min_pixels or np.ptp(X) < 5.0:
                self._since_track += 1
                return []
            Y = np.concatenate([ys[inl_l] + 0.5 * Wc * g[inl_l], ys[inl_r] - 0.5 * Wc * g[inl_r]])
            if np.ptp(X) >= 9.0:
                a, b, k = np.polyfit(X, Y, 2)[::-1]
            else:                                            # short stretch seen: keep the curvature found by the search
                b, a = np.polyfit(X, Y - k * X * X, 1)
            if abs(a - a_grid_best) > 1.0:                    # refinement must not wander to another lane
                self._since_track += 1
                return []
        nl, nr = int(inl_l.sum()), int(inl_r.sum())
        if nl >= 150 and nr >= 150:                          # both edges well seen: learn the width
            Xa = np.concatenate([xs[inl_l], xs[inl_r]])
            ga = np.concatenate([g[inl_l], g[inl_r]])
            A = np.column_stack([np.ones(nl + nr), Xa, Xa ** 2,
                                 ga * np.concatenate([-0.5 * np.ones(nl), 0.5 * np.ones(nr)])])
            yy = np.concatenate([ys[inl_l], ys[inl_r]])
            sol = np.linalg.lstsq(A, yy, rcond=None)[0]
            if 2.8 <= sol[3] <= 4.4:
                self.width_est = 0.9 * self.width_est + 0.1 * float(sol[3])
                a, b, k, Wc = sol[0], sol[1], sol[2], float(sol[3])
        if abs(a) > self.max_abs_offset or abs(b) > 0.6:     # not a believable position/heading: do not remember it
            self._since_track += 1
            return []
        if tracking and abs(a - self._m_prev[0]) > 1.5 * self.track_gate:
            self._since_track += 1                           # a followed lane cannot jump: reject
            return []
        steady = (self._m_prev is not None and self._since_track <= self.track_max_gap
                  and abs(a - self._m_prev[0]) <= (1.5 * self.track_gate if tracking else self.track_gate))
        self._conf = self._conf + 1 if steady else 1
        self._m_prev, self._since_track = (a, b), 0
        self._support = (nl, nr)
        return [np.array([k, b, a - Wc / 2]), np.array([k, b, a + Wc / 2])]

    def reset_track(self):
        """Forget the lane edges seen so far (use between unrelated images or after a teleport)."""
        self._track, self._since_track, self._conf = None, 0, 0
        self._m_prev = None

    def _accept(self, edges, tracked):
        """Remember the lane edges. A track is only FOLLOWED after track_confirm steady frames, so a
        bad first reading (car just spawned, junction) cannot lock the detector onto a wrong lane."""
        t = self._track
        steady = (t is not None and self._since_track <= self.track_max_gap
                  and abs(edges[0] - t[0]) <= self.track_gate and abs(edges[1] - t[1]) <= self.track_gate)
        self._conf = self._conf + 1 if (tracked or steady) else 1
        self._track, self._since_track = edges, 0

    def _choose(self, lines):
        """Pick the ego lane's markings from the candidate lines.

        Tracking: the car cannot move sideways faster than a few cm per frame, so once the lane
        edges are known (where they cut the car's own position), the next frame's edges must be
        near them. Only lines within track_gate of the previous left / right edge are considered;
        this stops a kerb, a shadow edge or a far dashed line from stealing the choice in a bend.
        Without a recent track (start, or lost for a while) fall back to the nearest valid pair."""
        cand = [(np.polyval(c, 0.0), c, len(ix), float(np.ptp(ix)), float(ix.min()), float(ix.max()))
                for c, ix, _ in lines]
        cand = [t for t in cand if t[3] >= 6.0]              # must span at least 6 m of road
        lefts = [t for t in cand if t[0] < -0.3]
        rights = [t for t in cand if t[0] > 0.3]

        if (self._track is not None and self._since_track <= self.track_max_gap
                and self._conf >= self.track_confirm):
            pl, pr = self._track
            g = self.track_gate
            def near(group, pred):
                ok = [t for t in group if abs(t[0] - pred) <= g]
                # strongest support wins; distance only breaks ties
                return max(ok, key=lambda t: t[2] / (1.0 + ((t[0] - pred) / 0.3) ** 2)) if ok else None
            L, R = near(lefts, pl), near(rights, pr)
            if L is not None and R is not None and abs((R[0] - L[0]) - self.width_est) <= 2 * self.width_tol:
                self._accept((L[0], R[0]), True)
                return [L[1], R[1]]
            if R is not None and (L is None or R[2] >= 2 * L[2]) and R[3] >= 8.0:
                self._accept((R[0] - self.width_est, R[0]), True)
                return [R[1]]
            if L is not None and L[3] >= 8.0:
                self._accept((L[0], L[0] + self.width_est), True)
                return [L[1]]
            self._since_track += 1
            return []

        best, best_w = None, 1e9
        for l in lefts:
            for r in rights:
                width = r[0] - l[0]
                slope_diff = abs(np.polyval(np.polyder(r[1]), 0.0) - np.polyval(np.polyder(l[1]), 0.0))
                if abs(width - self.width_est) <= self.width_tol and slope_diff < 0.12 and width < best_w:
                    best, best_w = (l, r), width
        if best is not None:
            self._accept((best[0][0], best[1][0]), False)
            return [best[0][1], best[1][1]]
        reach = self.width_est / 2.0 + self.max_abs_offset
        singles = [t for t in cand if t[3] >= 8.0 and abs(t[0]) <= reach
                   and abs(np.polyval(np.polyder(t[1]), 0.0)) <= 0.2]
        if singles:
            t = max(singles, key=lambda t: t[2])
            w2 = self.width_est / 2.0
            self._accept((t[0], t[0] + self.width_est) if t[0] < 0 else (t[0] - self.width_est, t[0]), False)
            return [t[1]]
        self._since_track += 1
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

        if self.lane_model:
            lines = []                                        # no per-line candidates in this mode
            chosen = self._model_lines(x, y)
        else:
            lines = self._find_lines(x, y)
            chosen = self._choose(lines)

        res = LaneResult(n_lines=len(chosen))
        if self.lane_model and chosen:
            res.n_lines = 1 if min(self._support) < 60 else 2
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
                if (not self.lane_model) and len(chosen) == 2 and 2.0 <= width <= 5.0:   # remember the measured lane width
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
            x_draw = min(self.x_far, self.model_xmax) if self.lane_model else self.x_far
            for coeffs in chosen:                              # the ones actually used, red
                for xm in np.linspace(self.x_near, x_draw, 40):
                    cv2.circle(dbg, (int((np.polyval(coeffs, xm) + self.y_half) / self.res),
                                     int((self.x_far - xm) / self.res)), 2, (0, 0, 255), -1)
            cv2.line(dbg, (self.bev_w // 2, 0), (self.bev_w // 2, self.bev_h), (255, 255, 0), 1)
            txt = (f"off={res.lateral_offset:+.2f}m hdg={math.degrees(res.heading_error):+.1f}deg "
                   f"lines={res.n_lines}")
            cv2.putText(dbg, txt, (5, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            res.debug = {"bev": dbg, "mask": mask}
        return res
