"""
perception/synthetic_test.py

Geometry check for ClassicalLaneDetector without CARLA.

Renders what the CARLA front camera (800x600, fov 90, 2.4 m high, no pitch) would see for a
road with painted markings, at KNOWN lateral offset / heading error, then checks that the
detector recovers those numbers. This validates the projection, the sign conventions and the
fitting code. It does NOT validate colour thresholds on real CARLA frames; that needs a
real run (see --save-debug in simulation/run_lane_keeping.py).

Run from the repo root:  python perception/synthetic_test.py
"""
import math
import os
import sys

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from perception.classical_lane import ClassicalLaneDetector   # noqa: E402

W, H, FOV, CAM_H = 800, 600, 90.0, 2.4


def render(offset, heading, curvature=0.0, lane_w=3.5, left=True, right=True,
           dashed_right=False, noise=6.0, seed=0):
    """Frame for a vehicle `offset` m right of lane center, yawed `heading` rad right of the lane.
    Lane center line: r_c(l) = 0.5 * curvature * l^2 (l = distance along the lane)."""
    f = W / (2 * math.tan(math.radians(FOV) / 2))
    cx, cy = W / 2, H / 2
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    img = np.zeros((H, W, 3), np.uint8)
    img[:] = (170, 160, 150)                               # sky (BGR)
    ground = v > cy + 1
    vv = np.where(ground, v - cy, 1.0)
    X = f * CAM_H / vv                                     # forward (camera frame)
    Y = (u - cx) * X / f                                   # right
    l = X * math.cos(heading) - Y * math.sin(heading)      # along the lane
    r = offset + X * math.sin(heading) + Y * math.cos(heading)
    rel = r - 0.5 * curvature * l ** 2                     # lateral position relative to center
    asphalt = np.full((H, W, 3), 85, np.uint8)
    paint = np.zeros((H, W), bool)
    half = 0.075
    if left:
        paint |= np.abs(rel + lane_w / 2) < half
    if right:
        p = np.abs(rel - lane_w / 2) < half
        if dashed_right:
            p &= (np.floor(l / 3.0) % 2 == 0)
        paint |= p
    asphalt[paint] = (235, 235, 235)
    img[ground] = asphalt[ground]
    rng = np.random.default_rng(seed)
    img = np.clip(img.astype(np.float32) + rng.normal(0, noise, img.shape), 0, 255).astype(np.uint8)
    return img


def check(name, det, frame, true_off, true_hdg, tol_off=0.06, tol_hdg=0.03, expect_lines=None):
    det.reset_track()          # the cases are unrelated images, not consecutive frames
    det.width_est = 3.5        # ...so the lane width learned on an earlier case does not carry over
    r = det.detect(frame)
    err_o = r.lateral_offset - true_off
    err_h = r.heading_error - true_hdg
    ok = (r.detected and abs(err_o) <= tol_off and abs(err_h) <= tol_hdg
          and (expect_lines is None or r.n_lines == expect_lines))
    print(f"{'PASS' if ok else 'FAIL'}  {name:34s} off true {true_off:+.2f} got {r.lateral_offset:+.3f} "
          f"| hdg true {math.degrees(true_hdg):+5.1f} got {math.degrees(r.heading_error):+5.1f} deg "
          f"| lines {r.n_lines} width {r.lane_width:.2f}")
    return ok


def main():
    det = ClassicalLaneDetector(W, H, FOV, CAM_H)
    results = []
    for off in (0.0, 0.5, -0.5, 0.9, -0.9):
        results.append(check(f"straight, offset {off:+.1f}", det, render(off, 0.0), off, 0.0,
                             expect_lines=2))
    for hd in (0.05, -0.05, 0.1):
        results.append(check(f"heading {math.degrees(hd):+.0f} deg", det,
                             render(0.2, hd), 0.2, hd, expect_lines=2))
    for off, hd in ((0.5, 0.2), (-0.5, -0.2), (0.0, 0.3), (0.3, -0.3)):
        results.append(check(f"large heading {math.degrees(hd):+.0f} deg, off {off:+.1f}", det,
                             render(off, hd), off, hd, tol_off=0.1, tol_hdg=0.04))
    results.append(check("gentle curve (R=200 m)", det, render(0.3, 0.0, curvature=0.005), 0.3, 0.0,
                         tol_off=0.08))
    results.append(check("dashed right line", det, render(0.3, 0.0, dashed_right=True), 0.3, 0.0))
    results.append(check("left line only", det, render(0.4, 0.0, right=False), 0.4, 0.0,
                         expect_lines=1))
    results.append(check("right line only", det, render(-0.4, 0.0, left=False), -0.4, 0.0,
                         tol_off=0.08, expect_lines=1))   # one line: width assumed, so 8 cm is fine

    blank = render(0.0, 0.0, left=False, right=False)
    det.reset_track()
    r = det.detect(blank)
    ok = not r.detected
    print(f"{'PASS' if ok else 'FAIL'}  no markings -> detected=False")
    results.append(ok)

    print(f"\n{sum(results)}/{len(results)} passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
