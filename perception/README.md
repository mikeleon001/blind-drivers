# perception/

Lane detection pipeline.

## Files

- `classical_lane.py` - `ClassicalLaneDetector`, a classical (no learning) detector for the CARLA
  front camera. Takes a BGR frame and returns the lateral offset, heading error and lane width
  **in meters / radians**, so the controller can use it directly.
- `synthetic_test.py` - geometry check with no CARLA needed. It renders what the camera would
  see for known offsets/headings (straight, curved, dashed, single line, large heading, no
  markings) and checks that the detector recovers them. Run from the repo root:
  `python perception/synthetic_test.py` (expects `17/17 passed`).

## How the classical detector works

1. **Bird's-eye view.** The CARLA camera geometry is known (field of view, mounting height, no
   pitch), so the road ahead (5-25 m, +-6 m wide) is warped onto a top-down grid with 5 cm
   resolution. Pixels in that grid map straight to meters on the road.
2. **Paint mask.** Lane paint is found by *local contrast* (a top-hat filter keeps thin bright
   stripes) plus a yellow-hue test. This works in shadow, where paint is much darker than in
   sunlight. Small specks and solid blobs (sidewalk tiles, vehicles) are removed.
3. **Lane model.** Instead of picking lines one by one, the detector fits the whole lane: a
   center curve `c(x) = a + b*x + k*x^2` (offset, heading, curvature) with both edges a normal
   distance `W/2` away. Every paint point votes for the offset it would imply as a left or right
   edge; this is searched coarse-to-fine over heading and curvature (up to `k = 0.04`, a ~12 m
   radius bend), then polished with a least-squares fit on the inliers. Dashed lines count
   because the vote counts points. A short visible stretch keeps the curvature found by the
   search instead of being refit as a straight line.
4. **Tracking.** The lane width is learned when both edges are well seen. A track is only
   followed after 10 steady frames (so a bad first reading cannot lock on), while following the
   search prefers answers near the previous one, and an established track survives a moderate
   jump. Readings that jump too far, or are implausible, are dropped.
5. **Output.** Lateral offset and heading error come from the lane center curve at the car's
   position. Implausible results (offset over 2.5 m, heading over 34 degrees) are rejected as
   "not detected". Set `lane_model=False` for the older line-pair method.

```python
from perception.classical_lane import ClassicalLaneDetector

det = ClassicalLaneDetector(img_w=800, img_h=600, fov_deg=90.0, cam_height=2.4)
result = det.detect(frame_bgr, want_debug=True)
# result.lateral_offset  (m, + = vehicle right of lane center)
# result.heading_error   (rad, + = vehicle pointing right of the lane direction)
# result.lane_width, result.detected, result.n_lines, result.debug["bev"]
```

## Assumptions and known limitations

- The camera must match the detector settings: pinhole, no pitch, centered on the car,
  `cam_height` m above the road. If the detected lane width is consistently off compared to the
  map (`width_err_m` in `evaluation/compare_runs.py`), the camera height is the first thing to
  check.
- Tuned on a handful of frames from Town10HD; thresholds (`paint_contrast`, `paint_kernel_m`,
  yellow limits) may need adjusting for other maps and weather.
- It needs visible lane markings, and it assumes both lane edges run parallel. At junctions the
  paint disappears or splits: in Town10HD at ~150 m from spawn 10 the right white line peels
  away into a turn pocket and pulls the fit sideways (the 8 m/s run fails there). Shortening the
  fit range or re-weighting the edges did not fix it; it needs a check that the two edges stay
  parallel (and a way to ignore the one that diverges).
- The offset can read up to ~0.4-0.6 m off for a few frames in the bend, where only the far end
  of the lane is visible; heading is then noisy (a few degrees).
- Results on other spawns, maps and weather have not been measured yet.
- Speed: ~35-45 ms per frame on the lab CPU (U-Net: ~80 ms).

## Plan

- Classical computer vision approach: done as described above, tested on Town10HD spawn 10.
- CNN-based approach: `unet_lane.py` (U-Net lane mask from `unet/`, same bird's-eye fit) plugs in
  with `--detector unet`. A new detector only has to return the same four values (offset,
  heading error, lane width, detected) to plug into `simulation/run_lane_keeping.py` and the
  evaluation tools.
