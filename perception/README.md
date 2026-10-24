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
3. **Line finding.** A Hough-style vote finds the strongest lines `y = a + b*x`, which are then
   refined with a least-squares fit (quadratic if the line spans most of the view). Dashed lines
   are found because the vote counts points.
4. **Ego lane.** The lane is bounded by the *nearest* marking on each side of the car (a 2-6 m
   wide, roughly parallel pair). If only one marking is visible, the last measured lane width is
   used to place the lane center.
5. **Output.** Lateral offset and heading error come from the lane center line at the car's
   position. Implausible results (offset over 2.5 m, heading over 34 degrees) are rejected as
   "not detected".

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
- It needs visible lane markings. It loses the lane at junctions, where there are none, and
  currently also near sharper curves (under investigation).

## Plan

- Classical computer vision approach: done as described above; being tuned on real CARLA runs.
- CNN-based approach (stretch goal, if time permits): a neural-network-based lane detector for
  more robust performance under complex conditions. A new detector only has to return the same
  four values (offset, heading error, lane width, detected) to plug into
  `simulation/run_lane_keeping.py` and the evaluation tools.
