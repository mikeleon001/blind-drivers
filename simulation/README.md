# simulation/

CARLA closed-loop runner and helper scripts. CARLA itself must already be running
(`CarlaUE4.exe`); run everything from the repo root inside an environment that has the `carla`
package (`conda activate carla-sim312` on the lab machine).

## Files

- `run_lane_keeping.py` - closed loop in **synchronous mode** (fixed 0.05 s step, 20 Hz):
  camera frame -> lane offset -> `control.PIDController` -> steering. Throttle holds a target
  speed. Every frame is logged to a CSV and the run is scored with `evaluation/metrics.py` at
  the end. The simulator is returned to its original settings and all actors are destroyed
  afterwards, even if the run crashes.
- `list_spawns.py` - lists spawn points with at least 60 m of straight road ahead and no
  junction (junctions have no lane markings to follow). `--draw` labels the spawn indices in the
  3D view.

## Usage

```bash
python simulation/list_spawns.py --draw
python simulation/run_lane_keeping.py --spawn 10
python simulation/run_lane_keeping.py --spawn 10 --detector cv --save-debug 20
python simulation/run_lane_keeping.py --spawn 10 --start-offset 0.6
python simulation/run_lane_keeping.py --spawn 10 --kick 0.6 --kick-at 8 --kp 1.8 --tag kp18
```

| Option | Meaning |
|---|---|
| `--spawn N` | spawn point index (default 0, which is at an intersection - use `list_spawns.py`) |
| `--duration S` | run length in seconds (default 30) |
| `--speed V` | target speed in m/s (default 8) |
| `--kp --ki --kd` | PID gains (default 1.2 / 0.3 / 0.1) |
| `--start-offset M` | start M meters right of lane center (negative = left) |
| `--kick A` `--kick-at T` `--kick-dur D` | add steering disturbance A at time T for D seconds |
| `--detector gt\|cv\|unet` | `gt` = lane offset from the CARLA map (ground truth), `cv` = OpenCV detector from `perception/`, `unet` = U-Net lane mask (needs torch) |
| `--unet-weights`, `--unet-threshold` | with `unet`: weights file (default `perception/unet/unet_tusimple.pt`) and the probability needed to count a pixel as paint (0.5) |
| `--kh` | heading-error gain (0 = off); steers back toward the lane direction and damps overshoot, try 0.5 |
| `--smooth` | offset smoothing 0..1 (1 = off, 0.5 = average new and old) |
| `--steer-rate` | max steering change per second (0 = no limit), try 4 |
| `--max-jump` | reject a reading that moves more than this many m in one frame (0 = off), try 0.4 |
| `--bev-half`, `--bev-near` | bird's-eye view half-width (6 m) and nearest distance (5 m) |
| `--save-debug N` | with `cv`/`unet`: save the camera frame and bird's-eye debug image every N frames (the camera image shows the map lane: green = center, orange = edges) |
| `--debug-from S` `--debug-to S` | save debug images (`*_raw.png`, `*_bev.png`) for **every** frame in this time window |
| `--tag NAME` | label put in the CSV file name (use it to tell sweeps apart) |
| `--town NAME` | load another map, e.g. `Town04` |
| `--out DIR` | output folder (default `evaluation/runs/`) |

## Output

`evaluation/runs/run_<tag>_spawn<N>_<timestamp>.csv`, one row per frame, with the standard
`evaluation/logger.py` fields plus `steer`, `speed`, `disturbance`, and the map ground truth
`gt_offset` and `gt_lane_width` (always logged, so a detector's error can be measured), plus
`gt_offset_cam` and `gt_heading` (map offset measured at the camera point, and the car's heading
relative to the map lane; use these, not `gt_offset`, to judge the detector in a bend). The end
of run summary prints the detector's own numbers, the **TRUE deviation from the map lane
center**, and the detector error against the map. Note: after the car crosses the double yellow
the nearest map lane is the oncoming one, so the truth is meaningless once the car is off the
road. With
`--save-debug`, images go to `evaluation/runs/debug_<timestamp>/` (`*_cam.png` = camera,
`*_bev.png` = bird's-eye view with the detected paint in magenta and the chosen lines in red).

## Settings used for the Town10HD bend (spawn 10)

```bash
python simulation/run_lane_keeping.py --spawn 10 --detector cv --duration 30 --speed 4 \
    --kp 0.6 --ki 0.1 --kd 0.1 --kh 0.5 --smooth 0.5 --steer-rate 4
```

4 m/s: true deviation ~0.12 m RMSE, worst ~0.41 m (in the bend), 100% detection, 4 of 4 runs on
the final code. The default gains (1.2 / 0.3 / 0.1) are too aggressive with a noisy detector at
speed. At 8 m/s the bend is passed but the run fails at the junction ~150 m in.

## Notes

- The camera mount (`CAM_X`, `CAM_Z`, `CAM_FOV` at the top of `run_lane_keeping.py`) must match
  the detector's geometry in `perception/classical_lane.py`.
- If the lane is lost (`detected = False`), the steering command fades toward straight rather
  than being held.
- Always let the script finish or press Ctrl+C so it can clean up its actors.
