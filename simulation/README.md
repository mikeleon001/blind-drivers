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
| `--detector gt\|cv` | `gt` = lane offset from the CARLA map (ground truth), `cv` = OpenCV detector from `perception/` |
| `--save-debug N` | with `cv`: save the camera frame and bird's-eye debug image every N frames |
| `--tag NAME` | label put in the CSV file name (use it to tell sweeps apart) |
| `--town NAME` | load another map, e.g. `Town04` |
| `--out DIR` | output folder (default `evaluation/runs/`) |

## Output

`evaluation/runs/run_<tag>_spawn<N>_<timestamp>.csv`, one row per frame, with the standard
`evaluation/logger.py` fields plus `steer`, `speed`, `disturbance`, and the map ground truth
`gt_offset` and `gt_lane_width` (always logged, so a detector's error can be measured). With
`--save-debug`, images go to `evaluation/runs/debug_<timestamp>/` (`*_cam.png` = camera,
`*_bev.png` = bird's-eye view with the detected paint in magenta and the chosen lines in red).

## Notes

- The camera mount (`CAM_X`, `CAM_Z`, `CAM_FOV` at the top of `run_lane_keeping.py`) must match
  the detector's geometry in `perception/classical_lane.py`.
- If the lane is lost (`detected = False`), the steering command fades toward straight rather
  than being held.
- Always let the script finish or press Ctrl+C so it can clean up its actors.
