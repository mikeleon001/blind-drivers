# evaluation/

Metrics, logging, comparison and test scripts for assessing lane-keeping system performance.

## Files

- `metrics.py` - core evaluation metrics: lateral deviation (mean/max/RMSE), correction
  responsiveness (recovery time after a deviation), and lane detection accuracy. Operates on
  plain lists of per-frame dicts, so it doesn't depend on CARLA or any perception/control code.
- `logger.py` - `RunLogger` writes per-frame data to CSV incrementally during a run;
  `load_run()` reads it back into the record format `metrics.py` expects.
- `compare_runs.py` - prints one table with a row per run CSV in `evaluation/runs/`, so runs
  with different gains, spawn points or detectors can be compared side by side (see below).
- `synthetic_test.py` - generates a fake run (simulated drift-and-correct pattern) and runs
  it through the full logger -> metrics pipeline, to prove the infrastructure works without
  any simulation output.

## Real runs

`simulation/run_lane_keeping.py` writes one CSV per run to `evaluation/runs/` (not committed to
git). Besides the fields from `logger.py` (`timestamp`, `lateral_offset`, `heading_error`,
`lane_width`, `detected`, `processing_time_ms`), it logs `steer`, `speed`, `disturbance` (the
injected steering kick), and the map ground truth `gt_offset` and `gt_lane_width`.

Compare runs (from the repo root):

```bash
python evaluation/compare_runs.py                    # every run
python evaluation/compare_runs.py --tag kp --skip 3  # only runs tagged "kp", ignore first 3 s
python evaluation/compare_runs.py --out summary.csv  # also save the table
```

| Column | Meaning |
|---|---|
| `mean_m`, `max_m`, `rmse_m` | lateral deviation from lane center |
| `events`, `rec_s` | recovery events (over 0.3 m out, back under 0.1 m) and their mean duration |
| `kick_peak_m`, `kick_rec_s` | for runs with a steering kick: peak offset after the kick, and time from that peak back under 0.1 m |
| `det_pct` | percentage of frames where a lane was detected |
| `det_err_m` | RMSE between the logged offset and the map ground truth (0 for ground-truth runs, the detector's own error for `--detector cv`) |
| `width_err_m` | mean detected minus map lane width; far from 0 means the camera geometry is off |

## Synthetic test

Run from inside this folder:

```bash
python synthetic_test.py
```

## Using the logger and metrics directly

```python
from logger import RunLogger

with RunLogger("run_001.csv") as logger:
    for frame in drive_loop():
        logger.log(
            timestamp=frame.t,
            lateral_offset=frame.offset,
            heading_error=frame.heading_error,
            lane_width=frame.lane_width,
            detected=frame.lane_detected,
            processing_time_ms=frame.proc_time_ms,
        )
```

```python
from logger import load_run
from metrics import summarize

records = load_run("run_001.csv")
summary = summarize(records, lane_width=3.5)
```

## Metrics tracked

- Lateral deviation from lane center (mean, max, RMSE), in meters and % of lane width
- Correction responsiveness (time to return to lane center after a deviation)
- Lane detection accuracy (fraction of frames with successful detection)

Tests are intended to be run across multiple CARLA maps, road geometries (straight/curved),
and weather/lighting conditions. So far runs are on Town10HD with the default weather.
