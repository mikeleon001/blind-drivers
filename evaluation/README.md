# evaluation/

Metrics, logging, and test scripts for assessing lane-keeping system performance.

## Files

- `metrics.py` - core evaluation metrics: lateral deviation (mean/max/RMSE), correction
  responsiveness (recovery time after a deviation), and lane detection accuracy. Operates on
  plain lists of per-frame dicts, so it doesn't depend on CARLA or any perception/control code.
- `logger.py` - `RunLogger` writes per-frame data to CSV incrementally during a run;
  `load_run()` reads it back into the record format `metrics.py` expects.
- `synthetic_test.py` - generates a fake run (simulated drift-and-correct pattern) and runs
  it through the full logger → metrics pipeline, to prove the infrastructure works before
  there's real simulation/perception/control output to plug in.

## Usage

Run the smoke test:

```bash
python synthetic_test.py
```

Once real data exists (from simulation + perception + control), log each frame/step during a
run with:

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

Then evaluate it:

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
and weather/lighting conditions once simulation + perception + control are integrated.
