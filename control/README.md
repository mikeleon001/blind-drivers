# control/

PID controller for lane-keeping steering correction.

## Files

- `pid.py` - `PIDController`: generic PID controller. Takes an error signal (lateral
  offset or heading error) and `dt`, returns a correction output. Decoupled from CARLA
  so it can be written and tuned independently of the simulator.
- `synthetic_test.py` - closed-loop smoke test: a simple synthetic 1D vehicle model
  with a constant disturbance (simulating a curve/crosswind), controlled by the PID,
  to prove the controller actually stabilizes a drifting signal before it's wired to
  CARLA or real lane detection output.

## Usage

Run the smoke test (from inside this folder):

```bash
python synthetic_test.py
```

In CARLA the controller is used by `simulation/run_lane_keeping.py`:

```python
from control.pid import PIDController

pid = PIDController(kp=1.2, ki=0.3, kd=0.1, output_limits=(-1.0, 1.0),
                    integral_limits=(-2.0, 2.0))

# lateral_offset > 0 means the car is right of lane center, so steer the opposite way:
steer = -pid.compute(error=lateral_offset, dt=0.05)
```

Call `pid.reset()` if detection is lost for a stretch (stale integral/derivative
history would otherwise produce a bad correction when detection resumes).

## Results so far

Lane offset from the CARLA map (ground truth), Town10HD, spawn 0, 8 m/s, 20 Hz:

- Starting 0.6 m right of the lane center: back within 0.1 m in about 1.2 s with the default
  gains (kp 1.2, ki 0.3, kd 0.1), with a small overshoot.
- Steering kick of 0.6 for 0.5 s at t = 8 s, varying kp only:

| kp | peak offset after kick | RMSE (whole run) | recovery from peak |
|---|---|---|---|
| 0.8 | 0.90 m | 0.147 m | 0.40 s |
| 1.2 | 0.67 m | 0.102 m | 0.35 s |
| 1.8 | 0.50 m | 0.079 m | 0.30 s |

Higher kp rejected the disturbance better in this range; the upper limit (where the car starts
to oscillate) has not been found yet.

## Notes

- Only the lateral offset is used so far. The detector also provides a heading error, which
  is not used yet; a heading term is the next planned improvement, since an offset-only
  controller lags on curves.
- Gains are still not tuned beyond the kp comparison above; ki and kd have not been varied.
- `output_limits` and `integral_limits` exist to prevent output saturation and integral
  windup, which matter now that this is driving an actual steering command.
- `evaluation/` logs every run and can compare controller settings (see `evaluation/README.md`).
