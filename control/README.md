# control/

PID controller for lane-keeping steering correction.

## Files

- `pid.py` — `PIDController`: generic PID controller. Takes an error signal (lateral
  offset or heading error) and `dt`, returns a correction output. Decoupled from CARLA
  so it can be written and tuned now, before perception exists.
- `synthetic_test.py` — closed-loop smoke test: a simple synthetic 1D vehicle model
  with a constant disturbance (simulating a curve/crosswind), controlled by the PID,
  to prove the controller actually stabilizes a drifting signal before it's wired to
  CARLA or real lane detection output.

## Usage

Run the smoke test:

```bash
python synthetic_test.py
```

Once real lane detection exists, wire it up per-frame:

```python
from pid import PIDController

pid = PIDController(kp=1.2, ki=0.3, kd=0.1, output_limits=(-1.0, 1.0),
                     integral_limits=(-2.0, 2.0))

for frame in drive_loop():
    steering_correction = pid.compute(error=frame.lateral_offset, dt=frame.dt)
    apply_steering(steering_correction)  # CARLA vehicle control call
```

Call `pid.reset()` if detection is lost for a stretch (stale integral/derivative
history would otherwise produce a bad correction when detection resumes).

## Notes

- Gains (`kp=1.2, ki=0.3, kd=0.1`) are untuned placeholders that happen to converge on
  the synthetic disturbance model in `synthetic_test.py` -- real tuning needs real
  CARLA runs once lane detection is feeding actual `lateral_offset`/`heading_error`.
- `output_limits` and `integral_limits` exist to prevent output saturation and integral
  windup, which matter more once this is driving an actual steering command.
- This can eventually plug into `evaluation/`'s `RunLogger`/`metrics.py` the same way a
  real CARLA run would, to evaluate the controller in isolation (e.g. recovery time
  against a synthetic drift pattern) even before CARLA integration.
