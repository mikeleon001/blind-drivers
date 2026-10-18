"""
End-to-end smoke test of the evaluation infrastructure using synthetic (fake) data.

This simulates a vehicle that drifts out of lane and gets corrected, logs it via RunLogger,
reloads it via load_run, and runs it through metrics.py - proving the whole evaluation
pipeline works before there's any real CARLA/perception/control output to plug in.

Run with: python synthetic_test.py
"""

import math
import os

from logger import RunLogger, load_run
from metrics import summarize


def generate_synthetic_run(out_path, duration_s=10.0, dt=0.05, lane_width=3.5):
    """
    Generate a fake run: vehicle drifts away from center then gets pulled back,
    repeating a couple of times, with lane detection occasionally "failing" to
    simulate real-world imperfect detection.
    """
    t = 0.0
    with RunLogger(out_path) as logger:
        while t < duration_s:
            # Oscillating drift-and-correct pattern, decaying noise
            offset = 0.4 * math.sin(t * 0.8) * math.exp(-0.02 * t)
            heading_error = 0.1 * math.cos(t * 0.8)
            detected = not (int(t * 10) % 37 == 0)  # occasional dropped detection

            logger.log(
                timestamp=round(t, 3),
                lateral_offset=round(offset, 4),
                heading_error=round(heading_error, 4),
                lane_width=lane_width,
                detected=detected,
                processing_time_ms=round(15 + 5 * math.sin(t), 2),
            )
            t += dt


def main():
    tmp_path = "/tmp/_synthetic_run.csv"
    generate_synthetic_run(tmp_path)

    records = load_run(tmp_path)
    summary = summarize(records, lane_width=3.5, deviation_threshold=0.3, recovery_threshold=0.1)

    print(f"Loaded {summary['n_frames']} frames from synthetic run")
    print("Lateral deviation:")
    print(f"  mean: {summary['deviation']['mean_m']:.3f} m "
          f"({summary['deviation'].get('mean_pct', 0):.1f}% of lane width)")
    print(f"  max:  {summary['deviation']['max_m']:.3f} m "
          f"({summary['deviation'].get('max_pct', 0):.1f}% of lane width)")
    print(f"  RMSE: {summary['deviation']['rmse_m']:.3f} m")
    print(f"Recovery events: {len(summary['recovery_times_s'])}")
    if summary["mean_recovery_s"] is not None:
        print(f"  mean recovery time: {summary['mean_recovery_s']:.2f} s")
    print(f"Detection accuracy: {summary['detection_accuracy']:.1%}")

    os.remove(tmp_path)


if __name__ == "__main__":
    main()
