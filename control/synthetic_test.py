"""
Closed-loop smoke test for the PID controller using a synthetic vehicle model.

This simulates a very simple 1D lateral dynamics model (no CARLA): the vehicle
has a lateral_offset from lane center that drifts due to a disturbance (like a
road curve or crosswind), and the PID controller's output nudges it back
toward zero each step. This proves the controller actually stabilizes a
drifting signal before it's ever wired to CARLA or real lane detection.

Run with: python synthetic_test.py
"""

from pid import PIDController


def simulate(pid, duration_s=10.0, dt=0.05, disturbance=0.15, correction_gain=1.0):
    """
    Very simple closed-loop simulation:
        lateral_offset[t+dt] = lateral_offset[t] + disturbance*dt - correction_gain*output*dt

    i.e. the disturbance keeps pushing the vehicle off-center, and the PID
    controller's output pulls it back. Not a real vehicle dynamics model --
    just enough to exercise the controller in a closed loop.
    """
    t = 0.0
    offset = 0.0
    history = []

    while t < duration_s:
        output = pid.compute(error=offset, dt=dt)
        offset = offset + disturbance * dt - correction_gain * output * dt
        t += dt
        history.append({"t": round(t, 3), "offset": offset, "output": output})

    return history


def main():
    pid = PIDController(kp=1.2, ki=0.3, kd=0.1, output_limits=(-1.0, 1.0),
                         integral_limits=(-2.0, 2.0))
    history = simulate(pid)

    offsets = [h["offset"] for h in history]
    final_offset = offsets[-1]
    max_offset = max(abs(o) for o in offsets)

    print(f"Ran {len(history)} steps")
    print(f"Max |offset| during run: {max_offset:.4f} m")
    print(f"Final offset: {final_offset:.4f} m")

    # Sanity check: with a constant disturbance, a properly tuned PID should
    # settle close to zero, not diverge or oscillate wildly.
    assert abs(final_offset) < 0.05, (
        f"controller did not converge: final offset {final_offset:.4f} m"
    )
    print("Closed-loop synthetic test passed (controller converged).")


if __name__ == "__main__":
    main()
