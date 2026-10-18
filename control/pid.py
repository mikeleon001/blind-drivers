"""
Generic PID controller for lane-keeping steering correction.

Intentionally decoupled from CARLA: it just takes an error signal (e.g. lateral
offset from lane center, in meters, or heading error, in radians) and a time
step, and returns a correction output. This lets the control logic be written,
tested, and tuned now, before a real perception pipeline exists - once lane
detection produces a real lateral_offset/heading_error per frame, swap the
synthetic signal here for the real one and the controller doesn't change.
"""


class PIDController:
    def __init__(self, kp, ki, kd, output_limits=None, integral_limits=None):
        """
        Args:
            kp, ki, kd: proportional/integral/derivative gains.
            output_limits: optional (min, max) tuple to clamp the output
                (e.g. (-1.0, 1.0) for a normalized steering command).
            integral_limits: optional (min, max) tuple to clamp the accumulated
                integral term, to prevent integral windup during sustained
                error (e.g. a long curve or a stuck sensor).
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.integral_limits = integral_limits

        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        """Clear accumulated integral and derivative history. Call this when
        starting a new run or after a long gap in control (e.g. detection lost)."""
        self._integral = 0.0
        self._prev_error = None

    def compute(self, error, dt):
        """
        Compute one control step.

        Args:
            error: current error (e.g. lateral offset in meters; positive
                   meaning drifted to one side, by convention of the caller).
            dt: time elapsed since the last call, in seconds. Must be > 0.

        Returns:
            float: the control output (e.g. a steering correction).
        """
        if dt <= 0:
            raise ValueError(f"dt must be positive, got {dt}")

        # Proportional
        p_term = self.kp * error

        # Integral (with optional anti-windup clamp)
        self._integral += error * dt
        if self.integral_limits is not None:
            lo, hi = self.integral_limits
            self._integral = max(lo, min(hi, self._integral))
        i_term = self.ki * self._integral

        # Derivative (zero on the very first call -- no history yet)
        if self._prev_error is None:
            d_term = 0.0
        else:
            d_term = self.kd * (error - self._prev_error) / dt
        self._prev_error = error

        output = p_term + i_term + d_term

        if self.output_limits is not None:
            lo, hi = self.output_limits
            output = max(lo, min(hi, output))

        return output


if __name__ == "__main__":
    # Quick smoke test: constant error should converge the output toward
    # saturating the P term, integral should accumulate, derivative should be
    # ~0 once error stops changing.
    pid = PIDController(kp=1.0, ki=0.1, kd=0.05, output_limits=(-1.0, 1.0))
    out1 = pid.compute(error=0.5, dt=0.1)
    out2 = pid.compute(error=0.5, dt=0.1)
    assert out2 > out1, "integral term should grow output on sustained error"
    pid.reset()
    assert pid._integral == 0.0 and pid._prev_error is None
    print("pid.py smoke test passed")
