"""
Evaluation metrics for the lane-keeping system.

These functions operate on a sequence of per-timestep records (one dict per frame/step),
each expected to have at least:
    - timestamp:        float, seconds
    - lateral_offset:   float, meters (signed distance from lane center; 0 = centered)
    - lane_width:       float, meters (optional, used for percentage-of-lane-width stats)
    - detected:         bool (optional, whether lane detection succeeded this frame)

This format is intentionally simple so it doesn't depend on CARLA, OpenCV, or any
perception/control code — it just needs a list of dicts, which makes it easy to test now
with synthetic data and plug in real data later once perception/control pipelines exist.
"""

import math


def lateral_deviation_stats(records, lane_width=None):
    """
    Compute mean, max, and RMSE of lateral deviation from lane center.

    Args:
        records: list of dicts, each with a 'lateral_offset' key (meters).
        lane_width: optional float (meters). If provided, also returns stats as a
                    percentage of lane width.

    Returns:
        dict with keys: mean_m, max_m, rmse_m, and (if lane_width given) mean_pct, max_pct.
    """
    offsets = [abs(r["lateral_offset"]) for r in records]
    if not offsets:
        raise ValueError("No records provided")

    mean_m = sum(offsets) / len(offsets)
    max_m = max(offsets)
    rmse_m = math.sqrt(sum(o ** 2 for o in offsets) / len(offsets))

    result = {"mean_m": mean_m, "max_m": max_m, "rmse_m": rmse_m}

    if lane_width:
        result["mean_pct"] = 100 * mean_m / lane_width
        result["max_pct"] = 100 * max_m / lane_width

    return result


def correction_responsiveness(records, deviation_threshold=0.3, recovery_threshold=0.1):
    """
    Measure how quickly the system returns to lane center after a deviation.

    A "deviation event" starts the first time |lateral_offset| exceeds deviation_threshold,
    and ends the first time it drops back below recovery_threshold. Returns the list of
    recovery times (in seconds) for each such event found in the record sequence.

    Args:
        records: list of dicts with 'timestamp' and 'lateral_offset'.
        deviation_threshold: meters; offset above this starts a deviation event.
        recovery_threshold: meters; offset below this ends the event (should be < deviation_threshold).

    Returns:
        list of float recovery times in seconds, one per completed deviation event.
    """
    recovery_times = []
    event_start_time = None

    for r in records:
        offset = abs(r["lateral_offset"])
        t = r["timestamp"]

        if event_start_time is None:
            if offset > deviation_threshold:
                event_start_time = t
        else:
            if offset < recovery_threshold:
                recovery_times.append(t - event_start_time)
                event_start_time = None

    return recovery_times


def detection_accuracy(records):
    """
    Fraction of frames where lane detection succeeded.

    Args:
        records: list of dicts, each optionally with a 'detected' bool key.
                 Frames without a 'detected' key are ignored.

    Returns:
        float in [0, 1], or None if no records have a 'detected' field.
    """
    flagged = [r["detected"] for r in records if "detected" in r]
    if not flagged:
        return None
    return sum(1 for d in flagged if d) / len(flagged)


def summarize(records, lane_width=None, deviation_threshold=0.3, recovery_threshold=0.1):
    """
    Convenience function: run all metrics on a record sequence and return a combined summary.
    """
    deviation = lateral_deviation_stats(records, lane_width=lane_width)
    recovery_times = correction_responsiveness(
        records, deviation_threshold=deviation_threshold, recovery_threshold=recovery_threshold
    )
    accuracy = detection_accuracy(records)

    return {
        "n_frames": len(records),
        "deviation": deviation,
        "recovery_times_s": recovery_times,
        "mean_recovery_s": (sum(recovery_times) / len(recovery_times)) if recovery_times else None,
        "detection_accuracy": accuracy,
    }
