"""
evaluation/compare_runs.py

Print one comparison table for every run CSV in evaluation/runs/ (or another folder).

    python evaluation/compare_runs.py
    python evaluation/compare_runs.py --dir evaluation/runs --skip 3
    python evaluation/compare_runs.py --tag kp          # only files whose tag contains "kp"
    python evaluation/compare_runs.py --out summary.csv # also save the table as CSV

Columns:
    deviation stats  mean / max / RMSE of |lateral_offset| (from metrics.py)
    events, rec_s    generic recovery events (>0.3 m out, back <0.1 m) and their mean time
    kick_peak_m      biggest |offset| in the 3 s after a --kick started (blank if no kick)
    kick_rec_s       time from that peak until |offset| < 0.1 m (blank if no kick)
    det_pct          lane detection accuracy (% of frames where a lane was found)
    det_err_m        RMSE between the detector's offset and the map ground truth (gt_offset)
    width_err_m      mean (detected - map) lane width; far from 0 means the camera geometry is off
    true_*_m         mean / max / RMSE of |gt_offset| over ALL frames: where the car really was
                     relative to the map lane center. Unlike mean_m / max_m / rmse_m (which use the
                     offset the detector reported, 0 while the lane is lost) these cannot be
                     flattered by a detector that loses the lane. Blank for older runs.

--skip N ignores the first N seconds of each run (the car starts from a standstill), so the
numbers describe steady driving instead of the start-up transient.
Only the Python standard library is needed.
"""
import argparse
import csv
import glob
import os
import re
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from evaluation.logger import load_run          # noqa: E402
from evaluation.metrics import summarize        # noqa: E402

NAME_RE = re.compile(r"run_(?P<tag>.+?)_spawn(?P<spawn>\d+)_(?P<stamp>\d{8}_\d{6})\.csv$")
COLUMNS = ["tag", "spawn", "stamp", "frames", "mean_m", "max_m", "rmse_m",
           "events", "rec_s", "kick_peak_m", "kick_rec_s", "det_pct", "det_err_m", "width_err_m",
           "true_mean_m", "true_max_m", "true_rmse_m"]


def width_error(records):
    """Mean (detected lane_width - map lane width) in m over detected frames: a steady non-zero
    value means the camera geometry (height / fov) in the detector doesn't match the sensor."""
    d = [r["lane_width"] - r["gt_lane_width"] for r in records
         if "gt_lane_width" in r and "lane_width" in r and r.get("detected", True)]
    return sum(d) / len(d) if d else None


def detector_error(records):
    """RMSE (m) between the logged lateral_offset and the map ground truth (gt_offset), over
    frames where the lane was detected. ~0 for ground-truth runs; the detector's own error for
    --detector cv runs. None if the CSV has no gt_offset column (older runs)."""
    errs = [r["lateral_offset"] - r["gt_offset"] for r in records
            if "gt_offset" in r and r.get("detected", True)]
    if not errs:
        return None
    return (sum(e * e for e in errs) / len(errs)) ** 0.5


def true_deviation(records):
    """(mean, max, RMSE) of |gt_offset| over all frames, or (None, None, None) for old runs."""
    g = [abs(r["gt_offset"]) for r in records if "gt_offset" in r]
    if not g:
        return None, None, None
    return sum(g) / len(g), max(g), (sum(x * x for x in g) / len(g)) ** 0.5


def kick_response(records, window_s=3.0, recovery_threshold=0.1):
    """(peak |offset| in the window after the kick starts, seconds from that peak to
    |offset| < recovery_threshold). (None, None) if the run had no kick."""
    kicked = [r for r in records if abs(r.get("disturbance", 0.0)) > 0]
    if not kicked:
        return None, None
    t0 = kicked[0]["timestamp"]
    window = [r for r in records if t0 <= r["timestamp"] <= t0 + window_s]
    peak_r = max(window, key=lambda r: abs(r["lateral_offset"]))
    peak = abs(peak_r["lateral_offset"])
    if peak < recovery_threshold:
        return peak, 0.0
    for r in records:
        if r["timestamp"] >= peak_r["timestamp"] and abs(r["lateral_offset"]) < recovery_threshold:
            return peak, r["timestamp"] - peak_r["timestamp"]
    return peak, None   # never recovered within the run


def fmt(v, nd=3):
    return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", default=os.path.join(REPO_ROOT, "evaluation", "runs"))
    ap.add_argument("--tag", default=None, help="only runs whose tag contains this text")
    ap.add_argument("--skip", type=float, default=0.0, help="ignore the first N seconds")
    ap.add_argument("--dev", type=float, default=0.3, help="deviation threshold, m")
    ap.add_argument("--rec", type=float, default=0.1, help="recovery threshold, m")
    ap.add_argument("--out", default=None, help="also write the table to this CSV")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.dir, "run_*.csv")))
    rows = []
    for p in paths:
        m = NAME_RE.search(os.path.basename(p))
        tag, spawn, stamp = (m["tag"], m["spawn"], m["stamp"]) if m else (os.path.basename(p), "", "")
        if args.tag and args.tag not in tag:
            continue
        records = [r for r in load_run(p) if r.get("timestamp", 0.0) >= args.skip]
        if not records:
            print(f"skipping {os.path.basename(p)}: no records", file=sys.stderr)
            continue
        s = summarize(records, lane_width=records[0].get("lane_width"),
                      deviation_threshold=args.dev, recovery_threshold=args.rec)
        d = s["deviation"]
        kp, kr = kick_response(records, recovery_threshold=args.rec)
        acc = s["detection_accuracy"]
        tm, tx, tr = true_deviation(records)
        rows.append({
            "tag": tag, "spawn": spawn, "stamp": stamp, "frames": s["n_frames"],
            "mean_m": fmt(d["mean_m"]), "max_m": fmt(d["max_m"]), "rmse_m": fmt(d["rmse_m"]),
            "events": len(s["recovery_times_s"]), "rec_s": fmt(s["mean_recovery_s"], 2),
            "kick_peak_m": fmt(kp), "kick_rec_s": fmt(kr, 2),
            "det_pct": "" if acc is None else f"{100 * acc:.1f}",
            "det_err_m": fmt(detector_error(records)),
            "width_err_m": fmt(width_error(records)),
            "true_mean_m": fmt(tm), "true_max_m": fmt(tx), "true_rmse_m": fmt(tr),
        })

    if not rows:
        print(f"No matching run CSVs in {args.dir}")
        return
    rows.sort(key=lambda r: (r["tag"], r["spawn"], r["stamp"]))

    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in COLUMNS}
    print("  ".join(c.ljust(widths[c]) for c in COLUMNS))
    print("  ".join("-" * widths[c] for c in COLUMNS))
    for r in rows:
        print("  ".join(str(r[c]).ljust(widths[c]) for c in COLUMNS))
    if args.skip:
        print(f"\n(first {args.skip:g} s of each run ignored)")

    if args.out:
        with open(args.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS)
            w.writeheader()
            w.writerows(rows)
        print(f"saved {args.out}")


if __name__ == "__main__":
    main()
