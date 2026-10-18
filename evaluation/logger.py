"""
Simple CSV logger for per-frame evaluation data.

Whoever wires up CARLA/perception/control can call RunLogger.log(...) once per frame/step;
this writes incrementally to a CSV so a crashed or interrupted run doesn't lose data. The
resulting CSV's rows are exactly the kind of dicts metrics.py expects (list of dicts), via
load_run().
"""

import csv
import os

FIELDS = ["timestamp", "lateral_offset", "heading_error", "lane_width", "detected", "processing_time_ms"]


class RunLogger:
    def __init__(self, out_path, fields=None):
        self.fields = fields or FIELDS
        self.out_path = out_path
        self._file = open(out_path, "w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.fields)
        self._writer.writeheader()

    def log(self, **kwargs):
        """Log one frame/step. Unrecognized keys are ignored; missing keys are left blank."""
        row = {k: kwargs.get(k, "") for k in self.fields}
        self._writer.writerow(row)

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def load_run(csv_path):
    """
    Load a logged run CSV back into a list of dicts suitable for metrics.py.
    Numeric fields are cast to float/bool where possible; blanks are dropped from the dict.
    """
    records = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            record = {}
            for key, val in row.items():
                if val == "" or val is None:
                    continue
                if key == "detected":
                    record[key] = val.strip().lower() in ("1", "true", "yes")
                elif key == "timestamp":
                    record[key] = float(val)
                else:
                    try:
                        record[key] = float(val)
                    except ValueError:
                        record[key] = val
            records.append(record)
    return records


if __name__ == "__main__":
    # Quick smoke test
    path = "/tmp/_logger_smoke_test.csv"
    with RunLogger(path) as logger:
        logger.log(timestamp=0.0, lateral_offset=0.05, heading_error=0.01, lane_width=3.5, detected=True)
        logger.log(timestamp=0.1, lateral_offset=0.07, heading_error=0.02, lane_width=3.5, detected=True)

    loaded = load_run(path)
    assert len(loaded) == 2
    assert loaded[0]["lateral_offset"] == 0.05
    assert loaded[0]["detected"] is True
    os.remove(path)
    print("logger.py smoke test passed")
