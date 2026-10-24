# Blind Drivers - Vision-Based Lane Detection and Lane-Keeping Assistance

CS 5230 (Connected & Autonomous Vehicles) final project. We're building and evaluating a
vision-based lane detection and lane-keeping control system in the CARLA simulator, with a
stretch goal of testing on physical hardware (Jetson Nano / DRIVE AGX Orin).

## Team

- Daniel Plascencia
- Ethan J. Reidel
- Mihail Chitorog

## Project Overview

See [`docs/proposal.pdf`](docs/proposal.pdf) for the full accepted project proposal, including
background, goals, methodology, and references.

The system has three main pieces:

- **Simulation** - CARLA environment setup, scenarios, and data collection.
- **Perception** - lane detection (classical computer vision, with a CNN-based approach as a
  stretch goal).
- **Control** - a PID/geometric controller that uses detected lane geometry to keep the vehicle
  centered in its lane.

Performance is assessed via lateral deviation from lane center, correction responsiveness, and
lane detection accuracy, across multiple CARLA maps, road geometries, and weather/lighting
conditions (see `evaluation/`).

## Repo Structure

```
simulation/     CARLA closed-loop runner, spawn-point finder
perception/     Lane detection: classical CV detector (done, being tuned), CNN (stretch goal)
control/        Lane-centering controller (PID / geometric)
evaluation/     Metrics, run logging, run-comparison tool, test scripts
docs/           Proposal, reference papers, meeting notes
```

Each folder has its own README with details.

## Status

Working end to end in CARLA (synchronous mode, 20 Hz):

- Closed loop: camera frame -> lane offset -> PID -> steering, with every run logged to CSV and
  scored by `evaluation/`.
- With the lane offset taken from the CARLA map (ground truth), the PID holds the lane and
  recovers from a steering disturbance (see `control/README.md` for the gain comparison).
- A classical OpenCV lane detector (`perception/classical_lane.py`) produces the offset
  directly from the camera image, in meters. It follows straight roads in Town10HD but still
  loses the lane near sharper curves and at junctions - this is the current work item.
- Not started: CNN-based detector, hardware tests, multiple maps / weather sweeps.

## Getting Started

### Requirements

- CARLA 0.9.16 (Windows build, run from `CarlaUE4.exe`)
- Python 3.12 environment with the `carla` package installed (on the lab machine this is the
  conda env `carla-sim312`)
- `numpy` and `opencv-python` for the detector and tools; see `requirements.txt` for the full
  list (PyTorch is only needed for the CNN stretch goal)

### Setup

```bash
git clone https://github.com/mikeleon001/blind-drivers.git
cd blind-drivers
pip install -r requirements.txt
```

CARLA itself is not included in this repo (it's a large separate install) - see
[CARLA's Quick Start guide](https://carla.readthedocs.io/en/latest/start_quickstart/) to install it.

### Running (all commands from the repo root)

1. Start the simulator (`CarlaUE4.exe`) and wait until the town has loaded.
2. Activate the environment: `conda activate carla-sim312`
3. Pick a spawn point on a straight stretch: `python simulation/list_spawns.py --draw`
4. Drive:
   ```bash
   python simulation/run_lane_keeping.py --spawn 10                      # ground-truth lane offset
   python simulation/run_lane_keeping.py --spawn 10 --detector cv --save-debug 20   # OpenCV detector
   ```
5. Compare all runs: `python evaluation/compare_runs.py`

Run CSVs and debug images are written to `evaluation/runs/`, which is not committed to git.

Checks that need no CARLA: `python perception/synthetic_test.py` (from the repo root), and
`python synthetic_test.py` from inside `control/` or `evaluation/`.

## References

Full reference list is in the project proposal (`docs/proposal.pdf`). Key ones include CARLA itself, and prior work on CARLA-based lane-keeping and embedded lane detection systems.
