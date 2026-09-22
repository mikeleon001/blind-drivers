# Blind Drivers — Vision-Based Lane Detection and Lane-Keeping Assistance

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

- **Simulation** — CARLA environment setup, scenarios, and data collection.
- **Perception** — lane detection (classical computer vision, with a CNN-based approach as a
  stretch goal).
- **Control** — a PID/geometric controller that uses detected lane geometry to keep the vehicle
  centered in its lane.

Performance is assessed via lateral deviation from lane center, correction responsiveness, and
lane detection accuracy, across multiple CARLA maps, road geometries, and weather/lighting
conditions (see `evaluation/`).

## Repo Structure

```
simulation/     CARLA environment setup, scenario configs, data collection scripts
perception/     Lane detection: classical CV pipeline, CNN component (stretch goal)
control/        Lane-centering controller (PID / geometric)
evaluation/     Metrics, logging, test scripts, and results
docs/           Proposal, reference papers, meeting notes
```

## Getting Started

### Requirements

- CARLA (version TBD — confirm with the team)
- Python 3.x
- See `requirements.txt` for Python package dependencies

### Setup

```bash
git clone <repo-url>
cd blind-drivers
pip install -r requirements.txt
```

CARLA itself is not included in this repo (it's a large separate install) — see
[CARLA's Quick Start guide](https://carla.readthedocs.io/en/latest/start_quickstart/) to install it.

## References

Full reference list is in the project proposal (`docs/proposal.pdf`). Key ones include CARLA
itself, and prior work on CARLA-based lane-keeping and embedded lane detection systems.
