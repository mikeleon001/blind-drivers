"""
simulation/run_lane_keeping.py

Closed-loop lane keeping in CARLA (synchronous mode) for Blind Drivers, CS 5230.

    camera frame -> lane offset -> PIDController -> apply_control()
                                      |
                                      +-> RunLogger CSV -> evaluation.metrics.summarize()

Uses the repo's own control.pid.PIDController and evaluation.logger/metrics.

Run from the repo root (CarlaUE4.exe already running):
    conda activate carla-sim312
    python simulation/run_lane_keeping.py
    python simulation/run_lane_keeping.py --spawn 5 --duration 40
    python simulation/run_lane_keeping.py --start-offset 0.6            # start off-center
    python simulation/run_lane_keeping.py --kick 0.35 --kick-at 8       # steering disturbance

    python simulation/run_lane_keeping.py --detector cv --save-debug 20   # classical OpenCV detector
    python simulation/run_lane_keeping.py --detector unet --save-debug 20 # U-Net lane mask (needs torch)

--detector gt (default) uses ground truth from the CARLA map. --detector cv uses
perception/classical_lane.py on the camera image. --detector unet uses the trained U-Net
(perception/unet_lane.py) to find the paint, then the same bird's-eye fit as cv. The map ground truth is always logged as
gt_offset, so the detector's error can be measured afterwards (evaluation/compare_runs.py).
"""
import argparse
import math
import os
import queue
import sys
import time

import numpy as np
import carla

# Make `control` and `evaluation` importable no matter where this is launched from.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from control.pid import PIDController                      # noqa: E402
from evaluation.logger import RunLogger, FIELDS            # noqa: E402
from evaluation.logger import load_run                     # noqa: E402
from evaluation.metrics import summarize                   # noqa: E402
from simulation.truth_overlay import (                     # noqa: E402
    lane_points_in_camera_frame, draw_on_bev, draw_on_camera)

HOST, PORT = "localhost", 2000
FIXED_DT = 0.05            # 20 Hz simulated
IMG_W, IMG_H = 800, 600
# Extra per-frame columns beyond the repo's default FIELDS (load_run handles them as floats).
CAM_X, CAM_Z, CAM_FOV = 1.5, 2.4, 90.0     # camera mount; the CV detector's geometry must match
# gt_offset = map ground truth, always logged, so a detector's error can be measured afterwards.
LOG_FIELDS = FIELDS + ["steer", "speed", "disturbance", "gt_offset", "gt_lane_width",
                          "gt_offset_cam", "gt_heading"]


def carla_image_to_bgr(image):
    """carla.Image (BGRA bytes) -> HxWx3 uint8 BGR array (OpenCV-style)."""
    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
    arr = arr.reshape((image.height, image.width, 4))
    return arr[:, :, :3].copy()


def ground_truth_lane(vehicle, carla_map):
    """Return (lateral_offset_m, heading_error_rad, lane_width_m, detected) from the CARLA map.

    lateral_offset_m: + = vehicle is RIGHT of lane center.
    heading_error_rad: + = vehicle yaw is clockwise (to the right) of the lane direction.
    """
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(tf.location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    right = wp.transform.get_right_vector()
    d = tf.location - wp.transform.location
    offset = d.x * right.x + d.y * right.y
    heading_err = math.radians(
        (tf.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180)
    return offset, heading_err, wp.lane_width, True


def ground_truth_at_camera(vehicle, carla_map):
    """The map's lateral offset measured where the DETECTOR measures it: at the camera (CAM_X ahead
    of the vehicle origin), along the car's own sideways direction. gt_offset (at the vehicle
    origin) differs from this by up to ~0.5 m in a bend, so compare the detector with this one.
    Returns (offset_m, heading_error_rad); + offset = right of lane center."""
    tf = vehicle.get_transform()
    fwd = tf.get_forward_vector()
    cam = carla.Location(tf.location.x + fwd.x * CAM_X, tf.location.y + fwd.y * CAM_X, tf.location.z)
    wp = carla_map.get_waypoint(cam, project_to_road=True, lane_type=carla.LaneType.Driving)
    r = wp.transform.get_right_vector()
    perp = (cam.x - wp.transform.location.x) * r.x + (cam.y - wp.transform.location.y) * r.y
    heading = math.radians((tf.rotation.yaw - wp.transform.rotation.yaw + 180) % 360 - 180)
    return perp / max(math.cos(heading), 0.3), heading


def map_lane_ahead(vehicle, carla_map, start=1.0, stop=28.0, step=1.0):
    """The map's own lane ahead of the car, as plain numbers for simulation/truth_overlay.py."""
    tf = vehicle.get_transform()
    wp = carla_map.get_waypoint(tf.location, project_to_road=True,
                                lane_type=carla.LaneType.Driving)
    wps, d = [], start
    while d <= stop:
        nxt = wp.next(d)
        if nxt:
            w = nxt[0]
            r, loc = w.transform.get_right_vector(), w.transform.location
            wps.append((loc.x, loc.y, r.x, r.y, w.lane_width))
        d += step
    f, r = tf.get_forward_vector(), tf.get_right_vector()
    return (tf.location.x, tf.location.y), (f.x, f.y), (r.x, r.y), wps


def shifted_spawn(spawn_tf, lateral_m):
    """Spawn transform moved lateral_m to the right of the spawn point."""
    r = spawn_tf.get_right_vector()
    loc = carla.Location(spawn_tf.location.x + r.x * lateral_m,
                         spawn_tf.location.y + r.y * lateral_m,
                         spawn_tf.location.z + 0.3)   # small lift avoids ground clipping
    return carla.Transform(loc, spawn_tf.rotation)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--spawn", type=int, default=0, help="spawn point index (default 0)")
    p.add_argument("--duration", type=float, default=30.0, help="seconds (default 30)")
    p.add_argument("--speed", type=float, default=8.0, help="target speed m/s (default 8)")
    p.add_argument("--kp", type=float, default=1.2)
    p.add_argument("--ki", type=float, default=0.3)
    p.add_argument("--kd", type=float, default=0.1)
    p.add_argument("--start-offset", type=float, default=0.0,
                   help="start this many meters RIGHT of lane center (negative = left)")
    p.add_argument("--kick", type=float, default=0.0,
                   help="steering disturbance added to the command (e.g. 0.3)")
    p.add_argument("--kick-at", type=float, default=8.0, help="kick start time, s")
    p.add_argument("--kick-dur", type=float, default=0.5, help="kick length, s")
    p.add_argument("--out", default=os.path.join(REPO_ROOT, "evaluation", "runs"),
                   help="folder for the run CSV")
    p.add_argument("--tag", default="gt", help="label in the CSV filename (gt = ground truth)")
    p.add_argument("--town", default=None, help="optionally load a map, e.g. Town04")
    p.add_argument("--detector", choices=["gt", "cv", "unet"], default="gt",
                   help="gt = ground truth from the map; cv = classical OpenCV detector; "
                        "unet = U-Net lane mask + bird's-eye fit")
    p.add_argument("--unet-weights", default=None,
                   help="with --detector unet: weights file (default perception/unet/unet_tusimple.pt)")
    p.add_argument("--unet-threshold", type=float, default=0.5,
                   help="with --detector unet: lane probability needed to count a pixel as paint")
    p.add_argument("--bev-half", type=float, default=6.0,
                   help="bird's-eye view half-width in m (default 6; try 9 for sharp bends)")
    p.add_argument("--bev-near", type=float, default=5.0,
                   help="nearest distance ahead the bird's-eye view covers, m (default 5; try 4)")
    p.add_argument("--kh", type=float, default=0.0,
                   help="heading-error gain (0 = off, default). Steers back toward the lane direction "
                        "and damps overshoot; try 0.5")
    p.add_argument("--smooth", type=float, default=1.0,
                   help="offset smoothing, 0..1: 1 = off (default); 0.5 = average new and old half-half")
    p.add_argument("--steer-rate", type=float, default=0.0,
                   help="max steering change per second (0 = no limit, default); try 4")
    p.add_argument("--max-jump", type=float, default=0.0,
                   help="reject a detector reading that moves more than this many m in one frame "
                        "(0 = off, default; try 0.4). The car cannot really move that fast.")
    p.add_argument("--debug-from", type=float, default=0.0,
                   help="save debug images for EVERY frame between --debug-from and --debug-to seconds")
    p.add_argument("--debug-to", type=float, default=0.0)
    p.add_argument("--save-debug", type=int, default=0, metavar="N",
                   help="with --detector cv/unet: save a camera + bird's-eye debug image every N frames")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tag = args.tag if args.tag != "gt" or args.detector == "gt" else args.detector   # tag = "cv" / "unet"
    csv_path = os.path.join(args.out, f"run_{tag}_spawn{args.spawn}_{stamp}.csv")

    cv_detector, debug_dir = None, None
    if args.detector in ("cv", "unet"):
        import cv2
        if args.detector == "cv":
            from perception.classical_lane import ClassicalLaneDetector
            cv_detector = ClassicalLaneDetector(IMG_W, IMG_H, CAM_FOV, CAM_Z,
                                                 x_near=args.bev_near, y_half=args.bev_half)
        else:
            from perception.unet_lane import UNetLaneDetector
            cv_detector = UNetLaneDetector(IMG_W, IMG_H, CAM_FOV, CAM_Z,
                                           weights=args.unet_weights, threshold=args.unet_threshold,
                                           x_near=args.bev_near, y_half=args.bev_half)
        if args.save_debug or args.debug_to > args.debug_from:
            debug_dir = os.path.join(args.out, f"debug_{stamp}")
            os.makedirs(debug_dir, exist_ok=True)

    client = carla.Client(HOST, PORT)
    client.set_timeout(20.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    carla_map = world.get_map()
    original_settings = world.get_settings()
    actors = []
    logger = None

    try:
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DT
        world.apply_settings(settings)

        spawn_points = carla_map.get_spawn_points()
        if not 0 <= args.spawn < len(spawn_points):
            raise SystemExit(f"--spawn must be 0..{len(spawn_points) - 1}")
        spawn_tf = shifted_spawn(spawn_points[args.spawn], args.start_offset)

        bp_lib = world.get_blueprint_library()
        vehicle = world.spawn_actor(bp_lib.find("vehicle.tesla.model3"), spawn_tf)
        actors.append(vehicle)

        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", str(CAM_FOV))
        camera = world.spawn_actor(cam_bp, carla.Transform(carla.Location(x=CAM_X, z=CAM_Z)),
                                   attach_to=vehicle)
        actors.append(camera)
        frames = queue.Queue()
        camera.listen(frames.put)

        # Same placeholder gains as the repo; steering output clamped to CARLA's range.
        pid = PIDController(args.kp, args.ki, args.kd,
                            output_limits=(-1.0, 1.0), integral_limits=(-2.0, 2.0))
        logger = RunLogger(csv_path, fields=LOG_FIELDS)

        print(f"Logging to {csv_path}")
        print(f"detector={args.detector} spawn={args.spawn} start_offset={args.start_offset} "
              f"kick={args.kick} gains=({args.kp},{args.ki},{args.kd})")
        last_steer = 0.0
        off_f, hdg_f, prev_cmd = None, None, 0.0
        accepted, since_accept = None, 0

        n_steps = int(args.duration / FIXED_DT)
        for i in range(n_steps):
            t = i * FIXED_DT
            world.tick()
            image = frames.get(timeout=5.0)          # frame for this tick
            frame = carla_image_to_bgr(image)

            gt_offset, gt_heading, gt_lane_w, _ = ground_truth_lane(vehicle, carla_map)
            gt_off_cam, _ = ground_truth_at_camera(vehicle, carla_map)

            t0 = time.perf_counter()
            if cv_detector is None:
                offset, heading_err, lane_w, detected = gt_offset, gt_heading, gt_lane_w, True
            else:
                res = cv_detector.detect(frame, want_debug=bool(debug_dir))
                offset, heading_err, lane_w, detected = (res.lateral_offset, res.heading_error,
                                                         res.lane_width, res.detected)
                if debug_dir and ((args.save_debug and i % args.save_debug == 0)
                                  or args.debug_from <= t <= args.debug_to):
                    # Also draw the CARLA map's lane (green = center, orange = edges) so the
                    # camera and the map can be compared by eye.
                    veh_xy, fwd, right, wps = map_lane_ahead(vehicle, carla_map)
                    pts = lane_points_in_camera_frame(veh_xy, fwd, right, wps, CAM_X)
                    cam_img = draw_on_camera(frame.copy(), pts, cv_detector.ground_to_pixel)
                    bev_img = draw_on_bev(res.debug["bev"], pts, cv_detector.y_half,
                                          cv_detector.res, cv_detector.x_far)
                    cv2.imwrite(os.path.join(debug_dir, f"{i:05d}_raw.png"), frame)   # clean, for offline replay
                    if args.save_debug and i % args.save_debug == 0:      # (window frames: raw + bev only, to keep the folder small)
                        cv2.imwrite(os.path.join(debug_dir, f"{i:05d}_cam.png"), cam_img)
                    cv2.imwrite(os.path.join(debug_dir, f"{i:05d}_bev.png"), bev_img)
            proc_ms = (time.perf_counter() - t0) * 1000.0

            if cv_detector is not None and args.max_jump > 0 and detected:
                # Outlier gate: a reading that jumps further than the car can move is noise.
                if accepted is not None and abs(offset - accepted) > args.max_jump and since_accept < 10:
                    detected = False
                    since_accept += 1
                else:
                    accepted, since_accept = offset, 0
            elif cv_detector is not None and not detected:
                since_accept += 1

            if detected:
                # + offset = right of center -> steer left (negative)
                off_f = offset if off_f is None else args.smooth * offset + (1 - args.smooth) * off_f
                hdg_f = (heading_err if hdg_f is None
                         else args.smooth * heading_err + (1 - args.smooth) * hdg_f)
                # + heading = yaw right of the lane -> steer left (negative)
                steer = -pid.compute(off_f, FIXED_DT) - args.kh * hdg_f
                steer = max(-1.0, min(1.0, steer))
                if args.steer_rate > 0:
                    lim = args.steer_rate * FIXED_DT
                    steer = max(prev_cmd - lim, min(prev_cmd + lim, steer))
                last_steer = steer
            else:
                # Lane lost: fade the last command toward straight instead of holding it
                # (holding a large steer is what drives the car off the road).
                last_steer *= 0.8
                steer = last_steer

            disturbance = 0.0
            if args.kick and args.kick_at <= t < args.kick_at + args.kick_dur:
                disturbance = args.kick
            steer = float(max(-1.0, min(1.0, steer + disturbance)))
            prev_cmd = steer - disturbance

            v = vehicle.get_velocity()
            speed = math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2)
            throttle = 0.6 if speed < args.speed else 0.0
            vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer))

            logger.log(timestamp=round(t, 3), lateral_offset=round(offset, 4),
                       heading_error=round(heading_err, 4), lane_width=round(lane_w, 3),
                       detected=detected, processing_time_ms=round(proc_ms, 3),
                       steer=round(steer, 4), speed=round(speed, 3),
                       disturbance=disturbance, gt_offset=round(gt_offset, 4),
                       gt_lane_width=round(gt_lane_w, 3),
                       gt_offset_cam=round(gt_off_cam, 4), gt_heading=round(gt_heading, 4))

            if i % int(1 / FIXED_DT) == 0:
                extra = "" if cv_detector is None else (
                    f" (truth {gt_offset:+.2f}, {'ok' if detected else 'LOST'})")
                print(f"t={t:5.1f}s speed={speed:4.1f} offset={offset:+.2f} m{extra} "
                      f"steer={steer:+.2f}")

    finally:
        if logger is not None:
            logger.close()
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass
        for a in reversed(actors):
            try:
                if isinstance(a, carla.Sensor):
                    a.stop()
                a.destroy()
            except Exception:
                pass

    # Score the run with the repo's own metrics.
    records = load_run(csv_path)
    if not records:
        print("No records logged.")
        return
    lane_w = records[0].get("lane_width")
    s = summarize(records, lane_width=lane_w)
    d = s["deviation"]
    print(f"\n--- {os.path.basename(csv_path)} ({s['n_frames']} frames) ---")
    print(f"lateral deviation: mean {d['mean_m']:.3f} m, max {d['max_m']:.3f} m, "
          f"RMSE {d['rmse_m']:.3f} m")
    if "mean_pct" in d:
        print(f"  as % of lane width: mean {d['mean_pct']:.1f}%, max {d['max_pct']:.1f}%")
    print(f"recovery events (>0.3 m out, back <0.1 m): {len(s['recovery_times_s'])}", end="")
    print(f"  mean {s['mean_recovery_s']:.2f} s" if s["mean_recovery_s"] is not None else "")
    if s["detection_accuracy"] is not None:
        print(f"detection accuracy: {s['detection_accuracy']:.1%}")
    if "gt_offset" in records[0]:
        # The numbers above use the offset the detector REPORTED (0 while the lane is lost).
        # This line uses the map's offset, i.e. where the car really was.
        g = [abs(r["gt_offset"]) for r in records]
        print(f"TRUE deviation from the map lane center: mean {sum(g) / len(g):.3f} m, "
              f"max {max(g):.3f} m, RMSE {(sum(x * x for x in g) / len(g)) ** 0.5:.3f} m")
    if cv_detector is not None and "gt_offset_cam" in records[0]:
        e = [r["lateral_offset"] - r["gt_offset_cam"] for r in records if r["detected"]]
        if e:
            print(f"detector error vs the map at the camera (detected frames): "
                  f"mean {sum(e) / len(e):+.3f} m, RMSE {(sum(x * x for x in e) / len(e)) ** 0.5:.3f} m")


if __name__ == "__main__":
    main()
