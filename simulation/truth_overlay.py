"""
simulation/truth_overlay.py

Draw the CARLA map's own idea of the lane (center and both edges) on the detector's debug
images, so you can SEE whether the camera or the map is right where they disagree.

Pure geometry + OpenCV (no CARLA import), so it can be tested anywhere. run_lane_keeping.py
gathers the numbers from CARLA and passes them in.

Frames (same as perception/classical_lane.py): x = forward from the camera (m),
y = to the right (m). The camera sits CAM_X metres ahead of the vehicle origin.
"""
import cv2
import numpy as np

GREEN = (0, 255, 0)       # map lane center
ORANGE = (0, 165, 255)    # map lane edges


def lane_points_in_camera_frame(veh_xy, fwd_xy, right_xy, waypoints, cam_x, x_min=4.5, x_max=26.0):
    """waypoints: list of (x, y, right_x, right_y, lane_width) in world coordinates, ahead of
    the car. Returns {"center": [(x, y), ...], "left": [...], "right": [...]} in the camera frame
    (x forward, y right), keeping only points the bird's-eye view can show (x_min..x_max)."""
    out = {"center": [], "left": [], "right": []}
    for wx, wy, rx, ry, width in waypoints:
        half = 0.5 * width
        for key, sign in (("center", 0.0), ("left", -1.0), ("right", 1.0)):
            px, py = wx + rx * sign * half, wy + ry * sign * half
            dx, dy = px - veh_xy[0], py - veh_xy[1]
            x = dx * fwd_xy[0] + dy * fwd_xy[1] - cam_x
            y = dx * right_xy[0] + dy * right_xy[1]
            if x_min <= x <= x_max:
                out[key].append((x, y))
    return out


def draw_on_bev(bev, pts, y_half, res, x_far):
    """Paint the map lane on the bird's-eye debug image (in place)."""
    h, w = bev.shape[:2]
    for key, color, r in (("left", ORANGE, 2), ("right", ORANGE, 2), ("center", GREEN, 3)):
        for x, y in pts[key]:
            c, rw = int(round((y + y_half) / res)), int(round((x_far - x) / res))
            if 0 <= c < w and 0 <= rw < h:
                cv2.circle(bev, (c, rw), r, color, -1)
    cv2.putText(bev, "green=map center", (4, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.4, GREEN, 1)
    cv2.putText(bev, "orange=map edges", (4, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, ORANGE, 1)
    return bev


def draw_on_camera(img, pts, ground_to_pixel):
    """Paint the map lane on the camera image (in place). ground_to_pixel(x, y) -> (u, v)."""
    h, w = img.shape[:2]
    for key, color, r in (("left", ORANGE, 3), ("right", ORANGE, 3), ("center", GREEN, 4)):
        for x, y in pts[key]:
            u, v = ground_to_pixel(x, y)
            if 0 <= u < w and 0 <= v < h:
                cv2.circle(img, (int(u), int(v)), r, color, -1)
    return img
