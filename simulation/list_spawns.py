"""
simulation/list_spawns.py

List CARLA spawn points that suit a camera lane detector: a straight stretch of road with NO
junction in the first ~60 m ahead (intersections have no lane markings to follow, and their
crosswalk / stop-line paint confuses a classical detector).

    python simulation/list_spawns.py                 # print the best candidates
    python simulation/list_spawns.py --draw          # also label spawn indices in the 3D view
    python simulation/list_spawns.py --town Town04   # another map (Town04 has long highway stretches)

Then run:  python simulation/run_lane_keeping.py --detector cv --spawn <index>

CARLA must be running. This only reads the map; it doesn't spawn anything.
"""
import argparse
import math

import carla


def yaw_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ahead", type=float, default=60.0, help="meters of road that must be clear")
    p.add_argument("--max-turn", type=float, default=8.0, help="max heading change over that stretch (deg)")
    p.add_argument("--top", type=int, default=15)
    p.add_argument("--town", default=None)
    p.add_argument("--draw", action="store_true", help="label spawn indices in the simulator view")
    args = p.parse_args()

    client = carla.Client("localhost", 2000)
    client.set_timeout(20.0)
    world = client.load_world(args.town) if args.town else client.get_world()
    cmap = world.get_map()
    spawns = cmap.get_spawn_points()

    rows = []
    for i, sp in enumerate(spawns):
        wp = cmap.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue
        ok, cur, worst_turn, d = True, wp, 0.0, 0.0
        while d < args.ahead:
            nxt = cur.next(5.0)
            if not nxt:
                ok = False
                break
            cur = nxt[0]
            d += 5.0
            if cur.is_junction:
                ok = False
                break
            worst_turn = max(worst_turn, abs(yaw_diff(cur.transform.rotation.yaw, wp.transform.rotation.yaw)))
        if not ok or worst_turn > args.max_turn:
            continue
        rows.append((worst_turn, i, sp, wp))

    rows.sort(key=lambda r: r[0])
    print(f"{len(rows)} of {len(spawns)} spawn points have {args.ahead:.0f} m of clear, near-straight road ahead.\n")
    print(f"{'spawn':>5}  {'x':>8} {'y':>8}  {'turn_deg':>8}  left marking / right marking")
    for turn, i, sp, wp in rows[:args.top]:
        lm, rm = wp.left_lane_marking, wp.right_lane_marking
        print(f"{i:>5}  {sp.location.x:8.1f} {sp.location.y:8.1f}  {turn:8.1f}  "
              f"{lm.type if lm else '-'} {lm.color if lm else ''} / {rm.type if rm else '-'} {rm.color if rm else ''}")

    if args.draw:
        for turn, i, sp, wp in rows:
            world.debug.draw_string(sp.location + carla.Location(z=2.0), str(i), draw_shadow=False,
                                    color=carla.Color(0, 255, 0), life_time=120.0)
        print("\nGreen numbers drawn in the simulator for 2 minutes (fly around with the mouse + WASD).")


if __name__ == "__main__":
    main()
