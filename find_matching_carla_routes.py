#!/usr/bin/env python3
"""
Find CARLA map route segments whose curvature profile best matches a target road profile.

Usage:
    python scripts/find_matching_carla_routes.py \
        --target-road-profile path/to/road_profile.csv \
        --maps Town01 Town03 Town04 \
        --distance 200 \
        --ds 1.0 \
        --top-k 10 \
        --output-dir output/route_search
"""

import argparse
import math
import sys
from pathlib import Path

import carla
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target-road-profile", required=True,
                   help="Target road profile CSV with columns: s_m, curvature_1pm")
    p.add_argument("--maps", nargs="+", default=["Town01"],
                   help="CARLA map names to search (default: Town01)")
    p.add_argument("--distance", type=float, default=200.0,
                   help="Minimum route extraction length in meters (default: 200)")
    p.add_argument("--ds", type=float, default=1.0,
                   help="Waypoint sampling interval in meters (default: 1.0)")
    p.add_argument("--top-k", type=int, default=10,
                   help="Number of top candidates to save (default: 10)")
    p.add_argument("--output-dir", default="output/route_search",
                   help="Output directory (default: output/route_search)")
    p.add_argument("--host", default="localhost", help="CARLA server host")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port")
    return p.parse_args()


def _normalize_angle(angle):
    """Normalize a single angle to (-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi


def _normalize_angle_vec(angles):
    """Normalize a numpy array of angles to (-pi, pi]."""
    return (angles + np.pi) % (2 * np.pi) - np.pi


def build_route(carla_map, spawn_transform, target_distance, ds):
    """Walk waypoints from spawn_transform for at least target_distance meters.

    At junctions, always takes next_wps[0] and emits a warning.
    Returns list of carla.Waypoint; empty list if the start has no drivable lane.
    """
    wp = carla_map.get_waypoint(
        spawn_transform.location,
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if wp is None:
        return []

    wps = [wp]
    total_s = 0.0

    while total_s < target_distance:
        next_wps = wp.next(ds)
        if not next_wps:
            break
        if len(next_wps) > 1:
            print(f"[WARN]   Branch at ({wp.transform.location.x:.1f}, "
                  f"{wp.transform.location.y:.1f}): "
                  f"{len(next_wps)} options, taking [0]")
        wp = next_wps[0]
        wps.append(wp)
        total_s += ds

    return wps


def compute_route_profile(wps):
    """Compute s_m, x_m, y_m, z_m, yaw_rad, curvature_1pm from a waypoint list.

    Curvature uses central differences for interior points and
    forward/backward differences at endpoints. Angle wrap is handled via
    normalization to (-pi, pi].
    """
    xs = np.array([wp.transform.location.x for wp in wps], dtype=float)
    ys = np.array([wp.transform.location.y for wp in wps], dtype=float)
    zs = np.array([wp.transform.location.z for wp in wps], dtype=float)
    yaws_rad = np.radians([wp.transform.rotation.yaw for wp in wps])

    # Cumulative arc length from actual inter-waypoint distances
    deltas = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)
    s_vals = np.concatenate([[0.0], np.cumsum(deltas)])

    n = len(wps)
    curvatures = np.zeros(n)

    if n >= 3:
        # Central differences for interior points (vectorized)
        dyaw_cd = _normalize_angle_vec(yaws_rad[2:] - yaws_rad[:-2])
        ds_cd = s_vals[2:] - s_vals[:-2]
        curvatures[1:-1] = np.where(ds_cd > 1e-6, dyaw_cd / ds_cd, 0.0)

    if n >= 2:
        dyaw0 = _normalize_angle(float(yaws_rad[1] - yaws_rad[0]))
        ds0 = float(s_vals[1] - s_vals[0])
        curvatures[0] = dyaw0 / ds0 if ds0 > 1e-6 else 0.0

        dyaw_last = _normalize_angle(float(yaws_rad[-1] - yaws_rad[-2]))
        ds_last = float(s_vals[-1] - s_vals[-2])
        curvatures[-1] = dyaw_last / ds_last if ds_last > 1e-6 else 0.0

    return s_vals, xs, ys, zs, yaws_rad, curvatures


def compute_rmse(target_s, target_curv, route_s, route_curv):
    """Interpolate route_curv at target_s positions and return RMSE."""
    route_curv_interp = np.interp(target_s, route_s, route_curv)
    return float(np.sqrt(np.mean((target_curv - route_curv_interp) ** 2)))


def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    routes_dir = out_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load target profile ----------
    profile_path = Path(args.target_road_profile)
    if not profile_path.exists():
        sys.exit(f"[ERROR] Target profile not found: {profile_path}")
    profile_df = pd.read_csv(profile_path)
    missing = {"s_m", "curvature_1pm"} - set(profile_df.columns)
    if missing:
        sys.exit(f"[ERROR] Target profile missing columns: {sorted(missing)}")

    target_s = profile_df["s_m"].to_numpy(dtype=float)
    target_s = target_s - target_s[0]  # normalize to start at 0
    target_curv = profile_df["curvature_1pm"].to_numpy(dtype=float)
    target_end_s = float(target_s[-1])
    print(f"[INFO] Target profile: {len(target_s)} points, "
          f"s=[0, {target_end_s:.1f}]m, "
          f"curvature=[{target_curv.min():.4f}, {target_curv.max():.4f}] 1/m")

    # Route must cover the full target length; add a small margin
    route_distance = max(args.distance, target_end_s) + args.ds * 2

    # ---------- CARLA client ----------
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)

    all_candidates = []

    world = client.get_world()
    carla_map = world.get_map()
    loaded_map_name = carla_map.name.split("/")[-1]
    print(f"[INFO] Using already loaded map: {carla_map.name}")

    # args.maps が指定されていて、期待mapと違う場合は警告だけ出す
    if args.maps and loaded_map_name not in args.maps:
        print(f"[WARN] Loaded map is {loaded_map_name}, but requested maps={args.maps}")

    # Disable rendering for faster waypoint queries
    settings = world.get_settings()
    settings.no_rendering_mode = True
    world.apply_settings(settings)

    carla_map = world.get_map()
    spawn_points = carla_map.get_spawn_points()
    print(f"[INFO] {len(spawn_points)} spawn points")

    valid_count = 0
    skip_count = 0

    for sp_idx, sp in enumerate(spawn_points):
        wps = build_route(carla_map, sp, route_distance, args.ds)

        if len(wps) < 2:
            skip_count += 1
            continue

        s_vals, xs, ys, zs, yaw_rads, curvs = compute_route_profile(wps)

        if s_vals[-1] < target_end_s:
            skip_count += 1
            continue

        rmse_n = compute_rmse(target_s, target_curv, s_vals, curvs)
        rmse_f = compute_rmse(target_s, target_curv, s_vals, -curvs)
        flipped = rmse_f < rmse_n
        score = min(rmse_n, rmse_f)

        all_candidates.append({
            "map": loaded_map_name,
            "spawn_idx": sp_idx,
            "spawn_x": float(sp.location.x),
            "spawn_y": float(sp.location.y),
            "spawn_z": float(sp.location.z),
            "rmse": score,
            "rmse_normal": rmse_n,
            "rmse_flipped": rmse_f,
            "flipped": flipped,
            "route_length_m": float(s_vals[-1]),
            "n_waypoints": len(wps),
            "s_vals": s_vals,
            "xs": xs,
            "ys": ys,
            "zs": zs,
            "yaw_rads": yaw_rads,
            "curvs": curvs,
        })
        valid_count += 1

    print(f"[INFO] {loaded_map_name}: {valid_count} valid routes, {skip_count} skipped")

    if not all_candidates:
        sys.exit("[ERROR] No valid route candidates found. "
                 "Try increasing --distance or checking the maps.")

    # ---------- sort globally and select top-k ----------
    all_candidates.sort(key=lambda c: c["rmse"])
    top_k = all_candidates[: args.top_k]

    print(f"\n[INFO] Top {len(top_k)} candidates (of {len(all_candidates)} total):")

    summary_rows = []
    for rank, cand in enumerate(top_k):
        direction = "flipped" if cand["flipped"] else "normal"
        route_csv_name = f"{cand['map']}_sp{cand['spawn_idx']:03d}_{direction}.csv"
        route_csv_path = routes_dir / route_csv_name

        # Apply curvature sign flip for the saved CSV
        saved_curvs = -cand["curvs"] if cand["flipped"] else cand["curvs"]

        route_df = pd.DataFrame({
            "s_m": cand["s_vals"],
            "x_m": cand["xs"],
            "y_m": cand["ys"],
            "z_m": cand["zs"],
            "yaw_rad": cand["yaw_rads"],
            "curvature_1pm": saved_curvs,
        })
        route_df.to_csv(route_csv_path, index=False)

        print(f"  [{rank + 1:2d}] {cand['map']}  spawn={cand['spawn_idx']:3d}  "
              f"rmse={cand['rmse']:.6f}  {direction}  "
              f"len={cand['route_length_m']:.1f}m  wps={cand['n_waypoints']}")

        summary_rows.append({
            "rank": rank + 1,
            "map": cand["map"],
            "spawn_idx": cand["spawn_idx"],
            "spawn_x": cand["spawn_x"],
            "spawn_y": cand["spawn_y"],
            "spawn_z": cand["spawn_z"],
            "rmse": cand["rmse"],
            "rmse_normal": cand["rmse_normal"],
            "rmse_flipped": cand["rmse_flipped"],
            "flipped": cand["flipped"],
            "route_length_m": cand["route_length_m"],
            "n_waypoints": cand["n_waypoints"],
            "route_csv": str(route_csv_path.relative_to(out_dir)),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "route_candidates.csv"
    summary_df.to_csv(summary_path, index=False)

    print(f"\n[INFO] Summary saved: {summary_path}")
    print(f"[INFO] Route CSVs saved in: {routes_dir}")
    print(f"\n[DONE] {len(top_k)} candidates saved → {out_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
