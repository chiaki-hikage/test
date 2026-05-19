#!/usr/bin/env python3
"""
Replay PhysicsNeMo egomotion CSV in CARLA and capture front camera images.

Usage:
    python scripts/replay_egomotion_capture_images.py \
        --csv path/to/egomotion.csv \
        --sample-id sample_001 \
        [--output-dir output] \
        [--host localhost] [--port 2000] \
        [--map Town01] \
        [--flip-y]
"""

import argparse
import json
import math
import sys
import time  # [+] for get_image_for_frame deadline
from pathlib import Path
from queue import Empty, Queue

import carla
import pandas as pd

FIXED_DELTA_SECONDS = 0.1
CAM_X = 1.5
CAM_Z = 2.4
WARMUP_TICKS = 3  # [+]

REQUIRED_COLS = {"t_s", "timestamp_us", "x_m", "y_m", "z_m", "yaw_rad",
                 "qx", "qy", "qz", "qw"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, help="Path to PhysicsNeMo egomotion CSV")
    p.add_argument("--sample-id", required=True, help="Sample ID (used as output subdirectory name)")
    p.add_argument("--output-dir", default="output", help="Base output directory (default: output)")
    p.add_argument("--host", default="localhost", help="CARLA server host")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port")
    p.add_argument("--map", default=None, help="CARLA map to load, e.g. Town01 (optional)")
    p.add_argument("--flip-y", action="store_true",
                   help="Negate Y and yaw for coordinate system verification "
                        "(use when PhysicsNeMo uses right-handed Y-left convention)")
    p.add_argument("--image-width", type=int, default=1280, help="Camera image width (default: 1280)")
    p.add_argument("--image-height", type=int, default=720, help="Camera image height (default: 720)")
    p.add_argument("--fov", type=float, default=90.0, help="Camera horizontal FOV in degrees (default: 90)")
    return p.parse_args()


# [+] Drain sensor_queue until an image matching target_frame arrives.
#     Older frames (stale) are discarded. Raises Empty on timeout or frame skip.
def get_image_for_frame(q, target_frame, timeout=2.0):
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Empty(f"Timed out waiting for frame {target_frame}")
        image = q.get(True, max(0.01, remaining))
        if image.frame == target_frame:
            return image
        if image.frame < target_frame:
            continue  # stale frame, discard
        raise Empty(f"Skipped target frame {target_frame}, got {image.frame}")


def make_carla_transform(x_m, y_m, z_m, yaw_rad, flip_y, origin_transform=None):  # [+] origin_transform
    """Convert PhysicsNeMo relative pose to a CARLA world Transform.

    Egomotion pose is treated as relative to origin_transform (CARLA spawn point).
    --flip-y applies right-hand (Y-left) to left-hand (Y-right) conversion.
    """
    if flip_y:
        rel_y = -float(y_m)
        rel_yaw_deg = -math.degrees(float(yaw_rad))
    else:
        rel_y = float(y_m)
        rel_yaw_deg = math.degrees(float(yaw_rad))

    rel_loc = carla.Location(x=float(x_m), y=rel_y, z=float(z_m))

    if origin_transform is not None:  # [+] compose with spawn origin
        world_loc = origin_transform.transform(rel_loc)
        world_yaw = origin_transform.rotation.yaw + rel_yaw_deg
    else:
        world_loc = rel_loc
        world_yaw = rel_yaw_deg

    return carla.Transform(world_loc, carla.Rotation(yaw=world_yaw))


def drain_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break


def setup_output_dirs(base_dir, sample_id):
    out = Path(base_dir) / sample_id
    img_dir = out / "images" / "front"
    ego_dir = out / "ego_motion"
    img_dir.mkdir(parents=True, exist_ok=True)
    ego_dir.mkdir(parents=True, exist_ok=True)
    return out, img_dir, ego_dir


def spawn_vehicle(world):
    lib = world.get_blueprint_library()
    candidates = sorted(lib.filter("vehicle.lincoln.*"), key=lambda bp: bp.id)
    if not candidates:
        candidates = sorted(lib.filter("vehicle.*"), key=lambda bp: bp.id)
    if not candidates:
        sys.exit("[ERROR] No vehicle blueprints found in blueprint library")
    return candidates[0]


def build_camera_blueprint(world, width, height, fov):
    lib = world.get_blueprint_library()
    cam_bp = lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))  # [+] explicit sensor tick
    return cam_bp


def main():
    args = parse_args()

    # ---------- output directories ----------
    out_base, img_dir, ego_dir = setup_output_dirs(args.output_dir, args.sample_id)

    # ---------- load CSV ----------
    csv_path = Path(args.csv)
    if not csv_path.exists():
        sys.exit(f"[ERROR] CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] CSV is missing columns: {sorted(missing)}")
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")

    # ---------- CARLA client ----------
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)

    if args.map:
        print(f"[INFO] Loading map: {args.map}")
        world = client.load_world(args.map)
    else:
        world = client.get_world()

    map_name = world.get_map().name
    print(f"[INFO] Using map: {map_name}")

    # [+] spawn origin: egomotion pose is relative to map's first spawn point
    origin_transform = world.get_map().get_spawn_points()[0]
    print(f"[INFO] Origin transform: {origin_transform.location}")

    original_settings = world.get_settings()
    ego_vehicle = None
    camera = None
    sensor_queue = Queue()
    ego_records = []

    try:
        # ---------- synchronous mode ----------
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        # ---------- spawn ego vehicle ----------
        first = df.iloc[0]
        first_transform = make_carla_transform(
            first.x_m, first.y_m, first.z_m, first.yaw_rad, args.flip_y, origin_transform  # [+]
        )
        vehicle_bp = spawn_vehicle(world)
        ego_vehicle = world.spawn_actor(vehicle_bp, first_transform)
        ego_vehicle.set_autopilot(False)
        print(f"[INFO] Spawned {vehicle_bp.id} at {first_transform.location}")

        # ---------- front camera ----------
        cam_bp = build_camera_blueprint(world, args.image_width, args.image_height, args.fov)
        cam_offset = carla.Transform(
            carla.Location(x=CAM_X, y=0.0, z=CAM_Z),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_offset, attach_to=ego_vehicle)
        camera.listen(lambda data: sensor_queue.put(data))
        print(f"[INFO] Camera attached: {args.image_width}x{args.image_height} fov={args.fov} "
              f"sensor_tick={FIXED_DELTA_SECONDS}")

        # [+] warm-up: 3 ticks to let camera stabilize
        for _ in range(WARMUP_TICKS):
            world.tick()
            drain_queue(sensor_queue)

        # ---------- main loop ----------
        n_rows = len(df)
        for frame_idx, row in enumerate(df.itertuples(index=False)):
            t = make_carla_transform(row.x_m, row.y_m, row.z_m, row.yaw_rad, args.flip_y, origin_transform)  # [+]
            ego_vehicle.set_transform(t)
            target_frame = world.tick()  # [+] capture frame id returned by tick()

            try:
                image = get_image_for_frame(sensor_queue, target_frame)  # [+]
            except Empty as e:
                print(f"[WARN] Frame {frame_idx:06d}: {e}, skipping")
                continue

            img_rel_path = f"images/front/{frame_idx:06d}.png"  # [+] relative path for CSV
            image.save_to_disk(str(out_base / img_rel_path))

            carla_loc = t.location      # [+] derive from composed transform
            carla_yaw_deg = t.rotation.yaw

            ego_records.append({
                "frame": frame_idx,
                "t_s": row.t_s,
                "timestamp_us": row.timestamp_us,
                "x_m": row.x_m,
                "y_m": row.y_m,
                "z_m": row.z_m,
                "yaw_rad": row.yaw_rad,
                "qx": row.qx,
                "qy": row.qy,
                "qz": row.qz,
                "qw": row.qw,
                "image_path": img_rel_path,       # [+]
                "carla_frame": image.frame,        # [+]
                "carla_timestamp": image.timestamp, # [+]
                "carla_x": carla_loc.x,            # [+] world coords after origin compose
                "carla_y": carla_loc.y,
                "carla_z": carla_loc.z,
                "carla_yaw_deg": carla_yaw_deg,
            })

            if frame_idx % 50 == 0 or frame_idx == n_rows - 1:
                print(f"[{frame_idx:06d}/{n_rows-1:06d}] t={row.t_s:.3f}s  carla_frame={image.frame}  "
                      f"carla=({carla_loc.x:.2f}, {carla_loc.y:.2f}, {carla_loc.z:.2f})  "
                      f"yaw={carla_yaw_deg:.1f}deg")

        # ---------- save ego_history.csv ----------
        ego_df = pd.DataFrame(ego_records)
        ego_csv_path = ego_dir / "ego_history.csv"
        ego_df.to_csv(ego_csv_path, index=False)
        print(f"[INFO] ego_history.csv saved: {ego_csv_path}")

        # ---------- save metadata.json ----------
        orig_loc = origin_transform.location   # [+] CARLA spawn origin
        orig_rot = origin_transform.rotation
        metadata = {
            "map": map_name,
            "fps": 1.0 / FIXED_DELTA_SECONDS,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "camera": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov": args.fov,
                "transform": {
                    "x": CAM_X,
                    "y": 0.0,
                    "z": CAM_Z,
                    "roll": 0.0,
                    "pitch": 0.0,
                    "yaw": 0.0,
                },
            },
            "origin_transform": {              # [+] CARLA spawn point (not first egomotion pose)
                "x": orig_loc.x,
                "y": orig_loc.y,
                "z": orig_loc.z,
                "roll": orig_rot.roll,
                "pitch": orig_rot.pitch,
                "yaw_deg": orig_rot.yaw,
            },
            "flip_y": args.flip_y,
        }
        meta_path = out_base / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] metadata.json saved: {meta_path}")

        print(f"\n[DONE] {len(ego_records)} frames saved → {out_base}")

    finally:
        if camera is not None:
            camera.stop()
            camera.destroy()
        if ego_vehicle is not None:
            ego_vehicle.destroy()
        world.apply_settings(original_settings)
        print("[INFO] Cleaned up actors and restored world settings")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
