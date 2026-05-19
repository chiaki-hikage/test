#!/usr/bin/env python3
"""
Preview a CARLA route by replaying it with a front camera and saving images/video.

Route CSV columns (x_m, y_m, z_m, yaw_rad) are treated as CARLA world absolute
coordinates — no origin_transform composition is applied.

Usage:
    python scripts/preview_route_capture_video.py \
        --route-csv route_search/Town10HD_Opt/routes/Town10HD_Opt_sp147_normal.csv \
        --sample-id Town10HD_Opt_sp147 \
        [--output-dir output] \
        [--max-frames 200] \
        [--image-width 800] [--image-height 450] [--fov 90] \
        [--host localhost] [--port 2000]
"""

import argparse
import json
import math
import shutil
import subprocess
import sys
import time
from pathlib import Path
from queue import Empty, Queue

import carla
import pandas as pd

FIXED_DELTA_SECONDS = 0.1
CAM_X = 1.5
CAM_Z = 2.4
WARMUP_TICKS = 3
DEFAULT_MAX_FRAMES = 200

REQUIRED_COLS = {"x_m", "y_m", "z_m", "yaw_rad"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route-csv", required=True,
                   help="Route CSV with columns: s_m, x_m, y_m, z_m, yaw_rad, curvature_1pm")
    p.add_argument("--sample-id", required=True,
                   help="Sample ID for output directory naming")
    p.add_argument("--output-dir", default="output",
                   help="Base output directory (default: output)")
    p.add_argument("--max-frames", type=int, default=DEFAULT_MAX_FRAMES,
                   help=f"Maximum frames to capture, default: {DEFAULT_MAX_FRAMES} (= 20s at 10Hz)")
    p.add_argument("--image-width", type=int, default=800,
                   help="Camera image width (default: 800)")
    p.add_argument("--image-height", type=int, default=450,
                   help="Camera image height (default: 450)")
    p.add_argument("--fov", type=float, default=90.0,
                   help="Camera horizontal FOV in degrees (default: 90)")
    p.add_argument("--weather", default="clear",
                   choices=["clear", "hard_rain", "wet_cloudy", "hard_rain_fog"],
                   help="Weather preset: clear / hard_rain / wet_cloudy / hard_rain_fog (default: clear)")
    p.add_argument("--z-offset", type=float, default=0.3,
                   help="Z offset (m) added to route CSV z_m values (default: 0.3)")
    p.add_argument("--expected-map", default=None,
                   help="If specified, exit with error if server map does not match")
    p.add_argument("--host", default="localhost", help="CARLA server host")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port")
    return p.parse_args()


def build_weather(preset):
    """Return a carla.WeatherParameters for the given preset name."""
    if preset == "clear":
        return carla.WeatherParameters.ClearNoon
    if preset == "hard_rain":
        return carla.WeatherParameters.HardRainNoon
    if preset == "wet_cloudy":
        return carla.WeatherParameters.WetCloudyNoon
    # hard_rain_fog: custom heavy-rain + fog parameters
    return carla.WeatherParameters(
        cloudiness=100,
        precipitation=100,
        precipitation_deposits=100,
        wetness=100,
        fog_density=35,
        fog_distance=15,
        fog_falloff=1,
        wind_intensity=80,
        sun_altitude_angle=20,
    )


def get_image_for_frame(q, target_frame, timeout=2.0):
    """Return the camera image matching target_frame, discarding older frames.

    Raises Empty on timeout or if a future frame arrives first (frame skip).
    """
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


def drain_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break


def freeze_traffic_lights_green(world):
    """Set all traffic lights in the world to green and freeze them."""
    tl_list = list(world.get_actors().filter("traffic.traffic_light"))
    for tl in tl_list:
        tl.set_state(carla.TrafficLightState.Green)
        tl.freeze(True)
    return len(tl_list)


def spawn_vehicle(world):
    lib = world.get_blueprint_library()
    candidates = sorted(lib.filter("vehicle.lincoln.*"), key=lambda bp: bp.id)
    if not candidates:
        candidates = sorted(lib.filter("vehicle.*"), key=lambda bp: bp.id)
    if not candidates:
        sys.exit("[ERROR] No vehicle blueprints found in blueprint library")
    return candidates[0]


def build_camera_blueprint(world, width, height, fov):
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", str(FIXED_DELTA_SECONDS))
    return cam_bp


def make_video(img_dir, output_path, fps):
    """Create MP4 with ffmpeg if available; silently skips if ffmpeg is absent."""
    if shutil.which("ffmpeg") is None:
        print("[INFO] ffmpeg not found, skipping video creation")
        return
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(fps),
        "-i", str(img_dir / "%06d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, check=True, capture_output=True)
        print(f"[INFO] Video saved: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed: {e.stderr.decode()[:300]}")


def main():
    args = parse_args()

    # ---------- output directories ----------
    out_base = Path(args.output_dir) / args.sample_id
    img_dir = out_base / "images" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ---------- load route CSV ----------
    route_path = Path(args.route_csv)
    if not route_path.exists():
        sys.exit(f"[ERROR] Route CSV not found: {route_path}")
    df = pd.read_csv(route_path)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Route CSV missing columns: {sorted(missing)}")
    df = df.head(args.max_frames)
    n_frames = len(df)
    duration_sec = n_frames * FIXED_DELTA_SECONDS
    print(f"[INFO] Route CSV: {n_frames} rows → {n_frames} frames ({duration_sec:.1f}s)")

    # ---------- CARLA client (use already-running world) ----------
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    map_name = world.get_map().name
    print(f"[INFO] Using map: {map_name}")
    if args.expected_map is not None:
        loaded = map_name.split("/")[-1]
        if loaded != args.expected_map:
            sys.exit(f"[ERROR] Map mismatch: server has '{loaded}', expected '{args.expected_map}'")

    # ---------- log existing actors ----------
    existing = world.get_actors()
    n_vehicles = len(list(existing.filter("vehicle.*")))
    n_walkers = len(list(existing.filter("walker.*")))
    print(f"[INFO] Existing actors: {len(existing)} total "
          f"({n_vehicles} vehicles, {n_walkers} walkers)")

    original_settings = world.get_settings()
    ego_vehicle = None
    camera = None
    sensor_queue = Queue()

    try:
        # ---------- synchronous mode ----------
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = FIXED_DELTA_SECONDS
        world.apply_settings(settings)

        # ---------- freeze all traffic lights to green ----------
        n_tl = freeze_traffic_lights_green(world)
        print(f"[INFO] Frozen {n_tl} traffic lights to Green")

        # ---------- weather ----------
        world.set_weather(build_weather(args.weather))
        print(f"[INFO] Weather set to: {args.weather}")

        # ---------- spawn ego vehicle at first route pose ----------
        first = df.iloc[0]
        first_transform = carla.Transform(
            carla.Location(x=float(first.x_m), y=float(first.y_m),
                           z=float(first.z_m) + args.z_offset),
            carla.Rotation(yaw=math.degrees(float(first.yaw_rad))),
        )
        vehicle_bp = spawn_vehicle(world)
        ego_vehicle = world.try_spawn_actor(vehicle_bp, first_transform)
        if ego_vehicle is None:
            sys.exit(
                f"[ERROR] Failed to spawn {vehicle_bp.id} at "
                f"({first.x_m:.2f}, {first.y_m:.2f}, {float(first.z_m) + args.z_offset:.2f}) — "
                "position may be occupied or outside map bounds"
            )
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
        print(f"[INFO] Camera attached: {args.image_width}x{args.image_height} "
              f"fov={args.fov} sensor_tick={FIXED_DELTA_SECONDS}")

        # ---------- warm-up ----------
        for _ in range(WARMUP_TICKS):
            world.tick()
            drain_queue(sensor_queue)

        # ---------- main loop ----------
        saved = 0
        skipped = 0
        for frame_idx, row in enumerate(df.itertuples(index=False)):
            t = carla.Transform(
                carla.Location(x=float(row.x_m), y=float(row.y_m),
                               z=float(row.z_m) + args.z_offset),
                carla.Rotation(yaw=math.degrees(float(row.yaw_rad))),
            )
            ego_vehicle.set_transform(t)
            target_frame = world.tick()

            try:
                image = get_image_for_frame(sensor_queue, target_frame)
            except Empty as e:
                print(f"[WARN] Frame {frame_idx:06d}: {e}, skipping")
                skipped += 1
                continue

            image.save_to_disk(str(img_dir / f"{frame_idx:06d}.png"))
            saved += 1

            if frame_idx % 50 == 0 or frame_idx == n_frames - 1:
                s_m = getattr(row, "s_m", frame_idx * FIXED_DELTA_SECONDS)
                print(f"[{frame_idx:06d}/{n_frames-1:06d}] "
                      f"s={s_m:.1f}m  carla_frame={image.frame}  "
                      f"pos=({row.x_m:.1f}, {row.y_m:.1f}, {row.z_m:.1f})  "
                      f"yaw={math.degrees(row.yaw_rad):.1f}deg")

        print(f"[INFO] Captured {saved} frames, skipped {skipped}")

        # ---------- metadata.json ----------
        metadata = {
            "map": map_name,
            "route_csv": str(route_path),
            "fps": 1.0 / FIXED_DELTA_SECONDS,
            "fixed_delta_seconds": FIXED_DELTA_SECONDS,
            "duration_sec": duration_sec,
            "route_rows": n_frames,
            "saved_frames": saved,
            "skipped_frames": skipped,
            "warmup_ticks": WARMUP_TICKS,
            "n_frames": saved,
            "camera": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov": args.fov,
                "transform": {
                    "x": CAM_X, "y": 0.0, "z": CAM_Z,
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                },
            },
            "coordinate_mode": "absolute_route_csv",
            "weather": args.weather,
            "traffic_light_freeze": True,
        }
        meta_path = out_base / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] metadata.json saved: {meta_path}")

        # ---------- ffmpeg video ----------
        make_video(img_dir, out_base / "preview_front.mp4",
                   fps=int(1.0 / FIXED_DELTA_SECONDS))

        print(f"\n[DONE] {saved} frames saved → {out_base}")

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
