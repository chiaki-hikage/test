#!/usr/bin/env python3
"""
Open-loop VehicleControl slip behavior experiment for CARLA.

Drives ego vehicle with apply_control() under reduced tire_friction and
adverse weather, logging per-tick telemetry (speed, slip angle proxy, yaw rate)
to CSV and saving front camera images.

Usage:
    python scripts/vehicle_control_slip_experiment.py \
        --sample-id slip_exp_001 \
        --tire-friction 0.5 \
        --throttle 0.7 \
        --steer 0.3 \
        --steer-start-sec 3.0 \
        --duration-sec 15.0 \
        [--expected-map Town10HD_Opt] [--spawn-idx 147]
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

DEFAULT_FIXED_DELTA = 0.05
CAM_X = 1.5
CAM_Z = 2.4
WARMUP_TICKS = 3
EPS = 1e-6


# ──────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample-id", required=True,
                   help="Sample ID for output directory naming")
    p.add_argument("--output-dir", default="output",
                   help="Base output directory (default: output)")
    p.add_argument("--expected-map", default=None,
                   help="If specified, exit with error if server map does not match")
    p.add_argument("--spawn-idx", type=int, default=0,
                   help="Spawn point index (default: 0)")

    # Weather
    p.add_argument("--weather", default="hard_rain_fog",
                   choices=["clear", "hard_rain", "wet_cloudy", "hard_rain_fog"],
                   help="Weather preset (default: hard_rain_fog)")

    # Physics
    p.add_argument("--tire-friction", type=float, default=1.0,
                   help="Tire friction coefficient applied to all wheels (default: 1.0)")
    p.add_argument("--fixed-delta-seconds", type=float, default=DEFAULT_FIXED_DELTA,
                   help=f"Simulation step size in seconds (default: {DEFAULT_FIXED_DELTA})")

    # Open-loop control
    p.add_argument("--throttle", type=float, default=0.5,
                   help="Throttle [0, 1] applied throughout (default: 0.5)")
    p.add_argument("--steer", type=float, default=0.3,
                   help="Steer [-1, 1] applied from --steer-start-sec onward (default: 0.3)")
    p.add_argument("--brake", type=float, default=0.0,
                   help="Brake [0, 1] applied throughout (default: 0.0)")
    p.add_argument("--steer-start-sec", type=float, default=3.0,
                   help="Simulation time (s) at which steer input begins (default: 3.0)")
    p.add_argument("--duration-sec", type=float, default=15.0,
                   help="Total experiment duration in seconds (default: 15.0)")

    # Camera
    p.add_argument("--image-width", type=int, default=800,
                   help="Camera image width (default: 800)")
    p.add_argument("--image-height", type=int, default=450,
                   help="Camera image height (default: 450)")
    p.add_argument("--fov", type=float, default=90.0,
                   help="Camera horizontal FOV in degrees (default: 90)")

    # Connection
    p.add_argument("--host", default="localhost", help="CARLA server host")
    p.add_argument("--port", type=int, default=2000, help="CARLA server port")
    return p.parse_args()


# ──────────────────────────────────────────────
# Weather / vehicle helpers  (shared pattern with preview_route_capture_video.py)
# ──────────────────────────────────────────────

def build_weather(preset):
    if preset == "clear":
        return carla.WeatherParameters.ClearNoon
    if preset == "hard_rain":
        return carla.WeatherParameters.HardRainNoon
    if preset == "wet_cloudy":
        return carla.WeatherParameters.WetCloudyNoon
    # hard_rain_fog: custom heavy-rain + fog
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


def set_tire_friction(vehicle, friction):
    """Apply uniform tire_friction to all wheels via VehiclePhysicsControl."""
    physics = vehicle.get_physics_control()
    wheels = list(physics.wheels)
    for w in wheels:
        w.tire_friction = friction
    physics.wheels = wheels
    vehicle.apply_physics_control(physics)


def freeze_traffic_lights_green(world):
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


def build_camera_blueprint(world, width, height, fov, sensor_tick):
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", str(width))
    cam_bp.set_attribute("image_size_y", str(height))
    cam_bp.set_attribute("fov", str(fov))
    cam_bp.set_attribute("sensor_tick", str(sensor_tick))
    return cam_bp


# ──────────────────────────────────────────────
# Sensor queue helpers  (same pattern as other scripts)
# ──────────────────────────────────────────────

def get_image_for_frame(q, target_frame, timeout=2.0):
    """Return camera image matching target_frame; discard stale frames."""
    deadline = time.monotonic() + timeout
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise Empty(f"Timed out waiting for frame {target_frame}")
        image = q.get(True, max(0.01, remaining))
        if image.frame == target_frame:
            return image
        if image.frame < target_frame:
            continue  # stale, discard
        raise Empty(f"Skipped target frame {target_frame}, got {image.frame}")


def drain_queue(q):
    while not q.empty():
        try:
            q.get_nowait()
        except Empty:
            break


# ──────────────────────────────────────────────
# Telemetry
# ──────────────────────────────────────────────

def get_telemetry(vehicle):
    """Extract vehicle state post-tick for telemetry logging.

    Returns a dict with world/body velocities, slip angle proxy, and yaw rate.
    Note: CARLA get_angular_velocity() returns deg/s; converted to rad/s here.
    """
    transform = vehicle.get_transform()
    velocity = vehicle.get_velocity()
    ang_vel = vehicle.get_angular_velocity()  # deg/s

    loc = transform.location
    yaw_deg = transform.rotation.yaw

    vx_world = velocity.x
    vy_world = velocity.y

    # Project world velocity onto vehicle body axes
    fwd = transform.get_forward_vector()
    rgt = transform.get_right_vector()
    vx_body = vx_world * fwd.x + vy_world * fwd.y
    vy_body = vx_world * rgt.x + vy_world * rgt.y

    speed_mps = math.sqrt(vx_world ** 2 + vy_world ** 2 + velocity.z ** 2)
    # Slip angle proxy: lateral / longitudinal body-frame velocity ratio
    slip_angle_proxy_rad = math.atan2(vy_body, max(vx_body, EPS))
    yaw_rate_radps = math.radians(ang_vel.z)

    return {
        "x": loc.x,
        "y": loc.y,
        "z": loc.z,
        "yaw_deg": yaw_deg,
        "speed_mps": speed_mps,
        "vx_world": vx_world,
        "vy_world": vy_world,
        "vx_body": vx_body,
        "vy_body": vy_body,
        "slip_angle_proxy_rad": slip_angle_proxy_rad,
        "yaw_rate_radps": yaw_rate_radps,
    }


# ──────────────────────────────────────────────
# Video
# ──────────────────────────────────────────────

def make_video(img_dir, output_path, fps):
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
        subprocess.run(cmd, check=True, capture_output=True)
        print(f"[INFO] Video saved: {output_path}")
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg failed: {e.stderr.decode()[:300]}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    args = parse_args()
    fds = args.fixed_delta_seconds
    n_steps = round(args.duration_sec / fds)

    # ---------- output directories ----------
    out_base = Path(args.output_dir) / args.sample_id
    img_dir = out_base / "images" / "front"
    img_dir.mkdir(parents=True, exist_ok=True)

    # ---------- CARLA client (use already-running world) ----------
    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    map_name = world.get_map().name
    print(f"[INFO] Using map: {map_name}")
    if args.expected_map is not None:
        loaded = map_name.split("/")[-1]
        if loaded != args.expected_map:
            sys.exit(f"[ERROR] Map mismatch: server has '{loaded}', "
                     f"expected '{args.expected_map}'")

    # ---------- log existing actors ----------
    existing = world.get_actors()
    n_vehicles = len(list(existing.filter("vehicle.*")))
    n_walkers = len(list(existing.filter("walker.*")))
    print(f"[INFO] Existing actors: {len(existing)} total "
          f"({n_vehicles} vehicles, {n_walkers} walkers)")

    # ---------- resolve spawn point ----------
    spawn_points = world.get_map().get_spawn_points()
    if args.spawn_idx >= len(spawn_points):
        sys.exit(f"[ERROR] spawn_idx={args.spawn_idx} out of range "
                 f"(map has {len(spawn_points)} spawn points)")
    spawn_transform = spawn_points[args.spawn_idx]

    original_settings = world.get_settings()
    ego_vehicle = None
    camera = None
    sensor_queue = Queue()
    records = []

    try:
        # ---------- synchronous mode ----------
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = fds
        world.apply_settings(settings)

        # ---------- traffic lights ----------
        n_tl = freeze_traffic_lights_green(world)
        print(f"[INFO] Frozen {n_tl} traffic lights to Green")

        # ---------- weather ----------
        world.set_weather(build_weather(args.weather))
        print(f"[INFO] Weather set to: {args.weather}")

        # ---------- spawn vehicle ----------
        vehicle_bp = spawn_vehicle(world)
        ego_vehicle = world.try_spawn_actor(vehicle_bp, spawn_transform)
        if ego_vehicle is None:
            sys.exit(
                f"[ERROR] Failed to spawn {vehicle_bp.id} at "
                f"spawn_idx={args.spawn_idx} — position may be occupied"
            )
        ego_vehicle.set_autopilot(False)
        print(f"[INFO] Spawned {vehicle_bp.id} at spawn_idx={args.spawn_idx} "
              f"({spawn_transform.location})")

        # ---------- tire friction ----------
        set_tire_friction(ego_vehicle, args.tire_friction)
        print(f"[INFO] Tire friction set to: {args.tire_friction} (all wheels)")

        # ---------- front camera ----------
        cam_bp = build_camera_blueprint(
            world, args.image_width, args.image_height, args.fov, fds
        )
        cam_offset = carla.Transform(
            carla.Location(x=CAM_X, y=0.0, z=CAM_Z),
            carla.Rotation(pitch=0.0, yaw=0.0, roll=0.0),
        )
        camera = world.spawn_actor(cam_bp, cam_offset, attach_to=ego_vehicle)
        camera.listen(lambda data: sensor_queue.put(data))
        print(f"[INFO] Camera attached: {args.image_width}x{args.image_height} "
              f"fov={args.fov} sensor_tick={fds}")

        # ---------- warm-up (idle control) ----------
        for _ in range(WARMUP_TICKS):
            ego_vehicle.apply_control(carla.VehicleControl())
            world.tick()
            drain_queue(sensor_queue)

        # ---------- main loop ----------
        log_interval = max(1, n_steps // 10)
        print(f"[INFO] Running {n_steps} steps ({args.duration_sec:.1f}s at "
              f"{1.0/fds:.0f}Hz), steer={args.steer} from t={args.steer_start_sec:.1f}s")

        saved = 0
        skipped = 0

        for step in range(n_steps):
            t_s = step * fds

            # Open-loop control: steer delayed until steer_start_sec
            steer = args.steer if t_s >= args.steer_start_sec else 0.0
            ctrl = carla.VehicleControl(
                throttle=float(args.throttle),
                steer=float(steer),
                brake=float(args.brake),
                hand_brake=False,
                reverse=False,
                manual_gear_shift=False,
            )
            ego_vehicle.apply_control(ctrl)
            target_frame = world.tick()

            # Post-tick telemetry
            tel = get_telemetry(ego_vehicle)

            # Camera image
            try:
                image = get_image_for_frame(sensor_queue, target_frame)
                image.save_to_disk(str(img_dir / f"{step:06d}.png"))
                saved += 1
            except Empty as e:
                print(f"[WARN] Step {step:06d}: {e}, skipping image")
                skipped += 1

            records.append({
                "frame": target_frame,
                "t_s": round(t_s, 6),
                "x": tel["x"],
                "y": tel["y"],
                "z": tel["z"],
                "yaw_deg": tel["yaw_deg"],
                "speed_mps": tel["speed_mps"],
                "vx_world": tel["vx_world"],
                "vy_world": tel["vy_world"],
                "vx_body": tel["vx_body"],
                "vy_body": tel["vy_body"],
                "slip_angle_proxy_rad": tel["slip_angle_proxy_rad"],
                "yaw_rate_radps": tel["yaw_rate_radps"],
                "throttle": ctrl.throttle,
                "steer": ctrl.steer,
                "brake": ctrl.brake,
                "tire_friction": args.tire_friction,
            })

            if step % log_interval == 0 or step == n_steps - 1:
                print(f"[{step:06d}/{n_steps-1:06d}] "
                      f"t={t_s:.2f}s  "
                      f"speed={tel['speed_mps']:.2f}m/s  "
                      f"slip={math.degrees(tel['slip_angle_proxy_rad']):.1f}deg  "
                      f"yaw_rate={math.degrees(tel['yaw_rate_radps']):.1f}deg/s  "
                      f"steer={ctrl.steer:.2f}")

        print(f"[INFO] Captured {saved} images, skipped {skipped}")

        # ---------- telemetry CSV ----------
        telem_path = out_base / "telemetry.csv"
        pd.DataFrame(records).to_csv(telem_path, index=False)
        print(f"[INFO] Telemetry saved: {telem_path}")

        # ---------- metadata.json ----------
        metadata = {
            "map": map_name,
            "spawn_idx": args.spawn_idx,
            "fixed_delta_seconds": fds,
            "duration_sec": args.duration_sec,
            "n_steps": n_steps,
            "saved_frames": saved,
            "skipped_frames": skipped,
            "weather": args.weather,
            "tire_friction": args.tire_friction,
            "control": {
                "mode": "open_loop",
                "throttle": args.throttle,
                "steer": args.steer,
                "brake": args.brake,
                "steer_start_sec": args.steer_start_sec,
            },
            "camera": {
                "image_width": args.image_width,
                "image_height": args.image_height,
                "fov": args.fov,
                "transform": {
                    "x": CAM_X, "y": 0.0, "z": CAM_Z,
                    "roll": 0.0, "pitch": 0.0, "yaw": 0.0,
                },
            },
        }
        meta_path = out_base / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"[INFO] metadata.json saved: {meta_path}")

        # ---------- ffmpeg video ----------
        fps = int(round(1.0 / fds))
        make_video(img_dir, out_base / "preview_front.mp4", fps=fps)

        print(f"\n[DONE] {n_steps} steps, {saved} frames saved → {out_base}")

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
