#!/usr/bin/env python3
"""
Generate CARLA Leaderboard route XML from a CARLA route CSV.

Reads a route CSV (columns: s_m, x_m, y_m, z_m, yaw_rad, curvature_1pm) and
produces:
  - A Leaderboard-format route XML  (matching routes_devtest.xml style)
  - An empty scenario definition JSON
  - A metadata JSON for evaluation asset tracking

XML format matches CARLA Leaderboard 2.x (routes_devtest.xml / routes_training.xml):

    <routes>
       <route id="ROUTE_ID" town="TOWN">
          <weathers>
             <weather route_percentage="0"
                cloudiness="..." .../>
             <weather route_percentage="100"
                cloudiness="..." .../>
          </weathers>
          <waypoints>
             <position x="..." y="..." z="..."/>
             ...
          </waypoints>
          <scenarios>
          </scenarios>
       </route>
    </routes>

Example:
    python scripts/create_leaderboard_route_xml_from_csv.py \\
        --route-csv carla_maps/Town10HD_Opt_sp147_normal.csv \\
        --leaderboard-root ../leaderboard \\
        --town Town10HD_Opt \\
        --route-id 0 \\
        --route-name town10_sp147_normal \\
        --weather hard_rain_fog \\
        --waypoint-spacing-m 10 \\
        --route-z-mode fixed \\
        --fixed-z 0.5 \\
        --seed 0 \\
        --output-xml evaluation_assets/routes/routes_town10_sp147_regression.xml \\
        --output-metadata evaluation_assets/routes/routes_town10_sp147_regression.metadata.json \\
        --output-scenario evaluation_assets/routes/scenarios_town10_sp147_empty.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

# ---------------------------------------------------------------------------
# Weather presets
# ---------------------------------------------------------------------------
WEATHER_PRESETS = {
    "clear": dict(
        cloudiness=5.0, precipitation=0.0, precipitation_deposits=0.0,
        wetness=0.0, wind_intensity=10.0, sun_azimuth_angle=-1.0,
        sun_altitude_angle=90.0, fog_density=2.0,
    ),
    "hard_rain": dict(
        cloudiness=100.0, precipitation=100.0, precipitation_deposits=100.0,
        wetness=100.0, wind_intensity=80.0, sun_azimuth_angle=-1.0,
        sun_altitude_angle=15.0, fog_density=2.0,
    ),
    "wet_cloudy": dict(
        cloudiness=80.0, precipitation=30.0, precipitation_deposits=60.0,
        wetness=80.0, wind_intensity=40.0, sun_azimuth_angle=-1.0,
        sun_altitude_angle=45.0, fog_density=5.0,
    ),
    "hard_rain_fog": dict(
        cloudiness=100.0, precipitation=100.0, precipitation_deposits=100.0,
        wetness=100.0, wind_intensity=80.0, sun_azimuth_angle=0.0,
        sun_altitude_angle=20.0, fog_density=35.0,
        fog_distance=15.0, fog_falloff=1.0,
    ),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--route-csv", required=True,
                   help="CARLA route CSV (columns: s_m, x_m, y_m, z_m, yaw_rad, curvature_1pm)")
    p.add_argument("--leaderboard-root", default="../leaderboard",
                   help="Path to leaderboard repository root (default: ../leaderboard)")
    p.add_argument("--town", required=True,
                   help="CARLA map/town name (e.g. Town10HD_Opt). "
                        "Must match the map loaded by the CARLA server.")
    p.add_argument("--route-id", default="0",
                   help="Numeric route ID written to the XML route element (default: 0)")
    p.add_argument("--route-name", default="town10_sp147_normal",
                   help="Human-readable route name stored in metadata only "
                        "(default: town10_sp147_normal)")
    p.add_argument("--route-z-mode", default="fixed", choices=["csv", "fixed"],
                   help="How to set waypoint z in the XML: "
                        "'fixed' uses --fixed-z for all waypoints (default), "
                        "'csv' uses z_m from the route CSV.")
    p.add_argument("--fixed-z", type=float, default=0.5,
                   help="Fixed z value [m] for all waypoints when --route-z-mode=fixed "
                        "(default: 0.5)")
    p.add_argument("--weather", default="hard_rain_fog",
                   choices=list(WEATHER_PRESETS.keys()),
                   help="Weather preset (default: hard_rain_fog)")
    p.add_argument("--waypoint-spacing-m", type=float, default=10.0,
                   help="Arc-length spacing between waypoints [m] (default: 10.0). "
                        "First and last waypoints are always included.")
    p.add_argument("--seed", type=int, default=0,
                   help="Evaluation seed recorded in metadata (default: 0)")
    p.add_argument("--carla-version", default="0.9.16",
                   help="CARLA version recorded in metadata (default: 0.9.16)")
    p.add_argument("--output-xml", required=True,
                   help="Output route XML path")
    p.add_argument("--output-metadata", required=True,
                   help="Output metadata JSON path")
    p.add_argument("--output-scenario", required=True,
                   help="Output scenario definition JSON path")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Waypoint subsampling
# ---------------------------------------------------------------------------
def subsample_waypoints(df: pd.DataFrame, spacing_m: float) -> pd.DataFrame:
    """Keep waypoints at >= spacing_m arc-length apart. Always include first/last."""
    if len(df) <= 2:
        return df.copy()

    keep = [0]
    last_s = float(df["s_m"].iloc[0])

    for i in range(1, len(df) - 1):
        if float(df["s_m"].iloc[i]) - last_s >= spacing_m:
            keep.append(i)
            last_s = float(df["s_m"].iloc[i])

    last_idx = len(df) - 1
    if keep[-1] != last_idx:
        keep.append(last_idx)

    return df.iloc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# XML generation (matches routes_devtest.xml indentation style)
# ---------------------------------------------------------------------------
def fmt(v, decimals=4):
    """Format a float to fixed decimals, stripping trailing zeros."""
    s = f"{float(v):.{decimals}f}"
    # keep at least one decimal
    s = s.rstrip("0").rstrip(".")
    if "." not in s:
        s += ".0"
    return s


def build_weather_attrs(params: dict, route_percentage: float) -> str:
    """Return a string of XML attributes for one <weather> element."""
    fixed_keys = ["cloudiness", "precipitation", "precipitation_deposits",
                  "wetness", "wind_intensity", "sun_azimuth_angle",
                  "sun_altitude_angle", "fog_density"]
    extra_keys = [k for k in params if k not in fixed_keys]

    lines = [f'route_percentage="{fmt(route_percentage, 1)}"']
    for k in fixed_keys:
        if k in params:
            lines.append(f'{k}="{fmt(params[k], 1)}"')
    for k in extra_keys:
        lines.append(f'{k}="{fmt(params[k], 1)}"')

    # First attr on same line as <weather, rest indented 12 spaces
    return "\n            ".join(lines)


def generate_route_xml(route_id: str, town: str, weather_params: dict,
                       waypoints_df: pd.DataFrame, z_values) -> str:
    """Build route XML string matching routes_devtest.xml style."""
    lines = []
    lines.append("<routes>")
    lines.append(f'   <route id="{route_id}" town="{town}">')

    # weathers (applied uniformly at 0% and 100%)
    lines.append("      <weathers>")
    for pct in [0, 100]:
        attr_str = build_weather_attrs(weather_params, pct)
        lines.append(f"         <weather {attr_str}/>")
    lines.append("      </weathers>")

    # waypoints
    lines.append("      <waypoints>")
    for (_, row), z_val in zip(waypoints_df.iterrows(), z_values):
        x = fmt(row["x_m"], 4)
        y = fmt(row["y_m"], 4)
        z = fmt(z_val, 4)
        lines.append(f'         <position x="{x}" y="{y}" z="{z}"/>')
    lines.append("      </waypoints>")

    # scenarios (empty — no traffic actors, no pedestrians)
    lines.append("      <scenarios>")
    lines.append("      </scenarios>")

    lines.append("   </route>")
    lines.append("</routes>")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Scenario definition JSON
# ---------------------------------------------------------------------------
def generate_scenario_json(route_id: str, town: str, route_xml_path: str) -> dict:
    """Empty scenario definition — no traffic actors, no pedestrians."""
    return {
        "description": (
            f"Empty scenario set for route '{route_id}' in {town}. "
            "No traffic actors, no pedestrians, no events. "
            "Weather and seed are specified in the route XML and metadata."
        ),
        "route_id": route_id,
        "town": town,
        "route_xml": str(route_xml_path),
        "available_scenarios": [],
        "traffic_actors": "none",
        "pedestrians": "none",
        "traffic_manager": "none",
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def validate(xml_path: Path, scenario_path: Path, route_id: str, town: str,
             source_df: pd.DataFrame, waypoints_df: pd.DataFrame,
             weather_params: dict, z_mode: str = "fixed",
             fixed_z: float = 0.5) -> bool:
    ok = True

    # 1. XML is parseable
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
    except ET.ParseError as e:
        print(f"[FAIL] XML parse error: {e}")
        return False
    print("[OK]   XML is well-formed")

    # 2. route id
    route_el = root.find("route")
    if route_el is None:
        print("[FAIL] No <route> element found")
        return False
    got_id = route_el.get("id")
    if got_id != route_id:
        print(f"[FAIL] route id='{got_id}', expected '{route_id}'")
        ok = False
    else:
        print(f"[OK]   route id = '{got_id}'")

    # 3. route town
    got_town = route_el.get("town")
    if got_town != town:
        print(f"[FAIL] route town='{got_town}', expected '{town}'")
        ok = False
    else:
        print(f"[OK]   route town = '{got_town}' (matches expected loaded map)")

    # 4. waypoints count
    wps = route_el.findall("waypoints/position")
    if len(wps) < 2:
        print(f"[FAIL] Only {len(wps)} waypoint(s); need >= 2")
        ok = False
    else:
        print(f"[OK]   waypoints count = {len(wps)}")

    # 5. first waypoint matches CSV first row
    tol = 0.1
    first_csv = source_df.iloc[0]
    wp0 = wps[0]
    dx = abs(float(wp0.get("x")) - float(first_csv["x_m"]))
    dy = abs(float(wp0.get("y")) - float(first_csv["y_m"]))
    if dx > tol or dy > tol:
        print(f"[FAIL] First waypoint mismatch: XML=({wp0.get('x')},{wp0.get('y')}) "
              f"CSV=({first_csv['x_m']:.4f},{first_csv['y_m']:.4f})")
        ok = False
    else:
        print(f"[OK]   First waypoint matches CSV start "
              f"(x={wp0.get('x')}, y={wp0.get('y')})")

    # 6. last waypoint matches CSV last row
    last_csv = source_df.iloc[-1]
    wpN = wps[-1]
    dx = abs(float(wpN.get("x")) - float(last_csv["x_m"]))
    dy = abs(float(wpN.get("y")) - float(last_csv["y_m"]))
    if dx > tol or dy > tol:
        print(f"[FAIL] Last waypoint mismatch: XML=({wpN.get('x')},{wpN.get('y')}) "
              f"CSV=({last_csv['x_m']:.4f},{last_csv['y_m']:.4f})")
        ok = False
    else:
        print(f"[OK]   Last waypoint matches CSV end "
              f"(x={wpN.get('x')}, y={wpN.get('y')})")

    # 7. z values in fixed mode
    if z_mode == "fixed":
        z_vals = [float(wp.get("z", 0)) for wp in wps]
        bad = [z for z in z_vals if abs(z - fixed_z) > 1e-6]
        if bad:
            print(f"[FAIL] {len(bad)} waypoint(s) have z != {fixed_z} in fixed mode")
            ok = False
        else:
            print(f"[OK]   All {len(wps)} waypoints have z = {fixed_z} (fixed mode)")
    else:
        print(f"[OK]   z_mode=csv, waypoint z values from source CSV")

    # 8. weather elements present
    weather_els = route_el.findall("weathers/weather")
    if not weather_els:
        print("[FAIL] No <weather> elements found")
        ok = False
    else:
        key_check = "cloudiness"
        if weather_els[0].get(key_check) is None:
            print(f"[FAIL] <weather> missing '{key_check}' attribute")
            ok = False
        else:
            print(f"[OK]   weather elements = {len(weather_els)}, "
                  f"cloudiness[0]={weather_els[0].get('cloudiness')}")

    # 9. scenarios element present
    sc_el = route_el.find("scenarios")
    if sc_el is None:
        print("[FAIL] No <scenarios> element found")
        ok = False
    else:
        print(f"[OK]   <scenarios> present (empty: {len(list(sc_el))} scenario entries)")

    # 10. scenario definition JSON exists
    if not scenario_path.exists():
        print(f"[FAIL] Scenario JSON not found: {scenario_path}")
        ok = False
    else:
        print(f"[OK]   Scenario definition JSON exists: {scenario_path}")

    return ok


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    csv_path = Path(args.route_csv)
    lb_root = Path(args.leaderboard_root)
    out_xml = Path(args.output_xml)
    out_meta = Path(args.output_metadata)
    out_scenario = Path(args.output_scenario)

    # --- load CSV ---
    if not csv_path.exists():
        sys.exit(f"[ERROR] Route CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    required = {"s_m", "x_m", "y_m", "z_m"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] Route CSV missing columns: {sorted(missing)}")
    print(f"[INFO] Loaded {len(df)} rows from {csv_path}")
    print(f"[INFO] Route length: {df['s_m'].iloc[-1]:.2f} m")

    # --- leaderboard git hash ---
    try:
        lb_commit = subprocess.check_output(
            ["git", "-C", str(lb_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        lb_commit = "unknown"
    print(f"[INFO] Leaderboard commit: {lb_commit}")

    # --- subsample waypoints ---
    waypoints_df = subsample_waypoints(df, args.waypoint_spacing_m)
    print(f"[INFO] Waypoints: {len(df)} → {len(waypoints_df)} "
          f"(spacing={args.waypoint_spacing_m} m)")

    # --- compute output z values ---
    source_z = waypoints_df["z_m"].tolist()
    if args.route_z_mode == "fixed":
        z_values = [args.fixed_z] * len(waypoints_df)
        print(f"[INFO] z_mode=fixed, all waypoint z = {args.fixed_z} m")
    else:
        z_values = source_z
        print(f"[INFO] z_mode=csv, waypoint z from CSV "
              f"(min={min(source_z):.4f}, max={max(source_z):.4f})")

    # --- weather ---
    weather_params = WEATHER_PRESETS[args.weather]
    print(f"[INFO] Weather preset: {args.weather}")

    # --- output dirs ---
    for p in [out_xml, out_meta, out_scenario]:
        p.parent.mkdir(parents=True, exist_ok=True)

    # --- generate XML ---
    xml_content = generate_route_xml(
        route_id=args.route_id,
        town=args.town,
        weather_params=weather_params,
        waypoints_df=waypoints_df,
        z_values=z_values,
    )
    out_xml.write_text(xml_content, encoding="utf-8")
    print(f"[INFO] Route XML saved: {out_xml}")

    # --- generate scenario JSON ---
    scenario_data = generate_scenario_json(
        route_id=args.route_id,
        town=args.town,
        route_xml_path=out_xml,
    )
    out_scenario.write_text(
        json.dumps(scenario_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Scenario definition saved: {out_scenario}")

    # --- generate metadata JSON ---
    first_wp = waypoints_df.iloc[0]
    last_wp = waypoints_df.iloc[-1]
    source_z_all = waypoints_df["z_m"].tolist()
    metadata = {
        "map_asset": "CARLA standard map",
        "map_name": args.town,
        "route_xml_town": args.town,
        "loaded_carla_map_name_expected": args.town,
        "town_matches_loaded_map": True,
        "carla_version": args.carla_version,
        "leaderboard_root": str(lb_root.resolve()),
        "leaderboard_version_or_commit": lb_commit,
        "route_id": args.route_id,
        "route_name": args.route_name,
        "source_route_csv": str(csv_path),
        "route_xml": str(out_xml),
        "scenario_definition": str(out_scenario),
        "scenario_definition_note": (
            "Empty scenario set (no traffic actors, no pedestrians). "
            "<scenarios> is also present but empty in the route XML."
        ),
        "waypoint_spacing_m": args.waypoint_spacing_m,
        "waypoints_total_in_xml": len(waypoints_df),
        "source_csv_rows": len(df),
        "route_length_m": float(df["s_m"].iloc[-1]),
        "route_z_mode": args.route_z_mode,
        "fixed_z": args.fixed_z,
        "source_csv_z_min": float(min(source_z_all)),
        "source_csv_z_max": float(max(source_z_all)),
        "output_xml_z_min": float(min(z_values)),
        "output_xml_z_max": float(max(z_values)),
        "first_waypoint": {"x": float(first_wp["x_m"]), "y": float(first_wp["y_m"]),
                           "z": float(z_values[0])},
        "last_waypoint": {"x": float(last_wp["x_m"]), "y": float(last_wp["y_m"]),
                          "z": float(z_values[-1])},
        "weather_name": args.weather,
        "weather_parameters": weather_params,
        "traffic_actors": "none",
        "pedestrians": "none",
        "traffic_manager": "none",
        "seed": args.seed,
        "intended_run_list": ["smoke", "regression"],
        "submitted_assets": {
            "map_name": args.town,
            "route_xml": str(out_xml),
            "scenario_definition": str(out_scenario),
            "weather_parameters": args.weather,
            "seed": args.seed,
        },
        "note": (
            "Learning data route (Town10HD_Opt sp147) reused as regression evaluation "
            "to compare before/after additional training under the same conditions. "
            "A held-out route for generalization evaluation will be created separately."
        ),
    }
    out_meta.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Metadata saved: {out_meta}")

    # --- validate ---
    print("\n[INFO] Running validation ...")
    ok = validate(
        xml_path=out_xml,
        scenario_path=out_scenario,
        route_id=args.route_id,
        town=args.town,
        source_df=df,
        waypoints_df=waypoints_df,
        weather_params=weather_params,
        z_mode=args.route_z_mode,
        fixed_z=args.fixed_z,
    )

    if ok:
        print("\n[DONE] All validation checks passed.")
        print(f"  Route XML       : {out_xml}")
        print(f"  Scenario JSON   : {out_scenario}")
        print(f"  Metadata JSON   : {out_meta}")
    else:
        print("\n[WARN] Some validation checks failed — review output above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
