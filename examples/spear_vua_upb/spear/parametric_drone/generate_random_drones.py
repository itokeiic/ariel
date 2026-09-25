#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import subprocess
from copy import deepcopy
from pathlib import Path


FAMILIES = ["quad_x", "quad_plus", "hex_wide", "hex_long", "octo_corner"]


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def fill8(vals, default):
    out = list(vals)
    while len(out) < 8:
        out.append(default)
    return out[:8]


def anchor_face_front(L, W):
    return ( L / 2.0, 0.0)

def anchor_face_rear(L, W):
    return (-L / 2.0, 0.0)

def anchor_face_left(L, W):
    return (0.0,  W / 2.0)

def anchor_face_right(L, W):
    return (0.0, -W / 2.0)

def anchor_corner_fl(L, W):
    return ( L / 2.0,  W / 2.0)

def anchor_corner_fr(L, W):
    return ( L / 2.0, -W / 2.0)

def anchor_corner_rl(L, W):
    return (-L / 2.0,  W / 2.0)

def anchor_corner_rr(L, W):
    return (-L / 2.0, -W / 2.0)


def blend(p0, p1, a):
    return ((1.0 - a) * p0[0] + a * p1[0], (1.0 - a) * p0[1] + a * p1[1])


def point_angle(x, y):
    return math.atan2(y, x)


def station_from_anchor(anchor, arm_len, extra_angle=0.0, rotor_z=0.0):
    x, y = anchor
    ang = point_angle(x, y) + extra_angle
    return {
        "mount_x": round(x, 4),
        "mount_y": round(y, 4),
        "arm_len": round(arm_len, 4),
        "arm_ang": round(ang, 6),
        "rotor_z": round(rotor_z, 4),
        "active": True,
    }


def base_genome(idx, family):
    num_rotors = 4 if "quad" in family else 6 if "hex" in family else 8
    return {
        "id": f"cube_{idx:03d}",
        "family": family,
        "num_rotors": num_rotors,
        "body": {
            "length": 0.18,
            "width": 0.18,
            "height": 0.06,
        },
        "global": {
            "arm_radius": 0.010,
            "rotor_radius": 0.055 if num_rotors == 4 else 0.050 if num_rotors == 6 else 0.045,
            "rotor_thickness": 0.008,
            "base_mass": 1.10 + 0.08 * (num_rotors - 4),
            "arm_mass": 0.035,
            "rotor_mass": 0.022,
            "imu_mass": 0.010,
        },
        "shape": {
            "face_to_corner_blend": 0.35,
            "front_rear_bias": 0.00,
            "side_bias": 0.00,
            "vertical_stagger": 0.00,
        },
        "stations": [],
    }


def apply_family_template(genome):
    g = deepcopy(genome)
    family = g["family"]

    if family == "quad_x":
        g["body"]["length"] = 0.18
        g["body"]["width"] = 0.18
        g["shape"]["face_to_corner_blend"] = 0.75

    elif family == "quad_plus":
        g["body"]["length"] = 0.17
        g["body"]["width"] = 0.17
        g["shape"]["face_to_corner_blend"] = 0.0

    elif family == "hex_wide":
        g["body"]["length"] = 0.18
        g["body"]["width"] = 0.22
        g["shape"]["face_to_corner_blend"] = 0.25
        g["shape"]["side_bias"] = 0.06

    elif family == "hex_long":
        g["body"]["length"] = 0.24
        g["body"]["width"] = 0.16
        g["shape"]["face_to_corner_blend"] = 0.20
        g["shape"]["front_rear_bias"] = 0.08

    elif family == "octo_corner":
        g["body"]["length"] = 0.20
        g["body"]["width"] = 0.20
        g["shape"]["face_to_corner_blend"] = 1.0
        g["shape"]["vertical_stagger"] = 0.01

    return g


def mutate_shape(g, rng):
    g["body"]["length"] = clamp(g["body"]["length"] + rng.uniform(-0.02, 0.02), 0.14, 0.30)
    g["body"]["width"] = clamp(g["body"]["width"] + rng.uniform(-0.02, 0.02), 0.14, 0.30)
    g["body"]["height"] = clamp(g["body"]["height"] + rng.uniform(-0.01, 0.01), 0.04, 0.10)

    g["shape"]["face_to_corner_blend"] = clamp(g["shape"]["face_to_corner_blend"] + rng.uniform(-0.15, 0.15), 0.0, 1.0)
    g["shape"]["front_rear_bias"] = clamp(g["shape"]["front_rear_bias"] + rng.uniform(-0.05, 0.05), -0.12, 0.12)
    g["shape"]["side_bias"] = clamp(g["shape"]["side_bias"] + rng.uniform(-0.05, 0.05), -0.12, 0.12)
    g["shape"]["vertical_stagger"] = clamp(g["shape"]["vertical_stagger"] + rng.uniform(-0.01, 0.01), -0.03, 0.03)

    g["global"]["rotor_radius"] = clamp(g["global"]["rotor_radius"] + rng.uniform(-0.004, 0.004), 0.035, 0.07)
    return g


def build_stations(g):
    L = g["body"]["length"] * (1.0 + g["shape"]["front_rear_bias"])
    W = g["body"]["width"] * (1.0 + g["shape"]["side_bias"])
    b = g["shape"]["face_to_corner_blend"]
    z = g["shape"]["vertical_stagger"]

    arm_base = max(L, W) * 0.38
    fam = g["family"]
    sts = []

    if fam == "quad_x":
        anchors = [
            blend(anchor_face_front(L, W), anchor_corner_fl(L, W), b),
            blend(anchor_face_front(L, W), anchor_corner_fr(L, W), b),
            blend(anchor_face_rear(L, W), anchor_corner_rl(L, W), b),
            blend(anchor_face_rear(L, W), anchor_corner_rr(L, W), b),
        ]
        zs = [ z, z, -z, -z]
        for i, a in enumerate(anchors):
            sts.append(station_from_anchor(a, arm_base, 0.0, zs[i]))

    elif fam == "quad_plus":
        anchors = [
            anchor_face_front(L, W),
            anchor_face_left(L, W),
            anchor_face_rear(L, W),
            anchor_face_right(L, W),
        ]
        for a in anchors:
            sts.append(station_from_anchor(a, arm_base, 0.0, 0.0))

    elif fam == "hex_wide":
        anchors = [
            blend(anchor_face_front(L, W), anchor_corner_fl(L, W), b),
            blend(anchor_face_front(L, W), anchor_corner_fr(L, W), b),
            anchor_face_left(L, W),
            anchor_face_right(L, W),
            blend(anchor_face_rear(L, W), anchor_corner_rl(L, W), b),
            blend(anchor_face_rear(L, W), anchor_corner_rr(L, W), b),
        ]
        zs = [ z, z, 0.0, 0.0, -z, -z]
        lengths = [arm_base, arm_base, arm_base * 0.92, arm_base * 0.92, arm_base, arm_base]
        for i, a in enumerate(anchors):
            sts.append(station_from_anchor(a, lengths[i], 0.0, zs[i]))

    elif fam == "hex_long":
        anchors = [
            anchor_face_front(L, W),
            blend(anchor_face_front(L, W), anchor_corner_fl(L, W), b),
            blend(anchor_face_front(L, W), anchor_corner_fr(L, W), b),
            blend(anchor_face_rear(L, W), anchor_corner_rl(L, W), b),
            blend(anchor_face_rear(L, W), anchor_corner_rr(L, W), b),
            anchor_face_rear(L, W),
        ]
        lengths = [arm_base * 0.85, arm_base, arm_base, arm_base, arm_base, arm_base * 0.85]
        zs = [ z, z, z, -z, -z, -z]
        for i, a in enumerate(anchors):
            sts.append(station_from_anchor(a, lengths[i], 0.0, zs[i]))

    elif fam == "octo_corner":
        anchors = [
            anchor_corner_fl(L, W),
            anchor_corner_fr(L, W),
            anchor_corner_rl(L, W),
            anchor_corner_rr(L, W),
            blend(anchor_face_front(L, W), anchor_corner_fl(L, W), 0.5),
            blend(anchor_face_front(L, W), anchor_corner_fr(L, W), 0.5),
            blend(anchor_face_rear(L, W), anchor_corner_rl(L, W), 0.5),
            blend(anchor_face_rear(L, W), anchor_corner_rr(L, W), 0.5),
        ]
        zs = [ z, z, -z, -z, 0.0, 0.0, 0.0, 0.0]
        lengths = [arm_base * 0.95, arm_base * 0.95, arm_base * 0.95, arm_base * 0.95,
                   arm_base * 0.80, arm_base * 0.80, arm_base * 0.80, arm_base * 0.80]
        for i, a in enumerate(anchors):
            sts.append(station_from_anchor(a, lengths[i], 0.0, zs[i]))

    g["stations"] = sts
    return g


def min_rotor_clearance(g):
    rr = g["global"]["rotor_radius"]
    centers = []
    for st in g["stations"]:
        px = st["mount_x"] + st["arm_len"] * math.cos(st["arm_ang"])
        py = st["mount_y"] + st["arm_len"] * math.sin(st["arm_ang"])
        pz = st["rotor_z"]
        centers.append((px, py, pz))
    min_d = 1e9
    for i in range(len(centers)):
        for j in range(i + 1, len(centers)):
            dx = centers[i][0] - centers[j][0]
            dy = centers[i][1] - centers[j][1]
            dz = centers[i][2] - centers[j][2]
            d = math.sqrt(dx * dx + dy * dy + dz * dz)
            min_d = min(min_d, d)
    return min_d - 2.1 * rr


def valid_genome(g):
    if len(g["stations"]) != g["num_rotors"]:
        return False
    if min_rotor_clearance(g) < 0.0:
        return False
    if g["body"]["length"] < 2.2 * g["global"]["rotor_radius"]:
        return False
    if g["body"]["width"] < 2.2 * g["global"]["rotor_radius"]:
        return False
    return True


def to_xacro_args(genome):
    stations = fill8(genome["stations"], {
        "mount_x": 0.0, "mount_y": 0.0, "arm_len": 0.0, "arm_ang": 0.0, "rotor_z": 0.0, "active": False
    })
    args = {
        "genome_id": genome["id"],
        "family": genome["family"],
        "num_rotors": genome["num_rotors"],
        "body_length": round(genome["body"]["length"], 4),
        "body_width": round(genome["body"]["width"], 4),
        "body_height": round(genome["body"]["height"], 4),
        "arm_radius": round(genome["global"]["arm_radius"], 4),
        "rotor_radius": round(genome["global"]["rotor_radius"], 4),
        "rotor_thickness": round(genome["global"]["rotor_thickness"], 4),
        "base_mass": round(genome["global"]["base_mass"], 4),
        "arm_mass": round(genome["global"]["arm_mass"], 4),
        "rotor_mass": round(genome["global"]["rotor_mass"], 4),
        "imu_mass": round(genome["global"]["imu_mass"], 4),
    }
    for i, st in enumerate(stations, start=1):
        args[f"mount_x_{i}"] = st["mount_x"]
        args[f"mount_y_{i}"] = st["mount_y"]
        args[f"arm_len_{i}"] = st["arm_len"]
        args[f"arm_ang_{i}"] = st["arm_ang"]
        args[f"rotor_z_{i}"] = st["rotor_z"]
        args[f"active_{i}"] = "true" if st["active"] else "false"
    return args


def run_xacro(xacro_file, output_urdf, args_dict):
    cmd = ["xacro", str(xacro_file)]
    for k, v in args_dict.items():
        cmd.append(f"{k}:={v}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    output_urdf.write_text(result.stdout)


def generate_one(idx, rng):
    for _ in range(50):
        fam = rng.choice(FAMILIES)
        g = base_genome(idx, fam)
        g = apply_family_template(g)
        g = mutate_shape(g, rng)
        g = build_stations(g)
        if valid_genome(g):
            g["id"] = f"{g['family']}_{idx:03d}"
            return g
    raise RuntimeError(f"Failed to generate valid genome for index {idx}")


def main():
    parser = argparse.ArgumentParser(description="Generate cube-scaffold parametric drone variants.")
    parser.add_argument("--xacro", required=True, help="Path to parametric_drone.urdf.xacro")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--count", type=int, default=10, help="Number of variants")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    xacro_file = Path(args.xacro).resolve()
    out_dir = Path(args.out_dir).resolve()
    urdf_dir = out_dir / "urdf"
    genome_dir = out_dir / "genomes"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    genome_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for i in range(args.count):
        genome = generate_one(i, rng)

        genome_path = genome_dir / f"{genome['id']}.json"
        with open(genome_path, "w") as f:
            json.dump(genome, f, indent=2)

        urdf_path = urdf_dir / f"{genome['id']}.urdf"
        run_xacro(xacro_file, urdf_path, to_xacro_args(genome))

        manifest.append({
            "id": genome["id"],
            "family": genome["family"],
            "num_rotors": genome["num_rotors"],
            "body_length": round(genome["body"]["length"], 3),
            "body_width": round(genome["body"]["width"], 3),
            "body_height": round(genome["body"]["height"], 3),
            "blend": round(genome["shape"]["face_to_corner_blend"], 3),
            "front_rear_bias": round(genome["shape"]["front_rear_bias"], 3),
            "side_bias": round(genome["shape"]["side_bias"], 3),
            "vertical_stagger": round(genome["shape"]["vertical_stagger"], 3),
            "genome": str(genome_path),
            "urdf": str(urdf_path),
        })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Generated {len(manifest)} cube-scaffold genomes in {genome_dir}")
    print(f"Generated {len(manifest)} URDFs in {urdf_dir}")


if __name__ == "__main__":
    main()