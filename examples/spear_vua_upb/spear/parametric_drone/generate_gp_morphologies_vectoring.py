#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import subprocess
from pathlib import Path

import numpy as np


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def lerp(a, b, t):
    return (1.0 - t) * a + t * b


def smoothstep(t):
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def run_xacro(xacro_file, output_urdf, args_dict):
    cmd = ["xacro", str(xacro_file)]
    for k, v in args_dict.items():
        cmd.append(f"{k}:={v}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("\n[ERROR] xacro failed")
        print(" ".join(cmd))
        print("\n[STDERR]\n", result.stderr)
        raise RuntimeError(f"xacro failed for {output_urdf}")
    output_urdf.write_text(result.stdout)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ], dtype=float)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def arm_rot(base_yaw, tilt_x, tilt_y):
    return rot_z(base_yaw) @ rot_x(tilt_x) @ rot_y(tilt_y)


def genotype_family(seed):
    rng = random.Random(seed)
    return {
        "family_id": f"gptm_family_{seed}",
        "topology": "articulated_quad_4r_tilt_mount",
        "genes": {
            "body_length": clamp(0.24 + rng.uniform(-0.02, 0.02), 0.20, 0.28),
            "body_width": clamp(0.12 + rng.uniform(-0.015, 0.015), 0.10, 0.16),
            "body_height": clamp(0.05 + rng.uniform(-0.008, 0.008), 0.04, 0.07),
            "hub_x": clamp(0.055 + rng.uniform(-0.008, 0.008), 0.04, 0.07),
            "hub_y": clamp(0.035 + rng.uniform(-0.006, 0.006), 0.025, 0.05),
            "hub_z": clamp(0.020 + rng.uniform(-0.006, 0.006), 0.01, 0.04),
            "arm_length_closed": clamp(0.060 + rng.uniform(-0.008, 0.008), 0.050, 0.080),
            "arm_length_open": clamp(0.150 + rng.uniform(-0.015, 0.015), 0.130, 0.180),
            "tip_length_closed": clamp(0.035 + rng.uniform(-0.008, 0.008), 0.025, 0.050),
            "tip_length_open": clamp(0.100 + rng.uniform(-0.010, 0.010), 0.085, 0.120),
            "arm_radius": clamp(0.012 + rng.uniform(-0.002, 0.002), 0.009, 0.016),
            "tip_radius": clamp(0.010 + rng.uniform(-0.002, 0.002), 0.008, 0.014),
            "rotor_radius": clamp(0.060 + rng.uniform(-0.008, 0.008), 0.050, 0.070),
            "rotor_thickness": clamp(0.008 + rng.uniform(-0.002, 0.002), 0.006, 0.012),
            "body_z": clamp(0.030 + rng.uniform(-0.005, 0.005), 0.02, 0.04),
            "rotor_mount_z_closed": clamp(0.020 + rng.uniform(-0.006, 0.006), 0.008, 0.03),
            "rotor_mount_z_open": clamp(0.008 + rng.uniform(-0.004, 0.004), 0.0, 0.02),
            "front_yaw_closed_deg": clamp(8.0 + rng.uniform(-4.0, 4.0), 2.0, 15.0),
            "front_yaw_open_deg": clamp(45.0 + rng.uniform(-2.0, 2.0), 42.0, 48.0),
            "rear_yaw_closed_deg": clamp(172.0 + rng.uniform(-4.0, 4.0), 165.0, 178.0),
            "rear_yaw_open_deg": clamp(135.0 + rng.uniform(-2.0, 2.0), 132.0, 138.0),
            "body_mass": clamp(1.10 + rng.uniform(-0.10, 0.10), 0.90, 1.30),
            "hinge_mass": clamp(0.020 + rng.uniform(-0.005, 0.005), 0.01, 0.03),
            "arm_mass": clamp(0.080 + rng.uniform(-0.015, 0.015), 0.05, 0.11),
            "tip_mass": clamp(0.050 + rng.uniform(-0.010, 0.010), 0.03, 0.07),
            "rotor_mass": clamp(0.020 + rng.uniform(-0.005, 0.005), 0.01, 0.03),
            "tilt_link_mass": clamp(0.006 + rng.uniform(-0.002, 0.002), 0.003, 0.010),
            "tilt_deg_max": clamp(8.0 + rng.uniform(-3.0, 3.0), 3.0, 12.0),
        },
    }


def phenotype_from_genotype(g, alpha, rng, zero_tilt=False, correlated_tilt=True):
    a = smoothstep(alpha)
    genes = g["genes"]
    tilt_max = 0.0 if zero_tilt else genes["tilt_deg_max"]

    p = {
        "robot_name": f"{g['family_id']}_a{int(round(alpha * 100)):03d}",
        "alpha": round(alpha, 4),
        "body_length": round(genes["body_length"], 4),
        "body_width": round(genes["body_width"], 4),
        "body_height": round(genes["body_height"], 4),
        "body_mass": round(genes["body_mass"], 4),
        "hinge_mass": round(genes["hinge_mass"], 4),
        "arm_mass": round(genes["arm_mass"], 4),
        "tip_mass": round(genes["tip_mass"], 4),
        "rotor_mass": round(genes["rotor_mass"], 4),
        "tilt_link_mass": round(genes["tilt_link_mass"], 4),
        "arm_radius": round(genes["arm_radius"], 4),
        "arm_length": round(lerp(genes["arm_length_closed"], genes["arm_length_open"], a), 4),
        "tip_radius": round(genes["tip_radius"], 4),
        "tip_length": round(lerp(genes["tip_length_closed"], genes["tip_length_open"], a), 4),
        "rotor_radius": round(genes["rotor_radius"], 4),
        "rotor_thickness": round(genes["rotor_thickness"], 4),
        "hub_x": round(lerp(genes["hub_x"] * 0.85, genes["hub_x"], a), 4),
        "hub_y": round(lerp(genes["hub_y"] * 0.80, genes["hub_y"], a), 4),
        "hub_z": round(genes["hub_z"], 4),
        "front_yaw": round(math.radians(lerp(genes["front_yaw_closed_deg"], genes["front_yaw_open_deg"], a)), 6),
        "rear_yaw": round(math.radians(lerp(genes["rear_yaw_closed_deg"], genes["rear_yaw_open_deg"], a)), 6),
        "body_z": round(genes["body_z"], 4),
        "rotor_mount_z": round(lerp(genes["rotor_mount_z_closed"], genes["rotor_mount_z_open"], a), 4),
    }

    if correlated_tilt:
        pitch_bias = rng.uniform(-tilt_max, tilt_max)
        roll_bias = rng.uniform(-tilt_max, tilt_max)
        twist_bias = rng.uniform(-0.5 * tilt_max, 0.5 * tilt_max)

        p.update({
            "front_left_tilt_x": round(math.radians(pitch_bias + twist_bias), 6),
            "front_left_tilt_y": round(math.radians(roll_bias), 6),
            "front_right_tilt_x": round(math.radians(pitch_bias - twist_bias), 6),
            "front_right_tilt_y": round(math.radians(-roll_bias), 6),
            "rear_left_tilt_x": round(math.radians(-pitch_bias + twist_bias), 6),
            "rear_left_tilt_y": round(math.radians(roll_bias), 6),
            "rear_right_tilt_x": round(math.radians(-pitch_bias - twist_bias), 6),
            "rear_right_tilt_y": round(math.radians(-roll_bias), 6),
        })
    else:
        p.update({
            "front_left_tilt_x": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
            "front_left_tilt_y": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
            "front_right_tilt_x": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
            "front_right_tilt_y": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
            "rear_left_tilt_x": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
            "rear_left_tilt_y": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
            "rear_right_tilt_x": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
            "rear_right_tilt_y": round(math.radians(rng.uniform(-tilt_max, tilt_max)), 6),
        })

    return p


def phenotype_to_xacro_args(p):
    return {k: p[k] for k in [
        "robot_name", "body_length", "body_width", "body_height", "body_mass", "hinge_mass",
        "arm_mass", "tip_mass", "rotor_mass", "tilt_link_mass", "arm_radius", "arm_length",
        "tip_radius", "tip_length", "rotor_radius", "rotor_thickness", "hub_x", "hub_y",
        "hub_z", "front_yaw", "rear_yaw", "body_z", "rotor_mount_z",
        "front_left_tilt_x", "front_left_tilt_y", "front_right_tilt_x", "front_right_tilt_y",
        "rear_left_tilt_x", "rear_left_tilt_y", "rear_right_tilt_x", "rear_right_tilt_y",
    ]}


def compute_rotor_pose(p, arm_name):
    if arm_name == "front_left":
        base_yaw = p["front_yaw"]
        mount = np.array([p["hub_x"], p["hub_y"], p["hub_z"]], dtype=float)
        tilt_x = p["front_left_tilt_x"]
        tilt_y = p["front_left_tilt_y"]
    elif arm_name == "front_right":
        base_yaw = -p["front_yaw"]
        mount = np.array([p["hub_x"], -p["hub_y"], p["hub_z"]], dtype=float)
        tilt_x = p["front_right_tilt_x"]
        tilt_y = p["front_right_tilt_y"]
    elif arm_name == "rear_left":
        base_yaw = p["rear_yaw"]
        mount = np.array([-p["hub_x"], p["hub_y"], p["hub_z"]], dtype=float)
        tilt_x = p["rear_left_tilt_x"]
        tilt_y = p["rear_left_tilt_y"]
    elif arm_name == "rear_right":
        base_yaw = -p["rear_yaw"]
        mount = np.array([-p["hub_x"], -p["hub_y"], p["hub_z"]], dtype=float)
        tilt_x = p["rear_right_tilt_x"]
        tilt_y = p["rear_right_tilt_y"]
    else:
        raise ValueError(f"Unknown arm name: {arm_name}")

    R_yaw = rot_z(base_yaw)
    tip_end = np.array([p["arm_length"] + p["tip_length"], 0.0, 0.0], dtype=float)
    tilt_mount_origin = mount + R_yaw @ tip_end

    R_mount = arm_rot(base_yaw, tilt_x, tilt_y)

    rotor_offset = np.array([0.0, 0.0, p["rotor_mount_z"] + 0.012], dtype=float)
    rotor_pos = tilt_mount_origin + R_mount @ rotor_offset

    thrust_dir = R_mount @ np.array([0.0, 0.0, 1.0], dtype=float)
    thrust_dir = thrust_dir / np.linalg.norm(thrust_dir)

    return rotor_pos, thrust_dir


def compute_allocation_matrices(p, thrust_coeff=1.0, moment_coeff=0.05, spin_signs=None):
    if spin_signs is None:
        spin_signs = {
            "front_left": 1.0,
            "front_right": -1.0,
            "rear_left": -1.0,
            "rear_right": 1.0,
        }

    rotor_names = ["front_left", "front_right", "rear_left", "rear_right"]
    positions = []
    directions = []
    B_cols = []

    for name in rotor_names:
        r_i, d_i = compute_rotor_pose(p, name)
        s_i = spin_signs[name]
        f_col = thrust_coeff * d_i
        tau_col = np.cross(r_i, f_col) + s_i * moment_coeff * d_i
        b_i = np.concatenate([f_col, tau_col])
        positions.append(r_i)
        directions.append(d_i)
        B_cols.append(b_i)

    B_full = np.column_stack(B_cols)
    B_rpyt = np.vstack([
        B_full[2, :],
        B_full[3, :],
        B_full[4, :],
        B_full[5, :],
    ])

    rank_full = int(np.linalg.matrix_rank(B_full))
    rank_rpyt = int(np.linalg.matrix_rank(B_rpyt))
    cond_full = float(np.linalg.cond(B_full))
    cond_rpyt = float(np.linalg.cond(B_rpyt))

    return {
        "rotor_names": rotor_names,
        "positions": np.array(positions),
        "directions": np.array(directions),
        "B_full": B_full,
        "B_rpyt": B_rpyt,
        "B_full_pinv": np.linalg.pinv(B_full),
        "B_rpyt_pinv": np.linalg.pinv(B_rpyt),
        "rank_full": rank_full,
        "rank_rpyt": rank_rpyt,
        "cond_full": cond_full,
        "cond_rpyt": cond_rpyt,
        "spin_signs": spin_signs,
        "thrust_coeff": thrust_coeff,
        "moment_coeff": moment_coeff,
    }


def alpha_schedule(count):
    if count < 2:
        return [0.0]
    return [i / (count - 1) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description="Generate correlated transformable quad morphologies and precompute control allocation matrices.")
    parser.add_argument("--xacro", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--families", type=int, default=1)
    parser.add_argument("--per-family", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--zero-tilt", action="store_true", help="Generate leveled mounts with no random vectoring tilt.")
    parser.add_argument("--independent-tilt", action="store_true", help="Use independent per-arm tilt instead of correlated latent tilt.")
    parser.add_argument("--thrust-coeff", type=float, default=1.0)
    parser.add_argument("--moment-coeff", type=float, default=0.05)
    args = parser.parse_args()

    xacro_file = Path(args.xacro).resolve()
    out_dir = Path(args.out_dir).resolve()
    urdf_dir = out_dir / "urdf"
    genotype_dir = out_dir / "genotypes"
    phenotype_dir = out_dir / "phenotypes"
    allocation_dir = out_dir / "allocation"
    urdf_dir.mkdir(parents=True, exist_ok=True)
    genotype_dir.mkdir(parents=True, exist_ok=True)
    phenotype_dir.mkdir(parents=True, exist_ok=True)
    allocation_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for fam_idx in range(args.families):
        family_seed = args.seed + fam_idx
        genotype = genotype_family(family_seed)
        genotype_path = genotype_dir / f"{genotype['family_id']}.json"
        genotype_path.write_text(json.dumps(genotype, indent=2))

        for ph_idx, alpha in enumerate(alpha_schedule(args.per_family)):
            phenotype_rng = random.Random(family_seed * 1000 + ph_idx)
            phenotype = phenotype_from_genotype(
                genotype,
                alpha,
                phenotype_rng,
                zero_tilt=args.zero_tilt,
                correlated_tilt=not args.independent_tilt,
            )
            ph_name = f"{genotype['family_id']}_morph_{ph_idx:02d}"
            phenotype["robot_name"] = ph_name

            phenotype_path = phenotype_dir / f"{ph_name}.json"
            phenotype_path.write_text(json.dumps(phenotype, indent=2))

            urdf_path = urdf_dir / f"{ph_name}.urdf"
            run_xacro(xacro_file, urdf_path, phenotype_to_xacro_args(phenotype))

            allocation = compute_allocation_matrices(
                phenotype,
                thrust_coeff=args.thrust_coeff,
                moment_coeff=args.moment_coeff,
            )

            alloc_base = allocation_dir / ph_name
            np.savez(
                alloc_base.with_suffix(".npz"),
                B_full=allocation["B_full"],
                B_rpyt=allocation["B_rpyt"],
                B_full_pinv=allocation["B_full_pinv"],
                B_rpyt_pinv=allocation["B_rpyt_pinv"],
                rotor_positions=allocation["positions"],
                rotor_directions=allocation["directions"],
            )

            allocation_json = {
                "family_id": genotype["family_id"],
                "morphology_id": ph_name,
                "rotor_names": allocation["rotor_names"],
                "rotor_positions": allocation["positions"].tolist(),
                "rotor_directions": allocation["directions"].tolist(),
                "spin_signs": allocation["spin_signs"],
                "thrust_coeff": allocation["thrust_coeff"],
                "moment_coeff": allocation["moment_coeff"],
                "rank_full": allocation["rank_full"],
                "rank_rpyt": allocation["rank_rpyt"],
                "cond_full": allocation["cond_full"],
                "cond_rpyt": allocation["cond_rpyt"],
                "B_full": allocation["B_full"].tolist(),
                "B_rpyt": allocation["B_rpyt"].tolist(),
            }
            allocation_json_path = alloc_base.with_suffix(".json")
            allocation_json_path.write_text(json.dumps(allocation_json, indent=2))

            manifest.append({
                "family_id": genotype["family_id"],
                "morphology_id": ph_name,
                "alpha": phenotype["alpha"],
                "arm_length": phenotype["arm_length"],
                "tip_length": phenotype["tip_length"],
                "front_yaw": phenotype["front_yaw"],
                "rear_yaw": phenotype["rear_yaw"],
                "front_left_tilt_x": phenotype["front_left_tilt_x"],
                "front_left_tilt_y": phenotype["front_left_tilt_y"],
                "front_right_tilt_x": phenotype["front_right_tilt_x"],
                "front_right_tilt_y": phenotype["front_right_tilt_y"],
                "rear_left_tilt_x": phenotype["rear_left_tilt_x"],
                "rear_left_tilt_y": phenotype["rear_left_tilt_y"],
                "rear_right_tilt_x": phenotype["rear_right_tilt_x"],
                "rear_right_tilt_y": phenotype["rear_right_tilt_y"],
                "rank_rpyt": allocation["rank_rpyt"],
                "cond_rpyt": allocation["cond_rpyt"],
                "genotype": str(genotype_path),
                "phenotype": str(phenotype_path),
                "urdf": str(urdf_path),
                "allocation_npz": str(alloc_base.with_suffix('.npz')),
                "allocation_json": str(allocation_json_path),
            })

    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Generated {args.families} genotype families and {len(manifest)} phenotypes.")
    if args.zero_tilt:
        print("Tilting red mounts disabled: all mount tilt values are zero for alignment debug.")
    print("Saved allocation matrices (.npz) and metadata (.json) for each morphology.")


if __name__ == "__main__":
    main()