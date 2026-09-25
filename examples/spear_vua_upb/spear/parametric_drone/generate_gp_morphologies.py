#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import subprocess
from pathlib import Path


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


def genotype_family(seed):
    rng = random.Random(seed)

    g = {
        "family_id": f"gp_family_{seed}",
        "topology": "articulated_quad_4r",
        "genes": {
            "body_length": clamp(0.24 + rng.uniform(-0.02, 0.02), 0.20, 0.28),
            "body_width": clamp(0.12 + rng.uniform(-0.015, 0.015), 0.10, 0.16),
            "body_height": clamp(0.05 + rng.uniform(-0.008, 0.008), 0.04, 0.07),
            "hub_x": clamp(0.055 + rng.uniform(-0.008, 0.008), 0.04, 0.07),
            "hub_y": clamp(0.035 + rng.uniform(-0.006, 0.006), 0.025, 0.05),
            "hub_z": clamp(0.020 + rng.uniform(-0.006, 0.006), 0.01, 0.04),
            "arm_length_closed": clamp(0.08 + rng.uniform(-0.01, 0.01), 0.06, 0.10),
            "arm_length_open": clamp(0.13 + rng.uniform(-0.015, 0.015), 0.11, 0.16),
            "tip_length_closed": clamp(0.05 + rng.uniform(-0.01, 0.01), 0.04, 0.07),
            "tip_length_open": clamp(0.09 + rng.uniform(-0.01, 0.01), 0.07, 0.11),
            "arm_radius": clamp(0.012 + rng.uniform(-0.002, 0.002), 0.009, 0.016),
            "tip_radius": clamp(0.010 + rng.uniform(-0.002, 0.002), 0.008, 0.014),
            "rotor_radius": clamp(0.060 + rng.uniform(-0.008, 0.008), 0.050, 0.070),
            "rotor_thickness": clamp(0.008 + rng.uniform(-0.002, 0.002), 0.006, 0.012),
            "body_z": clamp(0.030 + rng.uniform(-0.005, 0.005), 0.02, 0.04),
            "rotor_mount_z_closed": clamp(0.020 + rng.uniform(-0.006, 0.006), 0.008, 0.03),
            "rotor_mount_z_open": clamp(0.008 + rng.uniform(-0.004, 0.004), 0.0, 0.02),
            "front_yaw_closed_deg": clamp(25.0 + rng.uniform(-6.0, 6.0), 18.0, 35.0),
            "front_yaw_open_deg": clamp(45.0 + rng.uniform(-4.0, 4.0), 40.0, 50.0),
            "rear_yaw_closed_deg": clamp(155.0 + rng.uniform(-6.0, 6.0), 145.0, 165.0),
            "rear_yaw_open_deg": clamp(135.0 + rng.uniform(-4.0, 4.0), 130.0, 140.0),
            "body_mass": clamp(1.10 + rng.uniform(-0.10, 0.10), 0.90, 1.30),
            "hinge_mass": clamp(0.020 + rng.uniform(-0.005, 0.005), 0.01, 0.03),
            "arm_mass": clamp(0.080 + rng.uniform(-0.015, 0.015), 0.05, 0.11),
            "tip_mass": clamp(0.050 + rng.uniform(-0.010, 0.010), 0.03, 0.07),
            "rotor_mass": clamp(0.020 + rng.uniform(-0.005, 0.005), 0.01, 0.03),
        }
    }
    return g


def phenotype_from_genotype(g, alpha):
    a = smoothstep(alpha)
    genes = g["genes"]

    body_length = genes["body_length"]
    body_width = genes["body_width"]
    body_height = genes["body_height"]

    hub_x = lerp(genes["hub_x"] * 0.85, genes["hub_x"], a)
    hub_y = lerp(genes["hub_y"] * 0.80, genes["hub_y"], a)
    hub_z = genes["hub_z"]

    arm_length = lerp(genes["arm_length_closed"], genes["arm_length_open"], a)
    tip_length = lerp(genes["tip_length_closed"], genes["tip_length_open"], a)

    rotor_mount_z = lerp(genes["rotor_mount_z_closed"], genes["rotor_mount_z_open"], a)

    front_yaw = math.radians(lerp(genes["front_yaw_closed_deg"], genes["front_yaw_open_deg"], a))
    rear_yaw = math.radians(lerp(genes["rear_yaw_closed_deg"], genes["rear_yaw_open_deg"], a))

    p = {
        "robot_name": f"{g['family_id']}_a{int(round(alpha * 100)):03d}",
        "alpha": round(alpha, 4),
        "body_length": round(body_length, 4),
        "body_width": round(body_width, 4),
        "body_height": round(body_height, 4),
        "body_mass": round(genes["body_mass"], 4),
        "hinge_mass": round(genes["hinge_mass"], 4),
        "arm_mass": round(genes["arm_mass"], 4),
        "tip_mass": round(genes["tip_mass"], 4),
        "rotor_mass": round(genes["rotor_mass"], 4),
        "arm_radius": round(genes["arm_radius"], 4),
        "arm_length": round(arm_length, 4),
        "tip_radius": round(genes["tip_radius"], 4),
        "tip_length": round(tip_length, 4),
        "rotor_radius": round(genes["rotor_radius"], 4),
        "rotor_thickness": round(genes["rotor_thickness"], 4),
        "hub_x": round(hub_x, 4),
        "hub_y": round(hub_y, 4),
        "hub_z": round(hub_z, 4),
        "front_yaw": round(front_yaw, 6),
        "rear_yaw": round(rear_yaw, 6),
        "body_z": round(genes["body_z"], 4),
        "rotor_mount_z": round(rotor_mount_z, 4),
    }
    return p


def phenotype_to_xacro_args(p):
    return {
        "robot_name": p["robot_name"],
        "body_length": p["body_length"],
        "body_width": p["body_width"],
        "body_height": p["body_height"],
        "body_mass": p["body_mass"],
        "hinge_mass": p["hinge_mass"],
        "arm_mass": p["arm_mass"],
        "tip_mass": p["tip_mass"],
        "rotor_mass": p["rotor_mass"],
        "arm_radius": p["arm_radius"],
        "arm_length": p["arm_length"],
        "tip_radius": p["tip_radius"],
        "tip_length": p["tip_length"],
        "rotor_radius": p["rotor_radius"],
        "rotor_thickness": p["rotor_thickness"],
        "hub_x": p["hub_x"],
        "hub_y": p["hub_y"],
        "hub_z": p["hub_z"],
        "front_yaw": p["front_yaw"],
        "rear_yaw": p["rear_yaw"],
        "body_z": p["body_z"],
        "rotor_mount_z": p["rotor_mount_z"],
    }


def alpha_schedule(count):
    if count < 2:
        return [0.0]
    return [i / (count - 1) for i in range(count)]


def main():
    parser = argparse.ArgumentParser(description="Generate correlated quadrotor morphologies from a genotype-phenotype family.")
    parser.add_argument("--xacro", required=True, help="Path to transformable_quad_gp.urdf.xacro")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--families", type=int, default=1, help="Number of genotype families")
    parser.add_argument("--per-family", type=int, default=10, help="Phenotypes per family")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed")
    args = parser.parse_args()

    xacro_file = Path(args.xacro).resolve()
    out_dir = Path(args.out_dir).resolve()
    urdf_dir = out_dir / "urdf"
    genotype_dir = out_dir / "genotypes"
    phenotype_dir = out_dir / "phenotypes"

    urdf_dir.mkdir(parents=True, exist_ok=True)
    genotype_dir.mkdir(parents=True, exist_ok=True)
    phenotype_dir.mkdir(parents=True, exist_ok=True)

    manifest = []

    for fam_idx in range(args.families):
        family_seed = args.seed + fam_idx
        genotype = genotype_family(family_seed)

        genotype_name = f"{genotype['family_id']}.json"
        genotype_path = genotype_dir / genotype_name
        with open(genotype_path, "w") as f:
            json.dump(genotype, f, indent=2)

        for ph_idx, alpha in enumerate(alpha_schedule(args.per_family)):
            phenotype = phenotype_from_genotype(genotype, alpha)

            ph_name = f"{genotype['family_id']}_morph_{ph_idx:02d}"
            phenotype["robot_name"] = ph_name

            phenotype_path = phenotype_dir / f"{ph_name}.json"
            with open(phenotype_path, "w") as f:
                json.dump(phenotype, f, indent=2)

            urdf_path = urdf_dir / f"{ph_name}.urdf"
            run_xacro(xacro_file, urdf_path, phenotype_to_xacro_args(phenotype))

            manifest.append({
                "family_id": genotype["family_id"],
                "morphology_id": ph_name,
                "alpha": phenotype["alpha"],
                "body_length": phenotype["body_length"],
                "body_width": phenotype["body_width"],
                "body_height": phenotype["body_height"],
                "arm_length": phenotype["arm_length"],
                "tip_length": phenotype["tip_length"],
                "front_yaw": phenotype["front_yaw"],
                "rear_yaw": phenotype["rear_yaw"],
                "hub_x": phenotype["hub_x"],
                "hub_y": phenotype["hub_y"],
                "hub_z": phenotype["hub_z"],
                "rotor_radius": phenotype["rotor_radius"],
                "rotor_mount_z": phenotype["rotor_mount_z"],
                "genotype": str(genotype_path),
                "phenotype": str(phenotype_path),
                "urdf": str(urdf_path),
            })

    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    with open(out_dir / "manifest.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        writer.writeheader()
        writer.writerows(manifest)

    print(f"Generated {args.families} genotype families.")
    print(f"Generated {len(manifest)} correlated phenotypes.")
    print(f"Generated URDFs in: {urdf_dir}")


if __name__ == "__main__":
    main()