#!/usr/bin/env python3
"""Write a sample airevolve Spherical Angular genome .npy file for testing genome_to_usd.py.

Genome format: (narms, 6) array, one row per rotor:
    [magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction]
See genome_to_usd.py / airevolve's SphericalAngularDroneGenomeHandler for details.

This writes a fixed, hand-picked 4-rotor genome: three flat quad-style arms plus one
tilted arm whose rotor is redirected sideways, to exercise both the plain and the
orientation-correction code paths in genome_to_usd.py.
"""
import argparse
import math
from pathlib import Path

import numpy as np

SAMPLE_GENOME = np.array([
    # magnitude, arm_rotation, arm_pitch, motor_rotation, motor_pitch, direction
    [0.15, 0.0,             0.0, 0.0, 0.0,          0],  # flat arm, thrust straight up, CCW
    [0.15, math.pi / 2,     0.0, 0.0, 0.0,          1],  # flat arm, thrust straight up, CW
    [0.15, math.pi,         0.0, 0.0, 0.0,          0],  # flat arm, thrust straight up, CCW
    [0.15, -math.pi / 2,    0.3, 0.0, math.pi / 2,  1],  # tilted arm, thrust redirected sideways, CW
])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "sample_genome.npy")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, SAMPLE_GENOME)
    print(f"Wrote sample genome ({SAMPLE_GENOME.shape[0]} rotors) to: {args.output}")


if __name__ == "__main__":
    main()
