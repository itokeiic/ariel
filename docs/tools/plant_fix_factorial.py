#!/usr/bin/env python3
"""Which half of the 2026-09-10 plant fix moved each asymmetric body?

    uv run --no-sync python docs/tools/plant_fix_factorial.py VARIANT OUT.json \
        [--bodies=azimuth,length] [--matrix-gains]

The plant fix changed two things that matter for coplanar asymmetric bodies:
moment arms measured about the CG instead of the body origin, and angular
acceleration from the full inertia tensor instead of its diagonal. This flies
each reference body with a 2x2 factorial over those two, patching ONLY the
plant's parameter dict -- the controller (mixer from CG-relative Bm, gains from
`params["IB"]`, which is the configuration's inertia) is identical in every
variant:

  both          CG-relative arms, full inertia tensor      == current code
  cg_only       CG-relative arms, diagonal inertia
  inertia_only  body-origin arms, full inertia tensor
  neither       body-origin arms, diagonal inertia         == pre-fix plant

`neither` must reproduce docs/data/perturbation.csv and `both` the post-fix
perturbation run cell for cell; that is what validates the instrument, and it
did (12/12 cells, 2026-09-11). NOTE 2026-09-16: docs/data/perturbation.csv has
since been regenerated on the plant with the gyroscopic term and quadratic rotor
drag, so this validation reproduces only at commit 8d3174b or earlier. `--matrix-gains` switches the controller to
full-tensor attitude gains, to test whether an inertia effect is a
controller/plant mismatch rather than harder physics.

Operating point matches every reported sweep: strict completion, tuned cascade,
feedforward, --speed-tol 0.0625.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.simplefilter("ignore")
REPO = Path(__file__).resolve().parents[2]

VARIANT, OUT = sys.argv[1], Path(sys.argv[2])
extra = sys.argv[3:]
assert VARIANT in {"both", "cg_only", "inertia_only", "neither"}, VARIANT
bodies = ("azimuth", "length")
matrix_gains = False
for a in extra:
    if a.startswith("--bodies="):
        bodies = tuple(b for b in a.split("=", 1)[1].split(",") if b)
    elif a == "--matrix-gains":
        matrix_gains = True
    else:
        raise SystemExit(f"unknown argument {a!r}")

# The sweep parses its CLI at import, so bind the reported operating point first.
sys.argv = ["sweep", "--completion", "strict", "--speed-lo", "2", "--speed-hi", "12",
            "--speed-cap", "25", "--speed-tol", "0.0625", "--att-omega-n", "24",
            "--pos-omega-n", "2.0", "--pos-zeta", "1.0", "--feedforward", "--max-accel", "40"]
if matrix_gains:
    sys.argv.append("--matrix-gains")
sys.path.insert(0, str(REPO / "examples" / "spear"))
spec = importlib.util.spec_from_file_location(
    "sweep", REPO / "examples" / "spear" / "19_morphology_design_sweep.py")
sweep = importlib.util.module_from_spec(spec)
sys.modules["sweep"] = sweep
spec.loader.exec_module(sweep)

import ariel.simulation.drone.drone_simulator as ds  # noqa: E402
from ariel.simulation.drone.reference_morphologies import reference_morphologies  # noqa: E402
from ariel.simulation.tasks.slalom_course import slalom_gates  # noqa: E402

_orig = ds.derive_reference_params
calls = {"n": 0}


def _patched(*a, **kw):
    p = _orig(*a, **kw)
    calls["n"] += 1
    props = kw["propellers"] if "propellers" in kw else a[0]
    if VARIANT in ("inertia_only", "neither"):     # arms back to the body origin
        p["rotor_arms"] = np.array([np.asarray(q["loc"][:3], float) for q in props])
    if VARIANT in ("cg_only", "neither"):          # inertia back to its diagonal
        p["inertia"] = np.diag(np.diag(np.asarray(p["inertia"], float)))
    return p


ds.derive_reference_params = _patched

fam = reference_morphologies(arm_length=sweep.ARM_LENGTH)
res: dict = {}
for key in bodies:
    before = calls["n"]
    for turn in (60.0, 90.0, 120.0):
        course = slalom_gates(turn, leg=sweep.args.leg, n_gates=sweep.args.n_gates,
                              gate_size=sweep.args.gate_size)
        r = sweep.max_completing_speed(fam[key].genome, course)
        res[f"{key}@{turn:.0f}"] = {
            "max_speed": float(r["max_speed"]), "monotone": bool(r["monotone"]),
            "trk": float(r["tracking_err"]), "sat_lo": float(r["saturation_lo"])}
        print(VARIANT, "matrix_gains" if matrix_gains else "diag_gains",
              key, turn, r["max_speed"], flush=True)
    # A patch that silently misses would make every variant equal `both`.
    assert calls["n"] > before, f"patch never reached the plant for {key}"
res["_patch_calls"] = calls["n"]
res["_matrix_gains"] = matrix_gains
OUT.write_text(json.dumps(res, indent=1))
print("done", VARIANT, "patch calls", calls["n"])
