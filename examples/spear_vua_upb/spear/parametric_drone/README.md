# Bird-Morph Drone Generator

This workflow generates a **continuous set of bird-like drone configurations** from a single parametric xacro file. Instead of producing unrelated random drones, it samples a smooth morph transition from a compact maneuvering pose to an extended gliding pose. Xacro is a good fit for this because it supports constants, simple math, and macros for reusable robot descriptions. [page:10]

## Concept

The drone is treated as a rotor-based approximation of a morphing bird planform:

- Two front wing stations.
- Two mid wing stations.
- Two tail stations.

A single morph parameter `morph_t` moves the design continuously between:
- **compact**: shorter span, more sweep, smaller tail spread,
- **extended**: larger span, less sweep, wider tail spread.

This gives you a set of configurations that are related by a smooth embodiment transition rather than by unrelated random changes. [page:10]

## Files

- `parametric_drone.urdf.xacro`: the morphable bird-like geometry backend. [page:10]
- `generate_random_drones.py`: generates sampled transition states and emits URDF files. [page:10]
- `manifest.json` / `manifest.csv`: record the parameters used for each generated configuration.
- `checking_generated_drones.py`: optional viewer for checking converted USD files in Isaac Sim / Isaac Lab. Isaac Lab standalone tools should be launched through `AppLauncher` and `isaaclab.sh -p ...`. [page:9]

## Generate URDFs

Run from the Isaac Lab repository root:

```bash
python generate_random_drones.py \
  --xacro source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/parametric_drone.urdf.xacro \
  --out-dir source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated \
  --count 10
```

This creates:

```text
source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated/
├── manifest.csv
├── manifest.json
└── urdf/
    ├── bird_morph_00.urdf
    ├── bird_morph_01.urdf
    ├── ...
    └── bird_morph_09.urdf
```

## Convert URDF to USD

Isaac Lab uses USD assets at runtime, so the generated URDF files must be converted before being loaded into scenes or tasks. The Isaac Lab standalone workflow is built around launching tools through `isaaclab.sh`. [page:9]

Example:

```bash
./isaaclab.sh -p scripts/tools/convert_urdf.py \
  source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated/urdf/bird_morph_00.urdf \
  source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated/usd/bird_morph_00.usd \
  --merge-joints \
  --headless
```

Batch-convert the whole folder with a shell loop or a helper script.

## Visual inspection

After conversion, inspect the generated USD files in Isaac Sim or with your standalone checker script. Isaac Lab’s standalone workflow requires launching the app first with `AppLauncher`, and many simulator modules cannot be imported before the app is launched. [page:9]

Typical launch pattern:

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/checking_generated_drones.py
```

## What the morph means

As `morph_t` increases:
- wing span increases,
- wing sweep decreases,
- tail spread increases,
- wing camber rises,
- the body becomes longer and slimmer.

This is meant to approximate a continuous bird-like airborne transition while staying compatible with a rotor-based URDF and the Isaac Lab asset pipeline. [page:10]

## Next step

Once the USD assets look good, the next step is to connect them to an Isaac Lab `ArticulationCfg` and define a control model that maps motor commands to thrust and torques. Isaac Lab’s modular design is intended for exactly this kind of custom robot workflow. [page:9]

# Load and check

./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/spear/00_load_single_usd_apply_wrench.py \
 --usd source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated_gp/usd/gptm_family_42_morph_06.usd \
 --allocation source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated_gp/allocation/gptm_family_42_morph_06.json \
 --z 1.5 \
 --max_rotor_thrust 15.0  \
 --device cuda:0  \
 --use_full_wrench

# PD controller
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/spear/01_load_single_usd_apply_wrench_hover.py \
 --usd source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated_gp/usd/gptm_family_42_morph_06.usd \
 --allocation source/isaaclab_tasks/isaaclab_tasks/direct/spear/parametric_drone/generated_gp/allocation/gptm_family_42_morph_06.json \
 --z 1.5 \
 --mass_kg 1.2 \
 --max_rotor_thrust 15.0  \
 --device cuda:0  \
 --use_full_wrench