"""Render a recorded slalom flight to mp4.

Top-down is the right projection for a slalom: the task is lateral weaving at
constant altitude, so a plan view shows the whole course and every gate plane at
once, where a perspective 3D view would hide the far gates behind the near ones.
Altitude and bank are carried in the lower strip instead.

The drone is drawn from its own genome, so the morphology under test is visible
in the video rather than implied by the filename.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter

from ariel.simulation.drone.gate_metrics import crossing_report, n_passed

INK, ACCENT, GOOD, BAD, MUTED = "#1a202c", "#2b6cb0", "#2f855a", "#c53030", "#a0aec0"


def render(path: Path, log: dict, course, genome: np.ndarray, *,
           title: str, subtitle: str, prop_radius: float, fps: int = 50,
           dpi: int = 110) -> Path:
    """Write an mp4 of one flight. Returns the path written."""
    pos, ref, euler, t = log["pos"], log["ref"], log["euler"], log["t"]
    stride = max(1, int(round((1.0 / fps) / (t[1] - t[0])))) if len(t) > 1 else 1
    idx = np.arange(0, len(t), stride)
    # The flight now terminates ON the final gate crossing, so the last step is
    # rarely a multiple of the stride and subsampling would drop it -- the last
    # gate would never be seen turning green even though it was passed. Always
    # render the final step, then hold it so the finished course is readable.
    if idx[-1] != len(t) - 1:
        idx = np.append(idx, len(t) - 1)
    idx = np.append(idx, [len(t) - 1] * int(0.8 * fps))

    gates, gate_yaw = course.gate_pos, course.path_yaw[:len(course.gate_pos)]
    half = course.gate_size / 2.0
    # Verdict per gate, judged where it must be judged: in the gate plane, at
    # the crossing, altitude included.
    report = crossing_report(pos, gates, gate_yaw, course.gate_size)

    fig = plt.figure(figsize=(12.0, 6.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[3.0, 1.0], hspace=0.28)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # --- static course -----------------------------------------------------
    # The reference is the B-spline the controller actually chases, taken from
    # the flight log. Joining course.path_pos with straight lines would draw a
    # zigzag through the waypoints -- a path nothing ever follows.
    ax.plot(ref[:, 0], ref[:, 1], "-", color=MUTED, lw=1.1, zorder=1,
            label="reference (B-spline)")
    # Lead-out waypoints: not scored, but they are why the reference keeps
    # going past the last gate instead of decelerating onto it. Drawn so their
    # absence from the scoring is visible rather than implied.
    for g, yaw in zip(course.path_pos[len(gates):], course.path_yaw[len(gates):]):
        n = np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)]) * half
        ax.plot([g[0] - n[0], g[0] + n[0]], [g[1] - n[1], g[1] + n[1]], "-",
                color=MUTED, lw=2.0, alpha=0.7, solid_capstyle="butt", zorder=2,
                label="_lead-out")
    ax.plot([], [], "-", color=MUTED, lw=2.0, alpha=0.7, label="lead-out (unscored)")

    gate_lines = []
    for g, yaw in zip(gates, gate_yaw):
        n = np.array([np.cos(yaw + np.pi / 2), np.sin(yaw + np.pi / 2)]) * half
        (ln,) = ax.plot([g[0] - n[0], g[0] + n[0]], [g[1] - n[1], g[1] + n[1]],
                        "-", color=BAD, lw=3.0, solid_capstyle="butt", zorder=2)
        gate_lines.append(ln)

    pad = 1.2
    xs = np.concatenate([ref[:, 0], pos[:, 0], gates[:, 0]])
    ys = np.concatenate([ref[:, 1], pos[:, 1], gates[:, 1]])
    xs = np.concatenate([xs, course.path_pos[:, 0]])
    ys = np.concatenate([ys, course.path_pos[:, 1]])
    ax.set_xlim(xs.min() - pad, xs.max() + pad)
    ax.set_ylim(ys.min() - pad, ys.max() + pad)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_xlabel("x (m)", fontsize=9)
    ax.set_ylabel("y (m)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.set_title(f"{title}\n{subtitle}", fontsize=11, loc="left")
    ax.legend(fontsize=7, loc="lower left", ncol=3, framealpha=0.9)

    (trail,) = ax.plot([], [], "-", color=ACCENT, lw=1.6, zorder=4)
    (refdot,) = ax.plot([], [], "o", color=MUTED, ms=5, zorder=4)
    arms = [ax.plot([], [], "-", color=INK, lw=2.0, zorder=6)[0] for _ in genome]
    discs = [plt.Circle((0, 0), prop_radius, color=GOOD, alpha=0.45, zorder=5)
             for _ in genome]
    for d in discs:
        ax.add_patch(d)
    hud = ax.text(0.985, 0.04, "", transform=ax.transAxes, ha="right", va="bottom",
                  fontsize=10, family="monospace",
                  bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=MUTED, alpha=0.85))

    # --- lower strip: altitude and bank ------------------------------------
    # NED: z is down, so altitude is -z.
    gate_alt = float(-np.mean(gates[:, 2]))
    ax2.axhspan(gate_alt - half, gate_alt + half, color=GOOD, alpha=0.10, zorder=0)
    ax2.axhline(gate_alt + half, color=GOOD, lw=0.8, ls=":", zorder=1)
    ax2.axhline(gate_alt - half, color=GOOD, lw=0.8, ls=":", zorder=1)
    for c in report:
        if not c.crossed:
            continue
        ax2.axvline(t[c.step], color=GOOD if c.passed else BAD, lw=1.0,
                    alpha=0.65, zorder=1)
        ax2.plot([t[c.step]], [-pos[c.step, 2]], "o",
                 color=GOOD if c.passed else BAD, ms=4, zorder=6)
    ax2.plot(t, -ref[:, 2], "-", color=MUTED, lw=1.0, label="reference altitude")
    ax2.plot(t, -pos[:, 2], "-", color=ACCENT, lw=1.3, label="altitude (m)")
    lo_a = min(gate_alt - half, float((-pos[:, 2]).min())) - 0.05
    hi_a = max(gate_alt + half, float((-pos[:, 2]).max())) + 0.05
    ax2.set_ylim(lo_a, hi_a)
    ax2.set_ylabel("altitude (m)", fontsize=8, color=ACCENT)
    ax2.tick_params(axis="y", labelcolor=ACCENT)
    ax3 = ax2.twinx()
    ax3.plot(t, np.degrees(euler[:, 0]), "-", color=BAD, lw=1.0, alpha=0.85,
             label="roll (deg)")
    ax3.set_ylabel("roll (deg)", fontsize=8, color=BAD)
    ax3.tick_params(axis="y", labelcolor=BAD, labelsize=8)
    (cursor,) = ax2.plot([], [], "-", color=INK, lw=1.0)
    ax2.set_xlim(t[0], t[-1])
    ax2.grid(alpha=0.25, lw=0.4)
    ax2.set_xlabel("time (s)", fontsize=9)
    ax2.tick_params(labelsize=8)
    ax2.plot([], [], "-", color=GOOD, lw=0.8, ls=":", label="gate opening (±0.5 m)")
    lines = [l for l in ax2.get_lines() if l.get_label() and not l.get_label().startswith("_")]
    lines = lines[:3] + ax3.get_lines()[:1]
    ax2.legend(lines, [l.get_label() for l in lines],
               fontsize=7, loc="upper right", ncol=3, framealpha=0.9)

    lengths, az = genome[:, 0], genome[:, 1]

    writer = FFMpegWriter(fps=fps, bitrate=2400,
                          metadata={"title": title, "comment": subtitle})
    path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(path), dpi):
        for k in idx:
            p, yaw = pos[k], euler[k][2]
            c, s = np.cos(yaw), np.sin(yaw)
            for j, (ln, disc) in enumerate(zip(arms, discs)):
                bx, by = lengths[j] * np.cos(az[j]), lengths[j] * np.sin(az[j])
                wx, wy = p[0] + c * bx - s * by, p[1] + s * bx + c * by
                ln.set_data([p[0], wx], [p[1], wy])
                disc.center = (wx, wy)
            trail.set_data(pos[:k + 1, 0], pos[:k + 1, 1])
            refdot.set_data([ref[k][0]], [ref[k][1]])
            # Colour each gate once the drone has crossed its plane, by
            # whether it was inside the opening at that instant.
            for c, ln_g in zip(report, gate_lines):
                if c.crossed and k >= c.step:
                    ln_g.set_color(GOOD if c.passed else BAD)
                    ln_g.set_alpha(1.0)
                else:
                    ln_g.set_color(MUTED)
                    ln_g.set_alpha(0.9)
            speed = np.linalg.norm(pos[k] - pos[k - 1]) / (t[k] - t[k - 1]) if k else 0.0
            done = sum(1 for c in report if c.crossed and k >= c.step)
            ok = sum(1 for c in report if c.crossed and k >= c.step and c.passed)
            hud.set_text(f"t {t[k]:5.2f} s   v {speed:5.2f} m/s   "
                         f"roll {np.degrees(euler[k][0]):+6.1f}°   "
                         f"gates {ok}/{done}")
            cursor.set_data([t[k], t[k]], list(ax2.get_ylim()))
            writer.grab_frame()
    plt.close(fig)
    return path
