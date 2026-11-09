"""Plot training curves with Mean ± Std from local TensorBoard event logs.

Extracts scalar time-series across multi-seed runs directly from TensorBoard event
files (zero W&B dependency, 100% offline), computes mean and standard deviation,
and generates publication-quality comparison figures with shaded confidence bands.

Usage:
    uv run python scripts/plot_tensorboard.py --exp-dir results/checkpoints/ppo-easy-full_seed2026
    uv run python scripts/plot_tensorboard.py --tb-dir results/checkpoints/ppo-easy-full_seed2026/tb_logs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Metrics to visualize by default
DEFAULT_METRICS = [
    ("eval/mean_revealed_coverage", "Revealed Coverage", (0.0, 1.05), True),
    ("eval/mean_poi_discovery_rate", "POI Discovery Rate", (0.0, 1.05), True),
    ("eval/mean_reward", "Mean Evaluation Reward", None, False),
    ("eval/death_rate", "Monster Death Rate", (0.0, 1.05), True),
]


def load_scalars_from_tb(tb_run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Extract step and value arrays for a specific scalar tag from a TensorBoard directory."""
    acc = EventAccumulator(str(tb_run_dir), size_guidance={"scalars": 0})
    acc.Reload()

    scalar_tags = acc.Tags().get("scalars", [])
    if tag not in scalar_tags:
        return np.array([]), np.array([])

    events = acc.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    return steps, values


def find_tb_run_dirs(root_dir: Path) -> list[Path]:
    """Find all subdirectories containing TensorBoard event files."""
    run_dirs = []
    for path in sorted(root_dir.glob("**/*events.out.tfevents*")):
        parent = path.parent
        if parent not in run_dirs:
            run_dirs.append(parent)
    return run_dirs


def aggregate_curves(
    runs_data: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align multiple runs by common timesteps and compute mean ± std."""
    if not runs_data:
        return np.array([]), np.array([]), np.array([])

    all_steps = sorted(list({s for steps, _ in runs_data for s in steps}))
    if not all_steps:
        return np.array([]), np.array([]), np.array([])

    step_grid = np.array(all_steps)
    interpolated = []

    for steps, values in runs_data:
        if len(steps) == 0:
            continue
        # Linear interpolation onto unified step grid within run bounds
        val_interp = np.interp(step_grid, steps, values)
        interpolated.append(val_interp)

    matrix = np.array(interpolated)  # Shape: (N_runs, N_steps)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0) if matrix.shape[0] > 1 else np.zeros_like(mean)

    return step_grid, mean, std


def plot_experiment(
    tb_dir: Path,
    output_file: Path | None = None,
    show_individual: bool = True,
) -> None:
    """Generate multi-panel figure plotting Mean ± Std for key metrics."""
    run_dirs = find_tb_run_dirs(tb_dir)
    if not run_dirs:
        print(f"Error: No TensorBoard event files found in {tb_dir}", file=sys.stderr)
        return

    print(f"Found {len(run_dirs)} run logs in {tb_dir}:")
    for r in run_dirs:
        print(f"  - {r.name}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    axes = axes.flatten()

    colors = ["#2563eb", "#059669", "#d97706", "#dc2626"]

    for idx, (tag, title, ylim, is_pct) in enumerate(DEFAULT_METRICS):
        ax = axes[idx]
        runs_series: list[tuple[np.ndarray, np.ndarray]] = []

        for r in run_dirs:
            steps, vals = load_scalars_from_tb(r, tag)
            if len(steps) > 0:
                runs_series.append((steps, vals))
                if show_individual and len(run_dirs) > 1:
                    ax.plot(
                        steps,
                        vals,
                        alpha=0.25,
                        linewidth=1.0,
                        linestyle="--",
                        label=r.name if idx == 0 else None,
                    )

        if runs_series:
            grid, mean, std = aggregate_curves(runs_series)
            main_color = colors[idx % len(colors)]
            ax.plot(
                grid,
                mean,
                color=main_color,
                linewidth=2.2,
                label=f"Mean (N={len(runs_series)})",
            )
            ax.fill_between(
                grid,
                mean - std,
                mean + std,
                color=main_color,
                alpha=0.2,
                label="± 1 Std",
            )

        ax.set_title(title, fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Environment Steps", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)
        if ylim is not None:
            ax.set_ylim(ylim)
        if is_pct:
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y*100:.0f}%"))
        ax.legend(loc="best", fontsize=9, framealpha=0.8)

    plt.suptitle(f"Training Progress (Mean ± Std) - {tb_dir.parent.name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = output_file or (tb_dir.parent / "tensorboard_learning_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    print(f"\n[Saved] High-resolution curve plot saved to: {save_path}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot offline learning curves with Mean ± Std from TensorBoard.")
    parser.add_argument("--exp-dir", type=str, default=None, help="Experiment root directory containing tb_logs/")
    parser.add_argument("--tb-dir", type=str, default=None, help="Direct path to tb_logs directory")
    parser.add_argument("--output", type=str, default=None, help="Output image file path (.png or .pdf)")
    parser.add_argument("--no-individual", action="store_true", help="Hide individual run dashed curves")

    args = parser.parse_args()

    if args.tb_dir:
        tb_dir = Path(args.tb_dir).resolve()
    elif args.exp_dir:
        exp_dir = Path(args.exp_dir).resolve()
        tb_dir = exp_dir / "tb_logs"
    else:
        # Default to latest experiment in results/checkpoints
        default_root = Path("results/checkpoints")
        if default_root.exists():
            exps = sorted([d for d in default_root.iterdir() if d.is_dir()])
            if exps:
                tb_dir = exps[-1] / "tb_logs"
            else:
                parser.error("No experiment directories found in results/checkpoints/")
        else:
            parser.error("Must specify either --exp-dir or --tb-dir")

    out_file = Path(args.output).resolve() if args.output else None
    plot_experiment(tb_dir, out_file, show_individual=not args.no_individual)


if __name__ == "__main__":
    main()

