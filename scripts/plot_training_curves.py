"""Flexible Multi-Checkpoint Training Curve Plotter for MiniGrid Explorer.

Plots publication-quality comparative training curves from local TensorBoard event logs
across arbitrary model checkpoints and customizable metric selections.

Key Capabilities:
  1. Compare any list of checkpoint directories on the same plot(s).
  2. Select one or more specific metrics (e.g. coverage, reward, loss) or preset bundles (eval, loss, all).
  3. Automatic X-axis alignment across variable training horizons:
     If models were trained for different total timesteps (e.g. 300k vs 500k), the X-axis
     automatically uses the longest horizon as reference (0 to max timesteps).
     Shorter runs stop naturally at their training completion point.
  4. Adaptive layout: 1 metric -> 1 clean chart; multiple metrics -> structured subplot grid.

Supported Metrics:
  Evaluation Metrics:
    - coverage, revealed_coverage : Revealed map area coverage (%) [eval/mean_revealed_coverage]
    - visited, visited_coverage   : Walkable floor visited coverage (%) [eval/mean_visited_coverage]
    - poi, poi_discovery          : Points of Interest (POI) discovery rate (%) [eval/mean_poi_discovery_rate]
    - death, mortality            : Monster fatal collision rate (%) [eval/death_rate]
    - reward, return              : Mean episodic evaluation return [eval/mean_reward]
    - steps, ep_length            : Mean evaluation episode steps [eval/mean_ep_length]
  Optimization / Loss Metrics:
    - loss                        : Total combined PPO loss [train/loss]
    - value_loss                  : Critic value MSE loss [train/value_loss]
    - policy_loss                 : Actor policy gradient surrogate loss [train/policy_gradient_loss]
    - entropy                     : Policy exploration entropy loss [train/entropy_loss]
    - explained_var               : Critic explained variance ratio [train/explained_variance]
    - kl, approx_kl               : Approximate policy KL divergence [train/approx_kl]
  Preset Bundles:
    - eval                        : All 6 Evaluation metrics in a 2x3 grid
    - loss                        : All 6 Optimization/Loss metrics in a 2x3 grid
    - all                         : All 12 metrics in a 4x3 grid

Output Location:
  - By default, figures are saved into --output-dir (default: 'results/plots/').
  - If --output is a bare filename (e.g. 'comparison.png'), it is saved into <output-dir>/<output>.
  - If --output is an explicit/absolute path, it is saved directly to that path.
  - If --output is omitted, a timestamped filename like 'plot_<metrics>_<timestamp>.png' is created.

Usage Examples:
  # 1. Compare coverage across two models (single plot):
  uv run python scripts/plot_all_training_curves.py \\
      --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds \\
      --metrics coverage

  # 2. Compare 3 metrics across 3 models:
  uv run python scripts/plot_all_training_curves.py \\
      --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds recurrent_ppo_easy_5seeds \\
      --metrics coverage reward death \\
      --output easy_comparison.png

  # 3. Plot full evaluation metric suite (2x3 grid):
  uv run python scripts/plot_all_training_curves.py \\
      --checkpoints ppo_medium_5seeds ppo_nomem_medium_5seeds \\
      --metrics eval
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Ensure repository root is in python path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


# -----------------------------------------------------------------------------
# Metric Definition and Registry
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class MetricDef:
    """Specification of a scalar metric extracted from TensorBoard logs."""

    key: str
    tag: str
    label: str
    ylim: tuple[float, float] | None
    scale: float
    description: str


METRIC_REGISTRY: dict[str, MetricDef] = {
    # --- Evaluation Suite ---
    "coverage": MetricDef(
        key="coverage",
        tag="eval/mean_revealed_coverage",
        label="Revealed Coverage (%)",
        ylim=(0.0, 105.0),
        scale=100.0,
        description="Percentage of map area revealed from fog of war (%)",
    ),
    "visited": MetricDef(
        key="visited",
        tag="eval/mean_visited_coverage",
        label="Visited Floor (%)",
        ylim=(0.0, 105.0),
        scale=100.0,
        description="Percentage of walkable floor tiles physically visited by the agent (%)",
    ),
    "poi": MetricDef(
        key="poi",
        tag="eval/mean_poi_discovery_rate",
        label="POI Discovery Rate (%)",
        ylim=(0.0, 105.0),
        scale=100.0,
        description="Percentage of Points of Interest (POIs) discovered (%)",
    ),
    "death": MetricDef(
        key="death",
        tag="eval/death_rate",
        label="Monster Mortality Rate (%)",
        ylim=(0.0, 105.0),
        scale=100.0,
        description="Percentage of episodes ending in monster collision death (%)",
    ),
    "reward": MetricDef(
        key="reward",
        tag="eval/mean_reward",
        label="Mean Validation Return",
        ylim=None,
        scale=1.0,
        description="Mean cumulative return across the validation suite",
    ),
    "steps": MetricDef(
        key="steps",
        tag="eval/mean_ep_length",
        label="Mean Episode Length (steps)",
        ylim=None,
        scale=1.0,
        description="Mean survival length per episode (steps)",
    ),
    # --- Optimization / Loss Suite ---
    "loss": MetricDef(
        key="loss",
        tag="train/loss",
        label="Total Combined Loss",
        ylim=None,
        scale=1.0,
        description="Total combined optimization loss of the PPO algorithm",
    ),
    "value_loss": MetricDef(
        key="value_loss",
        tag="train/value_loss",
        label="Critic Value Loss (MSE)",
        ylim=None,
        scale=1.0,
        description="Mean squared error of the Critic value function prediction",
    ),
    "policy_loss": MetricDef(
        key="policy_loss",
        tag="train/policy_gradient_loss",
        label="Policy Gradient Loss",
        ylim=None,
        scale=1.0,
        description="Clipped surrogate objective loss of the Actor network",
    ),
    "entropy": MetricDef(
        key="entropy",
        tag="train/entropy_loss",
        label="Policy Entropy Loss",
        ylim=None,
        scale=1.0,
        description="Policy entropy reflecting the level of stochastic exploration",
    ),
    "explained_var": MetricDef(
        key="explained_var",
        tag="train/explained_variance",
        label="Explained Variance",
        ylim=(-0.1, 1.05),
        scale=1.0,
        description="Fraction of return variance explained by the Critic (ideal = 1.0)",
    ),
    "kl": MetricDef(
        key="kl",
        tag="train/approx_kl",
        label="Approximate KL Divergence",
        ylim=None,
        scale=1.0,
        description="Approximate policy distribution divergence between PPO updates",
    ),
}

# Alternate user-friendly aliases
ALIASES: dict[str, str] = {
    "revealed_coverage": "coverage",
    "revealed": "coverage",
    "visited_coverage": "visited",
    "floor": "visited",
    "poi_discovery": "poi",
    "pois": "poi",
    "mortality": "death",
    "deaths": "death",
    "return": "reward",
    "ep_length": "steps",
    "episode_length": "steps",
    "survival": "steps",
    "pg_loss": "policy_loss",
    "approx_kl": "kl",
}

# Metric bundles
BUNDLES: dict[str, list[str]] = {
    "eval": ["coverage", "visited", "poi", "death", "reward", "steps"],
    "loss": ["loss", "value_loss", "policy_loss", "entropy", "explained_var", "kl"],
    "all": [
        "coverage", "visited", "poi", "death", "reward", "steps",
        "loss", "value_loss", "policy_loss", "entropy", "explained_var", "kl",
    ],
}

# Professional, colorblind-friendly publication palette
PALETTE: list[str] = [
    "#1f77b4",  # Blue
    "#ff7f0e",  # Orange
    "#2ca02c",  # Green
    "#d62728",  # Red
    "#9467bd",  # Purple
    "#8c564b",  # Brown
    "#e377c2",  # Pink
    "#7f7f7f",  # Gray
    "#bcbd22",  # Olive
    "#17becf",  # Cyan
]


def resolve_requested_metrics(metric_inputs: list[str]) -> list[MetricDef]:
    """Parse metric inputs (names, aliases, or bundles) into ordered list of MetricDefs."""
    resolved_keys: list[str] = []
    for item in metric_inputs:
        cleaned = item.lower().strip()
        if cleaned in BUNDLES:
            resolved_keys.extend(BUNDLES[cleaned])
        else:
            canonical = ALIASES.get(cleaned, cleaned)
            if canonical in METRIC_REGISTRY:
                resolved_keys.append(canonical)
            else:
                # Direct check if user passed full TensorBoard tag
                matched_by_tag = None
                for m_def in METRIC_REGISTRY.values():
                    if m_def.tag.lower() == cleaned:
                        matched_by_tag = m_def.key
                        break
                if matched_by_tag:
                    resolved_keys.append(matched_by_tag)
                else:
                    print(
                        f"[Warning] Unknown metric '{item}'. "
                        f"Valid choices: {list(METRIC_REGISTRY.keys()) + list(BUNDLES.keys())}",
                        file=sys.stderr,
                    )

    # Deduplicate preserving insertion order
    seen: set[str] = set()
    final_defs: list[MetricDef] = []
    for k in resolved_keys:
        if k not in seen and k in METRIC_REGISTRY:
            seen.add(k)
            final_defs.append(METRIC_REGISTRY[k])

    return final_defs


# -----------------------------------------------------------------------------
# Data Extraction & Alignment
# -----------------------------------------------------------------------------
def load_scalar_series(tb_run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
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


def find_tb_run_dirs(folder_path: Path) -> list[Path]:
    """Locate all subdirectories containing TensorBoard event files."""
    run_dirs: list[Path] = []
    for path in sorted(folder_path.glob("**/*events.out.tfevents*")):
        parent = path.parent
        if parent not in run_dirs:
            run_dirs.append(parent)
    return run_dirs


def aggregate_runs(
    runs_data: list[tuple[np.ndarray, np.ndarray]],
    scale_factor: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align multiple runs of a model by common timesteps and compute mean ± std."""
    valid_runs = [(steps, vals * scale_factor) for steps, vals in runs_data if len(steps) > 0]
    if not valid_runs:
        return np.array([]), np.array([]), np.array([])

    # Create step grid up to this model's maximum timestep
    all_steps = sorted(list({s for steps, _ in valid_runs for s in steps}))
    step_grid = np.array(all_steps)

    interpolated: list[np.ndarray] = []
    for steps, values in valid_runs:
        val_interp = np.interp(step_grid, steps, values)
        interpolated.append(val_interp)

    matrix = np.array(interpolated)  # Shape: (num_runs, num_steps)
    mean = np.mean(matrix, axis=0)
    std = np.std(matrix, axis=0) if matrix.shape[0] > 1 else np.zeros_like(mean)
    return step_grid, mean, std


def resolve_checkpoint_dir(target_str: str, ckpt_root: Path) -> tuple[str, Path] | None:
    """Resolve a user-provided target string to a folder path and clean display label."""
    cleaned = target_str.strip()
    if not cleaned:
        return None

    target_path = Path(cleaned)
    if target_path.exists() and target_path.is_dir():
        return target_path.name, target_path

    rel_path = ckpt_root / cleaned
    if rel_path.exists() and rel_path.is_dir():
        return rel_path.name, rel_path

    # Substring match in ckpt_root
    matches = [d for d in ckpt_root.iterdir() if d.is_dir() and cleaned.lower() in d.name.lower()]
    if matches:
        matches.sort(key=lambda p: len(p.name))
        return matches[0].name, matches[0]

    return None


# -----------------------------------------------------------------------------
# Plotting Engine
# -----------------------------------------------------------------------------
def plot_flexible_training_curves(
    checkpoints_data: dict[str, list[Path]],
    metrics: list[MetricDef],
    out_file: Path,
    title: str | None = None,
    dpi: int = 300,
) -> None:
    """Generate and save multi-checkpoint comparison curves with dynamic layout.

    Handles variable training horizons by finding the maximum total timesteps across
    all models and setting the X-axis limit to max(T_all). Shorter runs stop naturally.
    """
    if not metrics:
        print("[Error] No valid metrics to plot.", file=sys.stderr)
        return

    n_metrics = len(metrics)
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # Determine layout dimensions
    if n_metrics == 1:
        nrows, ncols = 1, 1
        figsize = (10, 6)
    elif n_metrics == 2:
        nrows, ncols = 1, 2
        figsize = (15, 6)
    elif n_metrics <= 4:
        nrows, ncols = 2, 2
        figsize = (15, 10)
    elif n_metrics <= 6:
        nrows, ncols = 2, 3
        figsize = (18, 10)
    else:
        ncols = 3
        nrows = int(np.ceil(n_metrics / 3))
        figsize = (18, 4 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes_arr = np.array(axes).flatten() if n_metrics > 1 else np.array([axes])

    # Pre-extract data and compute global maximum timestep across all models and metrics
    extracted_data: dict[str, dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    global_max_step = 0

    for model_label, run_dirs in checkpoints_data.items():
        extracted_data[model_label] = {}
        for m_def in metrics:
            runs_series = [load_scalar_series(r, m_def.tag) for r in run_dirs]
            steps, mean, std = aggregate_runs(runs_series, scale_factor=m_def.scale)
            extracted_data[model_label][m_def.key] = (steps, mean, std)
            if len(steps) > 0 and steps[-1] > global_max_step:
                global_max_step = int(steps[-1])

    if global_max_step == 0:
        print("[Error] No scalar events could be extracted from the specified checkpoints.", file=sys.stderr)
        plt.close(fig)
        return

    # Render each metric on its corresponding subplot
    for idx, m_def in enumerate(metrics):
        ax = axes_arr[idx]

        for m_idx, (model_label, _) in enumerate(checkpoints_data.items()):
            color = PALETTE[m_idx % len(PALETTE)]
            steps, mean, std = extracted_data[model_label].get(
                m_def.key, (np.array([]), np.array([]), np.array([]))
            )

            if len(steps) == 0:
                continue

            # Plot mean line
            ax.plot(steps, mean, label=model_label, color=color, linewidth=2.0)
            # Plot shaded +/- 1 std confidence band
            ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.18)

        ax.set_title(m_def.label, fontsize=12, fontweight="bold", pad=8)
        ax.set_ylabel(m_def.label, fontsize=10)
        ax.set_xlabel("Environment Steps (Total Timesteps)", fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.5)

        # Reference X-axis: 0 to maximum timesteps across all models
        ax.set_xlim(0, global_max_step)
        ax.xaxis.set_major_formatter(
            plt.FuncFormatter(lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x)))
        )

        if m_def.ylim is not None:
            ax.set_ylim(m_def.ylim)

    # Hide any unused subplots
    for unused_idx in range(n_metrics, len(axes_arr)):
        axes_arr[unused_idx].set_visible(False)

    # Unified top legend
    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.99),
            ncol=min(len(labels), 4),
            frameon=True,
            facecolor="white",
            edgecolor="#cccccc",
            fontsize=11,
        )

    # Determine figure super-title: custom title if provided, else auto-generated
    if title:
        title_text = title
    elif n_metrics == 1:
        title_text = f"Training Progress: {metrics[0].label}"
    else:
        title_text = f"Multi-Checkpoint Training Comparison ({len(checkpoints_data)} Models)"

    plt.suptitle(title_text, fontsize=14, fontweight="bold", y=1.02 if n_metrics > 1 else 1.05)
    plt.tight_layout(rect=[0, 0, 1, 0.95 if n_metrics > 1 else 0.92])

    out_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_file, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Saved] High-resolution training curve figure successfully saved to:")
    print(f"  --> {out_file}\n")


# -----------------------------------------------------------------------------
# CLI Entrypoint
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Flexible Multi-Checkpoint Training Curve Plotter for MiniGrid Explorer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Supported Metric Shortcuts:
  Evaluation:
    coverage      : Revealed map coverage (%) [0..100%]
    visited       : Floor tiles visited (%) [0..100%]
    poi           : POI discovery rate (%) [0..100%]
    death         : Monster death / mortality rate (%) [0..100%]
    reward        : Mean validation return
    steps         : Mean episode steps / survival horizon
  Optimization:
    loss          : Total PPO loss
    value_loss    : Critic value MSE loss
    policy_loss   : Actor surrogate gradient loss
    entropy       : Policy entropy loss
    explained_var : Critic explained variance ratio [-0.1..1.0]
    kl            : Approximate KL divergence
  Preset Bundles:
    eval          : All 6 evaluation metrics (2x3 grid)
    loss          : All 6 optimization loss metrics (2x3 grid)
    all           : All 12 metrics (4x3 grid)

Output Saving Rules:
  - If --output is omitted, saves to: <output-dir>/plot_<metrics>_<timestamp>.png
  - If --output is a filename (e.g. 'chart.png'), saves to: <output-dir>/chart.png
  - If --output is an absolute or explicit path, saves directly to that path.
        """,
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=None,
        help="List of checkpoint folder names in results/checkpoints/ or paths to compare.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Single checkpoint folder name or path (shortcut for --checkpoints).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["coverage"],
        help="Metrics to plot (e.g. coverage, reward, loss) or bundles ('eval', 'loss', 'all').",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=ROOT_DIR / "results" / "checkpoints",
        help="Root directory containing checkpoint folders (default: results/checkpoints).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "results" / "plots",
        help="Target directory where generated plots are saved (default: results/plots).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output filename or path. Bare filenames are saved into --output-dir.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Custom super-title for the figure. If omitted, auto-generates a descriptive title.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Figure output resolution DPI (default: 300).",
    )

    args = parser.parse_args()

    # Collect checkpoint targets
    raw_targets: list[str] = []
    if args.checkpoints:
        raw_targets.extend(args.checkpoints)
    elif args.checkpoint:
        raw_targets.append(args.checkpoint)
    else:
        print("[Error] No checkpoints specified! Please provide at least one folder via --checkpoints.", file=sys.stderr)
        print("Example: uv run python scripts/plot_all_training_curves.py --checkpoints ppo_easy_5seeds ppo_nomem_easy_5seeds", file=sys.stderr)
        sys.exit(1)

    ckpt_root = args.checkpoint_dir
    checkpoints_data: dict[str, list[Path]] = {}

    print("\n" + "=" * 80)
    print(" MiniGrid Explorer: Flexible Training Curve Comparison Plotter")
    print("=" * 80)

    for target in raw_targets:
        resolved = resolve_checkpoint_dir(target, ckpt_root)
        if resolved is None:
            print(f"[Warning] Checkpoint target '{target}' could not be resolved! Skipping.")
            continue

        label, folder_path = resolved
        run_dirs = find_tb_run_dirs(folder_path)
        if not run_dirs:
            print(f"[Warning] No TensorBoard event logs found in '{folder_path}'! Skipping.")
            continue

        checkpoints_data[label] = run_dirs
        print(f"  - Loaded '{label}': {len(run_dirs)} run log(s) found.")

    if not checkpoints_data:
        print("\n[Error] No valid checkpoint directories with TensorBoard logs were found.", file=sys.stderr)
        sys.exit(1)

    # Resolve metric specifications
    resolved_metrics = resolve_requested_metrics(args.metrics)
    if not resolved_metrics:
        print(f"\n[Error] No valid metrics matched from inputs: {args.metrics}", file=sys.stderr)
        sys.exit(1)

    print(f"  - Metrics to plot: {[m.label for m in resolved_metrics]}")

    # Determine output file path according to rules:
    # 1. If --output is an absolute or explicit path -> use it directly.
    # 2. If --output is a bare filename -> save inside args.output_dir.
    # 3. If omitted -> auto-generate 'plot_<metrics>_<timestamp>.png' inside args.output_dir.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.output:
        custom_out = Path(args.output)
        if custom_out.is_absolute() or len(custom_out.parts) > 1:
            out_file = custom_out.resolve()
        else:
            out_file = (args.output_dir / args.output).resolve()
    else:
        metric_slug = "_".join([m.key for m in resolved_metrics[:3]])
        if len(resolved_metrics) > 3:
            metric_slug += f"_plus{len(resolved_metrics) - 3}"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_file = (args.output_dir / f"plot_{metric_slug}_{timestamp}.png").resolve()

    # Generate and save plot
    plot_flexible_training_curves(
        checkpoints_data=checkpoints_data,
        metrics=resolved_metrics,
        out_file=out_file,
        title=args.title,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
