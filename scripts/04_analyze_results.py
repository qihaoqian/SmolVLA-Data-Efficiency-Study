#!/usr/bin/env python
"""
Step 4: Analyze results and generate plots.

Reads eval_info.json for each condition and produces:
  results/scaling_curve.png       -- main figure: success rate vs. demo count
  results/per_task_heatmap.png    -- per-task breakdown across all conditions
  results/per_task_lines.png      -- per-task line plots
  results/summary_table.csv       -- numerical summary

Run from the lerobot root directory:
    pipenv run python project1_vla_data_efficiency/scripts/04_analyze_results.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS_DIR = Path("results")

# (label, approx #demos, display name)
CONDITIONS = [
    ("zero_shot",  0,   "Zero-shot"),
    ("frac005",   21,   "5% (~2/task)"),
    ("frac010",   43,   "10% (~4/task)"),
    ("frac025",  108,   "25% (~11/task)"),
    ("frac050",  216,   "50% (~21/task)"),
    ("frac100",  432,   "100% (~43/task)"),
]

TASK_DESCRIPTIONS = {
    0: "between plate & ramekin",
    1: "next to ramekin",
    2: "table center",
    3: "on cookie box",
    4: "top drawer",
    5: "on ramekin",
    6: "next to cookie box",
    7: "on stove",
    8: "next to plate",
    9: "on wooden cabinet",
}
# ─────────────────────────────────────────────────────────────────────────────


def load_eval_info(results_dir: Path, label: str) -> dict | None:
    path = results_dir / label / "eval_info.json"
    if not path.exists():
        print(f"  [{label}] Not found: {path}")
        return None
    with open(path) as f:
        return json.load(f)


def parse_success_rates(info: dict) -> dict[int, float]:
    """Return {task_id: success_rate} from eval_info.json."""
    rates = {}
    for task_entry in info["per_task"]:
        tid = task_entry["task_id"]
        successes = task_entry["metrics"]["successes"]
        rates[tid] = sum(successes) / len(successes) * 100
    return rates


def build_dataframe() -> pd.DataFrame:
    rows = []
    for label, n_eps, display in CONDITIONS:
        info = load_eval_info(RESULTS_DIR, label)
        if info is None:
            continue
        rates = parse_success_rates(info)
        avg = np.mean(list(rates.values()))
        row = {
            "label": label,
            "n_episodes": n_eps,
            "display": display,
            "avg_success": avg,
        }
        for tid, rate in rates.items():
            row[f"task_{tid}"] = rate
        rows.append(row)
    return pd.DataFrame(rows)


def plot_scaling_curve(df: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))

    # All points
    x = df["n_episodes"].values
    y = df["avg_success"].values
    labels = df["display"].values

    ax.plot(x, y, "o-", color="#1f77b4", linewidth=2, markersize=8, zorder=3)

    # Annotate each point
    for xi, yi, lbl in zip(x, y, labels):
        offset = (5, 5) if xi > 0 else (5, -12)
        ax.annotate(
            f"{yi:.1f}%",
            xy=(xi, yi),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            color="#333333",
        )

    # Zero-shot horizontal reference
    if df[df["label"] == "zero_shot"].shape[0] > 0:
        zs = float(df[df["label"] == "zero_shot"]["avg_success"].iloc[0])
        ax.axhline(zs, linestyle="--", color="gray", alpha=0.6, label=f"Zero-shot ({zs:.1f}%)")
        ax.legend(fontsize=10)

    ax.set_xlabel("Number of demonstration episodes (total)", fontsize=12)
    ax.set_ylabel("Average success rate (%)", fontsize=12)
    ax.set_title("SmolVLA Data Efficiency on LIBERO-Spatial\n(10 tasks, 50 rollouts/task)", fontsize=13)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    # Log scale for x-axis (skip the 0 point)
    non_zero = df[df["n_episodes"] > 0]
    if not non_zero.empty:
        ax.set_xscale("symlog", linthresh=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_task_heatmap(df: pd.DataFrame, out_path: Path) -> None:
    task_cols = [f"task_{i}" for i in range(10) if f"task_{i}" in df.columns]
    if not task_cols:
        return

    data = df[task_cols].values  # [n_conditions, n_tasks]
    row_labels = df["display"].values
    col_labels = [TASK_DESCRIPTIONS.get(int(c.split("_")[1]), c) for c in task_cols]

    fig, ax = plt.subplots(figsize=(14, len(row_labels) * 0.7 + 1.5))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=10)

    for r in range(len(row_labels)):
        for c in range(len(task_cols)):
            v = data[r, c]
            color = "white" if v < 40 or v > 80 else "black"
            ax.text(c, r, f"{v:.0f}%", ha="center", va="center", fontsize=9, color=color)

    plt.colorbar(im, ax=ax, label="Success rate (%)", shrink=0.8)
    ax.set_title("Per-Task Success Rate by Data Fraction\n(SmolVLA, LIBERO-Spatial)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def plot_per_task_lines(df: pd.DataFrame, out_path: Path) -> None:
    task_cols = [f"task_{i}" for i in range(10) if f"task_{i}" in df.columns]
    if not task_cols:
        return

    fig, ax = plt.subplots(figsize=(9, 6))
    cmap = plt.cm.get_cmap("tab10", 10)

    for i, col in enumerate(task_cols):
        tid = int(col.split("_")[1])
        ax.plot(
            df["n_episodes"], df[col],
            "o-", color=cmap(i), linewidth=1.5, markersize=5,
            label=f"T{tid}: {TASK_DESCRIPTIONS.get(tid, col)[:20]}"
        )

    ax.set_xlabel("Number of episodes", fontsize=12)
    ax.set_ylabel("Success rate (%)", fontsize=12)
    ax.set_title("Per-Task Scaling Curves", fontsize=13)
    ax.set_ylim(0, 105)
    ax.legend(fontsize=7, loc="lower right", ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("symlog", linthresh=10)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{int(v)}"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


def print_summary_table(df: pd.DataFrame) -> None:
    task_cols = [f"task_{i}" for i in range(10) if f"task_{i}" in df.columns]
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)

    header = f"{'Condition':<22} {'#Eps':>6} {'Avg':>7}"
    for i in range(len(task_cols)):
        header += f"  T{i:1d}"
    print(header)
    print("-" * 80)

    for _, row in df.iterrows():
        line = f"{row['display']:<22} {row['n_episodes']:>6.0f} {row['avg_success']:>6.1f}%"
        for col in task_cols:
            if col in row:
                line += f"  {row[col]:>3.0f}"
        print(line)
    print("=" * 80)


def save_csv(df: pd.DataFrame, out_path: Path) -> None:
    df.to_csv(out_path, index=False, float_format="%.1f")
    print(f"Saved: {out_path}")


def print_insights(df: pd.DataFrame) -> None:
    task_cols = [f"task_{i}" for i in range(10) if f"task_{i}" in df.columns]
    if df.empty or not task_cols:
        return

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)

    if "zero_shot" in df["label"].values and "frac100" in df["label"].values:
        zs = float(df[df["label"] == "zero_shot"]["avg_success"].iloc[0])
        f100 = float(df[df["label"] == "frac100"]["avg_success"].iloc[0])
        print(f"  Zero-shot vs. 100% fine-tune: {zs:.1f}% → {f100:.1f}% (+{f100-zs:.1f}pp)")

    # Which task benefits most
    if "zero_shot" in df["label"].values and "frac100" in df["label"].values:
        zs_row = df[df["label"] == "zero_shot"].iloc[0]
        f100_row = df[df["label"] == "frac100"].iloc[0]
        improvements = {col: f100_row[col] - zs_row[col] for col in task_cols if col in zs_row}
        best_task = max(improvements, key=improvements.get)
        worst_task = min(improvements, key=improvements.get)
        tid_best = int(best_task.split("_")[1])
        tid_worst = int(worst_task.split("_")[1])
        print(f"  Most improved:  task {tid_best} ({TASK_DESCRIPTIONS.get(tid_best, '')})  +{improvements[best_task]:.1f}pp")
        print(f"  Least improved: task {tid_worst} ({TASK_DESCRIPTIONS.get(tid_worst, '')}) +{improvements[worst_task]:.1f}pp")

    # Find "good enough" knee (first condition exceeding 80% avg)
    for _, row in df[df["n_episodes"] > 0].iterrows():
        if row["avg_success"] >= 80.0:
            print(f"  First >80% avg: {row['display']} ({row['n_episodes']:.0f} episodes)")
            break

    print("=" * 80)


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading results...")
    df = build_dataframe()

    if df.empty:
        print("\nNo results found yet.")
        print("Run training (02_run_finetuning.py) and evaluation (03_run_eval.sh) first.")
        return

    print(f"Loaded {len(df)} conditions: {list(df['label'])}")

    print_summary_table(df)
    print_insights(df)
    save_csv(df, RESULTS_DIR / "summary_table.csv")

    plot_scaling_curve(df, RESULTS_DIR / "scaling_curve.png")
    plot_per_task_heatmap(df, RESULTS_DIR / "per_task_heatmap.png")
    plot_per_task_lines(df, RESULTS_DIR / "per_task_lines.png")

    print("\nAnalysis complete. Plots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
