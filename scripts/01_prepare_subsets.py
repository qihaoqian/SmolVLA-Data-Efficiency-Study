#!/usr/bin/env python
"""
Step 1: Generate stratified episode subsets for each data fraction.

Sampling is stratified per task so each fraction contains equal proportions
of demonstrations from every task. This prevents accidental task imbalance
that would confound the data efficiency comparison.

Output: subsets/frac{pct}.json for each fraction, containing:
  {
    "fraction": 0.25,
    "n_episodes": 108,
    "episodes_per_task": {0: 11, 1: 11, ...},
    "episodes": [3, 7, 12, ...]   ← sorted episode indices
  }

Run from the lerobot root directory:
    pipenv run python project1_vla_data_efficiency/scripts/01_prepare_subsets.py
"""
import json
import random
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_REPO = "lerobot/libero_spatial_image"
FRACTIONS = [1.0, 0.50, 0.25, 0.10, 0.05]
SEED = 42
OUTPUT_DIR = Path(__file__).parent.parent / "subsets"
# ─────────────────────────────────────────────────────────────────────────────


def get_episodes_by_task(ds: LeRobotDataset) -> dict[int, list[int]]:
    """Return {task_id: [episode_index, ...]} using the hf_dataset."""
    import pandas as pd
    df = ds.hf_dataset.select_columns(["episode_index", "task_index"]).to_pandas()
    ep_task = df.groupby("episode_index")["task_index"].first()

    episodes_by_task: dict[int, list[int]] = {}
    for ep_i, task_id in ep_task.items():
        episodes_by_task.setdefault(int(task_id), []).append(int(ep_i))
    return episodes_by_task


def stratified_sample(
    episodes_by_task: dict[int, list[int]],
    fraction: float,
    rng: random.Random,
) -> list[int]:
    """Sample `fraction` of episodes from each task, at least 1 per task."""
    selected = []
    for task_id in sorted(episodes_by_task.keys()):
        eps = episodes_by_task[task_id].copy()
        rng.shuffle(eps)
        n = max(1, round(len(eps) * fraction))
        selected.extend(eps[:n])
    return sorted(selected)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)

    print(f"Loading {DATASET_REPO} ...")
    ds = LeRobotDataset(DATASET_REPO)
    episodes_by_task = get_episodes_by_task(ds)

    print(f"Total episodes: {ds.num_episodes} across {len(episodes_by_task)} tasks\n")

    for frac in FRACTIONS:
        episodes = stratified_sample(episodes_by_task, frac, rng)
        per_task = {}
        ep_set = set(episodes)
        for task_id, eps in episodes_by_task.items():
            per_task[str(task_id)] = sum(1 for e in eps if e in ep_set)

        pct = int(frac * 100)
        fname = OUTPUT_DIR / f"frac{pct:03d}.json"
        data = {
            "fraction": frac,
            "n_episodes": len(episodes),
            "episodes_per_task": per_task,
            "episodes": episodes,
        }
        with open(fname, "w") as f:
            json.dump(data, f, indent=2)

        print(f"frac{pct:03d}  {len(episodes):4d} eps  per-task: {list(per_task.values())}  → {fname.name}")

    print("\nDone. Subsets written to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
