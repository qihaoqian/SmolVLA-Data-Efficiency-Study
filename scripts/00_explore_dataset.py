#!/usr/bin/env python
"""
Step 0: Explore dataset structure and verify setup.

Run from the lerobot root directory:
    pipenv run python project1_vla_data_efficiency/scripts/00_explore_dataset.py
"""
from lerobot.datasets.lerobot_dataset import LeRobotDataset

DATASET_REPO = "lerobot/libero_spatial_image"


def main():
    print(f"Loading {DATASET_REPO} ...")
    ds = LeRobotDataset(DATASET_REPO)

    print(f"\n=== Dataset Overview ===")
    print(f"  num_episodes : {ds.num_episodes}")
    print(f"  num_frames   : {ds.num_frames}")
    print(f"  fps          : {ds.fps}")
    avg_len = ds.num_frames / ds.num_episodes
    print(f"  avg ep len   : {avg_len:.1f} frames  ({avg_len/ds.fps:.1f}s)")

    print(f"\n=== Features ===")
    for key, feat in ds.features.items():
        print(f"  {key:45s} {feat}")

    # Episode-level task distribution
    print(f"\n=== Task Distribution ===")
    import pandas as pd
    df = ds.hf_dataset.select_columns(["episode_index", "task_index", "frame_index"]).to_pandas()
    # episode length = number of frames per episode
    ep_lengths = df.groupby("episode_index")["frame_index"].count()
    # task per episode (first frame's task_index)
    ep_task = df.groupby("episode_index")["task_index"].first()

    tasks: dict[int, list[int]] = {}
    for ep_i, t in ep_task.items():
        tasks.setdefault(int(t), []).append(int(ep_i))

    for task_id in sorted(tasks.keys()):
        eps = tasks[task_id]
        lengths = ep_lengths[eps].values
        print(
            f"  task {task_id:2d}: {len(eps):3d} episodes  "
            f"len={lengths.mean():.0f}±{lengths.std():.0f} frames"
        )

    print(f"\n=== Subsample Preview ===")
    fractions = [1.0, 0.5, 0.25, 0.10, 0.05]
    for frac in fractions:
        n = sum(max(1, round(len(eps) * frac)) for eps in tasks.values())
        print(f"  {int(frac*100):3d}%  →  {n:4d} episodes")

    print("\nDone. Environment looks good!")


if __name__ == "__main__":
    main()
