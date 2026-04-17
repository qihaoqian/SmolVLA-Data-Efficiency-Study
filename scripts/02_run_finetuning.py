#!/usr/bin/env python
"""
Step 2: Fine-tune SmolVLA on a data fraction.

Reads the episode list from subsets/frac{pct}.json and launches training
by directly building a TrainPipelineConfig and calling train().

Usage (from lerobot root):
    pipenv run python project1_vla_data_efficiency/scripts/02_run_finetuning.py --fraction 1.00
    pipenv run python project1_vla_data_efficiency/scripts/02_run_finetuning.py --fraction 0.50
    pipenv run python project1_vla_data_efficiency/scripts/02_run_finetuning.py --fraction 0.25
    pipenv run python project1_vla_data_efficiency/scripts/02_run_finetuning.py --fraction 0.10
    pipenv run python project1_vla_data_efficiency/scripts/02_run_finetuning.py --fraction 0.05

Each run trains for 30k steps and saves one checkpoint at the end.
Expected GPU memory: ~20GB (SmolVLA 500M, batch 16, 512×512 images).
Expected wall time:  ~2–4h on RTX 4090 / A100.
"""
import argparse
import json
import os
import sys
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
DATASET_REPO = "lerobot/libero_spatial_image"
PRETRAINED_PATH = "lerobot/smolvla_base"

TRAIN_STEPS = 30_000
BATCH_SIZE = 16
SAVE_FREQ = 30_000   # save only final checkpoint (saves disk space)
LOG_FREQ = 200
SEED = 42

# W&B — set WANDB_API_KEY env var or disable
WANDB_PROJECT = "vla_data_efficiency_libero_spatial"
WANDB_ENABLE = True  # set to False if you don't have W&B set up

PROJECT_ROOT = Path(__file__).parent.parent
SUBSETS_DIR = PROJECT_ROOT / "subsets"
OUTPUT_BASE = Path("outputs/project1")
# ─────────────────────────────────────────────────────────────────────────────


def fraction_to_label(fraction: float) -> str:
    pct = round(fraction * 100)
    return f"frac{pct:03d}"


def load_episodes(fraction: float) -> list[int]:
    label = fraction_to_label(fraction)
    path = SUBSETS_DIR / f"{label}.json"
    if not path.exists():
        print(f"ERROR: {path} not found. Run 01_prepare_subsets.py first.")
        sys.exit(1)
    with open(path) as f:
        data = json.load(f)
    episodes = data["episodes"]
    per_task = data["episodes_per_task"]
    print(f"Loaded {len(episodes)} episodes for fraction {fraction:.0%}")
    print(f"  per-task: {list(per_task.values())}")
    return episodes


def build_config(fraction: float, episodes: list[int]):
    """Build TrainPipelineConfig programmatically (no sys.argv manipulation needed)."""
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

    label = fraction_to_label(fraction)
    output_dir = OUTPUT_BASE / label

    dataset_cfg = DatasetConfig(
        repo_id=DATASET_REPO,
        episodes=episodes,
    )

    policy_cfg = SmolVLAConfig(
        pretrained_path=Path(PRETRAINED_PATH),
        load_vlm_weights=True,   # load VLM backbone from smolvla_base
        freeze_vision_encoder=True,
        train_expert_only=False,  # also train state projector
        train_state_proj=True,
        push_to_hub=False,  # local checkpoints only (PreTrainedConfig defaults push_to_hub=True)
    )

    wandb_cfg = WandBConfig(
        enable=WANDB_ENABLE and bool(os.environ.get("WANDB_API_KEY")),
        project=WANDB_PROJECT,
        notes=f"SmolVLA fine-tune on libero_spatial, fraction={fraction:.0%}",
    )
    if WANDB_ENABLE and not os.environ.get("WANDB_API_KEY"):
        print("WARNING: WANDB_API_KEY not set, disabling W&B logging.")

    cfg = TrainPipelineConfig(
        dataset=dataset_cfg,
        policy=policy_cfg,
        output_dir=output_dir,
        job_name=f"smolvla_libero_spatial_{label}",
        steps=TRAIN_STEPS,
        batch_size=BATCH_SIZE,
        save_freq=SAVE_FREQ,
        log_freq=LOG_FREQ,
        seed=SEED,
        num_workers=4,
        # No inline eval — we evaluate separately with 03_run_eval.sh
        eval_freq=0,
        wandb=wandb_cfg,
    )
    return cfg


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SmolVLA on a data fraction")
    parser.add_argument(
        "--fraction", type=float, required=True,
        help="Data fraction to use: 1.0, 0.5, 0.25, 0.10, 0.05"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print config and exit without training"
    )
    args = parser.parse_args()

    valid = [1.0, 0.50, 0.25, 0.10, 0.05]
    if args.fraction not in valid:
        print(f"ERROR: --fraction must be one of {valid}")
        sys.exit(1)

    episodes = load_episodes(args.fraction)
    cfg = build_config(args.fraction, episodes)

    label = fraction_to_label(args.fraction)
    print(f"\n{'='*60}")
    print(f"  Fraction : {args.fraction:.0%}  ({len(episodes)} episodes)")
    print(f"  Steps    : {TRAIN_STEPS:,}")
    print(f"  Batch    : {BATCH_SIZE}")
    print(f"  Output   : {cfg.output_dir}")
    print(f"  W&B      : {'enabled' if cfg.wandb.enable else 'disabled'}")
    print(f"{'='*60}\n")

    if args.dry_run:
        print("Dry run — exiting before training.")
        return

    # Register any third-party plugins (e.g., custom envs)
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()

    # Call train() directly with pre-built config (bypasses draccus CLI parsing)
    from lerobot.scripts.lerobot_train import train
    train(cfg)


if __name__ == "__main__":
    main()
