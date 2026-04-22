# Project 1: VLA Data Efficiency Study on LIBERO

> **Resume description**: Fine-tuned SmolVLA on LIBERO-Spatial benchmark across 5 data
> regimes (5%–100% of demos). Quantified the zero-shot vs. fine-tuned gap and produced
> per-task scaling curves showing how demonstration count affects success rate.

---

## Background

Vision-Language-Action (VLA) models like SmolVLA are pretrained on large internet-scale
corpora and can zero-shot execute manipulation tasks via language instructions. But in
practice, teams always fine-tune on task-specific demonstrations. A key open question is:

**How many task demonstrations does a VLA actually need?**

This matters because collecting robot demos is expensive. If a VLA can achieve 80% of its
peak performance with only 25% of the data, practitioners can cut collection costs by 4×.

---

## Research Questions

1. What is the zero-shot baseline? (no fine-tuning)
2. How much does fine-tuning help on LIBERO-Spatial?
3. What is the minimum demo count for acceptable (>70%) performance?
4. Is the improvement uniform across tasks, or do some tasks benefit more?
5. Does the model overfit at very low data regimes?

---

## Setup

### Model: SmolVLA

- Base: `lerobot/smolvla_base` (500M VLM backbone + action expert)
- Fine-tuning（与 `02_run_finetuning.py` 一致）: freeze vision encoder，`train_expert_only=True`，`train_state_proj=False`
- Optimizer: AdamW, lr=1e-4, cosine decay to 2.5e-6 over 30k steps
- Action: chunk_size=50 (predict 50 steps at once)

### Dataset: `lerobot/libero_spatial_image`

- **10 tasks**, all testing spatial reasoning on black-bowl placement
- **432 total episodes** (~43 demos per task)
- **Observations**: agentview RGB + wrist RGB + 8D proprioception
- **Actions**: 7D (3D EEF pos delta, 3D rotation delta, 1D gripper)

#### Tasks (libero_spatial)


| ID  | Description                                                       |
| --- | ----------------------------------------------------------------- |
| 0   | pick up black bowl between the plate and ramekin → place on plate |
| 1   | pick up black bowl next to ramekin → place on plate               |
| 2   | pick up black bowl from table center → place on plate             |
| 3   | pick up black bowl on cookie box → place on plate                 |
| 4   | pick up black bowl in top drawer → place on plate                 |
| 5   | pick up black bowl on ramekin → place on plate                    |
| 6   | pick up black bowl next to cookie box → place on plate            |
| 7   | pick up black bowl on stove → place on plate                      |
| 8   | pick up black bowl next to plate → place on plate                 |
| 9   | pick up black bowl on wooden cabinet → place on plate             |


### Experimental Conditions


| Label     | Fraction | Episodes (total) | Demos/task |
| --------- | -------- | ---------------- | ---------- |
| zero-shot | 0%       | 0                | 0          |
| frac05    | 5%       | ~21              | ~2         |
| frac10    | 10%      | ~43              | ~4         |
| frac25    | 25%      | ~108             | ~11        |
| frac50    | 50%      | ~216             | ~21        |
| frac100   | 100%     | 432              | ~43        |


Subsampling is **stratified per task** — each fraction takes equal proportions from
every task to avoid task imbalance.

### Training Hyperparameters (same for all conditions)

以下为 `scripts/02_run_finetuning.py` 中的固定常量（不可通过 CLI 修改；改训练需在脚本内编辑）：


| 项            | 值                                                                  |
| ------------ | ------------------------------------------------------------------ |
| `steps`      | 30,000                                                             |
| `batch_size` | 64                                                                 |
| `eval_freq`  | 0（训练中不做在线评估，评估见 Step 3）                                            |
| `save_freq`  | 10,000（配合步数，实际主要保留最终 checkpoint）                                   |
| `log_freq`   | 100                                                                |
| `seed`       | 42                                                                 |
| 数据集          | `lerobot/libero_spatial_image`（子集由 `subsets/frac*.json` 指定）        |
| 预训练权重        | `lerobot/smolvla_base`                                             |
| WB 项目名       | `vla_data_efficiency_libero_spatial`（需设置 `WANDB_API_KEY`；未设置则自动关闭） |


策略侧（`SmolVLAConfig`）：`freeze_vision_encoder=True`，`train_expert_only=True`，`train_state_proj=False`，`load_vlm_weights=True`，`push_to_hub=False`。

### Evaluation Protocol

与 `scripts/03_run_eval.sh` 中调用 `lerobot_eval` 的设置一致：

- **50 episodes per task × 10 tasks = 500 total rollouts** 每次评估
- Metric: `success_rate` (%) = 成功回合占比
- `**--eval.batch_size=2`**（脚本内写死；并行环境规模由此决定，并非 10）
- 任务：`libero_spatial`，环境类型：`libero`

---

## Zero-Shot Baseline (Already Collected)

`smolvla_base` evaluated on all 10 LIBERO-Spatial tasks with **no fine-tuning**:


| Task    | Description (short)       | Success Rate |
| ------- | ------------------------- | ------------ |
| 0       | between plate and ramekin | 62%          |
| 1       | next to ramekin           | 94%          |
| 2       | table center              | 78%          |
| 3       | on cookie box             | 76%          |
| 4       | top drawer                | 72%          |
| 5       | on ramekin                | 36%          |
| 6       | next to cookie box        | 84%          |
| 7       | on stove                  | 80%          |
| 8       | next to plate             | 78%          |
| 9       | on wooden cabinet         | 70%          |
| **avg** |                           | **73.0%**    |


Source: `outputs/eval/2026-04-06/22-05-32_libero_smolvla/eval_info.json`

**Key observation**: Task 5 (on ramekin) is the hardest at 36% — it likely requires
precise spatial disambiguation that benefits most from fine-tuning.

---

## Directory Structure

```
<repo>/
├── README.md
├── subsets/                    ← generated by 01_prepare_subsets.py
│   ├── frac005.json … frac100.json
├── outputs/
│   └── finetuning-action-expert/   ← default --output-dir for 02_run_finetuning.py
│       └── fracXXX/checkpoints/last/pretrained_model
├── results/                    ← default --results-dir for 03_run_eval.sh
│   ├── zero_shot/eval_info.json  ← copy of zero-shot baseline (if source exists)
│   └── fracXXX/eval_info.json
└── scripts/
    ├── 00_explore_dataset.py
    ├── 01_prepare_subsets.py
    ├── 02_run_finetuning.py
    ├── 03_run_eval.sh
    └── 04_analyze_results.py
```

训练输出默认写入 `outputs/finetuning-action-expert/<fracXXX>/`（见下方 Step 2）。

---

## Step-by-Step Execution

**工作目录**：以下命令均在仓库根目录执行（与 LeRobot 作为依赖安装时的用法一致；若你的工程把本仓库嵌在 `lerobot` 根下，请将 `scripts/` 换为实际相对路径）。

### Step 0: Verify environment

```bash
pipenv run python scripts/00_explore_dataset.py
```

### Step 1: Generate stratified subsets

```bash
pipenv run python scripts/01_prepare_subsets.py
```

Creates `subsets/frac005.json` through `subsets/frac100.json`.

### Step 2: Fine-tune (`scripts/02_run_finetuning.py`)

**运行方式**

```bash
pipenv run python scripts/02_run_finetuning.py --fraction <FRAC> [选项]
```

**必选参数**


| 参数           | 说明                                                                            |
| ------------ | ----------------------------------------------------------------------------- |
| `--fraction` | 数据比例，**仅允许**：`1.0`、`0.50`、`0.25`、`0.10`、`0.05`（需已存在对应 `subsets/fracXXX.json`） |


**可选参数与默认值**


| 参数              | 默认值                                | 说明                                    |
| --------------- | ---------------------------------- | ------------------------------------- |
| `--output-dir`  | `outputs/finetuning-action-expert` | 训练根目录；实际写入路径为 `<output-dir>/fracXXX/` |
| `--num-workers` | `4`                                | DataLoader worker 数                   |
| `--dry-run`     | 关闭                                 | 仅打印配置并退出，不启动训练                        |


固定训练超参与模型设定见上文「Training Hyperparameters」表（如 `steps=30000`、`batch_size=64` 等）。单次在单卡上约 **2–4 小时**（4090 / A100 量级，依 IO 与 GPU 而定）。

**示例（按比例依次跑）**

```bash
pipenv run python scripts/02_run_finetuning.py --fraction 1.00
pipenv run python scripts/02_run_finetuning.py --fraction 0.50
pipenv run python scripts/02_run_finetuning.py --fraction 0.25
pipenv run python scripts/02_run_finetuning.py --fraction 0.10
pipenv run python scripts/02_run_finetuning.py --fraction 0.05
```

自定义输出目录示例：`pipenv run python scripts/02_run_finetuning.py --fraction 0.25 --output-dir /tmp/ft`。

**Checkpoint 路径**

```
<output-dir>/<fracXXX>/checkpoints/last/pretrained_model/
```

### Step 3: Evaluate (`scripts/03_run_eval.sh`)

**运行方式**

```bash
bash scripts/03_run_eval.sh [选项] [FRACTION ...]
```

**可选参数与默认值**


| 参数                           | 默认值                                         | 说明                                                                                |
| ---------------------------- | ------------------------------------------- | --------------------------------------------------------------------------------- |
| `--finetune-output-base-dir` | `<仓库根>/outputs/finetuning-action-expert`    | 与 Step 2 的 `--output-dir` 对齐；脚本会在其下查找 `fracXXX/checkpoints/last/pretrained_model` |
| `--results-dir`              | `<仓库根>/results`                             | 每条评估的 `eval_info.json` 等输出目录                                                      |
| 位置参数 `FRACTION ...`          | **未指定时**：`0.05`、`0.10`、`0.25`、`0.50`、`1.00` | 可写小数或与脚本内别名，如 `0.50`、`frac050`、`1`；**不支持**未列在脚本 `fraction_to_label` 中的比例          |


**评估时固定传入 `lerobot_eval` 的选项**（写在 shell 内）：`--policy.type=smolvla`、`--env.type=libero`、`--env.task=libero_spatial`、`--eval.n_episodes=500`、`--eval.batch_size=2`；需 headless 渲染时脚本使用 `MUJOCO_GL=egl`。

**精简镜像 / 容器（Debian/Ubuntu）**：若出现 `libGL.so.1`、`libEGL.so.0`、`libOpenGL.so.0` 缺失或 PyOpenGL EGL 相关报错，在容器内安装 Mesa/EGL 与 GLVND 用户态库后再跑评估：

```bash
apt-get update
apt install cmake
export PATH="/usr/bin:$PATH"
pip install libero
apt-get install -y libegl1 libopengl0 libglvnd0 libgl1 libglib2.0-0
```

**零样本基线复制**：若存在路径 `outputs/eval/2026-04-06/22-05-32_libero_smolvla/eval_info.json`，会复制到 `results/zero_shot/eval_info.json`；不存在则跳过并打印警告。

**示例**

```bash
# 默认评估全部五个比例
bash scripts/03_run_eval.sh

# 只评估 50% 与 100%
bash scripts/03_run_eval.sh 0.50 1.00

# 指定微调输出与结果目录
bash scripts/03_run_eval.sh --finetune-output-base-dir /tmp/ft --results-dir /tmp/eval 0.25
```

单次评估约 **20–40 分钟**（500 episodes）。输出：`results/fracXXX/eval_info.json`（以及各 run 的 `output_dir` 下日志）。

### Step 4: Analyze and visualize

```bash
pipenv run python scripts/04_analyze_results.py
```

Outputs:

- `results/scaling_curve.png` — success rate vs. demo count (main figure)
- `results/per_task_heatmap.png` — per-task breakdown across conditions
- `results/summary_table.csv` — numerical results table

---

## Expected Results

Based on prior VLA fine-tuning literature, we expect:

```
Condition     | Avg Success Rate
--------------|----------------
zero-shot     | 73%  (baseline)
frac05 (~2/task) | 60–70%  (may hurt: noisy gradient, overfit)
frac10 (~4/task) | 72–78%
frac25 (~11/task) | 78–84%
frac50 (~21/task) | 82–87%
frac100 (~43/task) | 84–90%
```

Interesting hypotheses to test:

- Does very small data (frac05) actually *hurt* vs zero-shot?
- Does task 5 (hardest, 36% zero-shot) benefit disproportionately?
- Is there a "good enough" knee in the curve around 25%?

---

## Analysis Extensions (Bonus)

Once the main results are collected, consider adding:

1. **Language ablation**: Re-run frac100 with empty task strings `""` instead of
  language instructions. Measures how much language grounding helps.
2. **Learning curve during training**: Add `--eval_freq=5000` in frac100 run to get
  5 checkpoints → plot success rate vs. training steps (not just data size).
3. **Task difficulty correlates**: Do harder tasks (lower zero-shot score) benefit
  more from fine-tuning? Scatter plot: (zero-shot SR) vs. (improvement from fine-tuning).

---

## Resume Description

> **VLA Data Efficiency Study** | LeRobot / LIBERO | Python, PyTorch
>
> Investigated how demonstration count affects fine-tuning performance of SmolVLA (500M
> parameter VLA) on 10-task LIBERO-Spatial benchmark. Designed stratified subsampling
> protocol across 5 data regimes (5%–100% of 432 demos). Measured zero-shot baseline
> (73% avg success) and quantified improvements from fine-tuning, producing scaling
> curves that reveal the minimum demonstration count for each task. All experiments run
> in MuJoCo simulation without physical hardware.

