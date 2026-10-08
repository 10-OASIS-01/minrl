# CartPole benchmark · 2026-10-08

All 15 runs completed: DQN, DQN with prioritized replay, and PPO, each with seeds 0–4
and exactly 100,000 training interactions. Total training budget: **1,500,000 environment steps**.
No runs were dropped, restarted, early-stopped or tuned based on their outcomes.

![Learning curves and final evaluation](../figure/benchmark.png)

## Protocol

The configuration is [`configs/benchmark.yaml`](../configs/benchmark.yaml).
Run `minrl benchmark --config configs/benchmark.yaml --output runs/cartpole` from the root.
The checkpoints reported below are the **final**, not the best, checkpoints.

- Training environments: CartPole-v1 with its native 500-step time limit.
- Periodic evaluation: 10 greedy episodes every 5,000 steps, reset seeds 10000–10009.
- Final evaluation: 100 greedy episodes, reset seeds 20000–20099.
- Random baseline: uniform actions, same final evaluation reset seeds.
- CPU, one PyTorch thread, sequential runs, arm64 macOS with 24 GiB RAM.
- Python 3.11.16; NumPy 2.4.6; PyTorch 2.14.1; Gymnasium 1.4.0; Matplotlib 3.11.2.

Timing includes training, logging, periodic evaluation and final evaluation. These timings
are specific to this machine and concurrent local workload, not a hardware-normalized comparison.
The total measured run time was 435.7 seconds (7.3 minutes), excluding environment setup,
software development, test execution and report generation.

## Results by training seed

Each entry is the mean over that checkpoint's 100 final evaluation episodes.

| Seed | DQN | DQN + PER | PPO |
| --- | ---: | ---: | ---: |
| 0 | 500.00 | 413.74 | 499.90 |
| 1 | 461.41 | 468.27 | 498.59 |
| 2 | 500.00 | 500.00 | 500.00 |
| 3 | 500.00 | 500.00 | 500.00 |
| 4 | 500.00 | 500.00 | 500.00 |
| Mean | **492.28** | **476.40** | **499.70** |
| Population SD across seeds | 15.44 | 33.65 | 0.56 |
| Mean run time | 23.3 s | 34.7 s | 29.1 s |

The random-policy mean was **23.74** over 100 episodes. Its single evaluation batch is a
reference line, not a five-seed training result. The final distribution shows individual
seed means; the learning curve uses periodic evaluation means. These differ because the
final evaluation uses a larger, independent set of reset seeds.

## What students should notice

PPO learned quickly in these runs, but even PPO had intermediate regressions. DQN also
showed large swings before its final scores became strong. Learning curves do not have to
increase monotonically, and choosing the best point after seeing a curve changes the evaluation
question. This report consistently uses a fixed training budget and the final checkpoint.

PER had a lower final mean and greater variability than uniform replay in this experiment.
That does not refute prioritized replay or show that it is generally worse: the comparison
covers one simple environment, one setting per method, and only five training seeds. The
plain NumPy PER implementation also does more work per sample than uniform replay.

The plotted ±1 SD bands describe variation across seed means. They are not uncertainty
intervals for a population effect; they can extend beyond the environment's reward range.
No statistical superiority claim is made.

## Settings

**DQN and DQN + PER:** two 64-unit ReLU layers, Adam 0.001, gamma 0.99, MSE TD loss,
gradient norm limit 10, replay capacity 50,000, batch 64, 1,000 warm-up steps, one update
per four environment steps, hard target synchronization every 500 steps. Epsilon decreases
linearly from 1 to 0.05 over 20,000 steps. PER additionally uses alpha 0.6, beta 0.4 → 1,
priority epsilon 1e-6, and normalized importance weights. Learning-rate decay is off.

**PPO:** independent 64×64 tanh actor and critic, Adam 0.0003 (epsilon 1e-5), rollout 2,048,
minibatch 64, ten epochs, gamma 0.99, GAE lambda 0.95, policy and value clipping 0.2,
critic coefficient 0.5, entropy coefficient 0.01, maximum gradient norm 0.5. Advantage
normalization uses population standard deviation. The final partial rollout is updated.
Learning-rate decay is off.

## Reproduction and provenance

Install the locked dependencies with `uv sync --python 3.11 --all-extras --frozen`.
Results were generated from the local 0.2 implementation while the repository was uncommitted,
based on commit `f221eb51c6aaea649d77662a0c5719744b22fb47`; that base commit alone is not
the implementation used for this benchmark. The delivered source archive and dependency
lock preserve the final implementation. Subsequent edits corrected documentation, plotting,
error reporting and optional early-stop handling, without changing these fixed-budget updates.

The delivery contains all requested configurations, resolved agent configurations, CSV metrics,
20 periodic evaluations per run, 100 final returns per run, checkpoints and TensorBoard logs.
`figure/benchmark_data.json` includes compact per-run results and periodic evaluations for
inspection in the repository; the larger training files remain outside Git.

Exact floating-point reproducibility across versions and platforms is not guaranteed. Rerun
with the same dependency lock and CPU thread count before attributing a discrepancy to an algorithm.
