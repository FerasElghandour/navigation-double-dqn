# Reproduction and benchmark scope

A new CartPole-v1 path trains the released agent, selects checkpoints using separate validation reset seeds, and evaluates against random and zero-action baselines on disjoint final reset seeds. Three 100-episode runs and reloadable checkpoints are included.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `control-benchmark` | Configurable Gymnasium benchmark with default 200-episode budget; distinct from the shorter published reference | None |
| `reference-control` | Published three-seed bounded Gymnasium budget; held-out rollout seeds and random/zero-action baselines; no Unity solve claim | None |

Supply named paths with `--input name=/absolute/path`. Exact commands are versioned in [reproduction.json](reproduction.json); direct training commands and preparation contracts are below.

## Evidence boundary

The bounded runs underperform the random baseline. They are not a solved CartPole claim or a reproduction of the legacy Banana environment.

Keep input acquisition and source credits in [DATA.md](DATA.md). Historical results, new reference runs and small contract fixtures have separate evidential scope.

## Direct commands and inputs

```sh
python -m pip install -r requirements.txt
python train.py --episodes 100 --seeds 0,1,2 --output runs/cartpole
python train.py --evaluate-checkpoint checkpoints/checkpoint-seed-0.pt --output runs/reloaded
```

CartPole-v1 training, validation and final rollout reset seeds are disjoint. `results/gymnasium/benchmark.json` retains budgets, all returns and both action baselines. These bounded runs do not solve the task.


## Recorded command definitions

### checks

Offline behavioral regressions; not a fresh dataset benchmark

```text
{python} smoke_test.py
{python} test_algorithm.py
```

### control-benchmark

Configurable Gymnasium benchmark with default 200-episode budget; distinct from the shorter published reference

```text
{python} train.py --output {output}/control
```

### reference-control

Published three-seed bounded Gymnasium budget; held-out rollout seeds and random/zero-action baselines; no Unity solve claim

```text
{python} train.py --episodes 100 --output {output}/reference-control
```

The observed local runtime is recorded in [reference-environment-macos.json](reference-environment-macos.json). The reproduction driver hashes source files and inputs for each run. Published result records retain their own original runtime and revision; changing the device or budget creates a different experiment.

## Longer-budget diagnostic and task-aware control baseline

An additional fixed **200-episode × three-seed** run retains [all learning traces and returns](results/gymnasium-200/benchmark.json), selected checkpoints, [controller rollouts](results/gymnasium-200/controller.json) and [source revision/commands](results/gymnasium-200/protocol.json). Selection uses validation reset seeds only. Previously inspected final rollout seeds are reused: this is an exploratory budget extension, not untouched independent validation.

| Policy | Mean return | Across training-seed groups |
|---|---:|---:|
| Released RL agent, 200 episodes | 14.93 | sample SD 6.44 |
| Random actions | 21.30 | same reset seeds |
| Fixed task-aware controller | 500.00 | same reset seeds |

Extra training does not resolve Double DQN's failure to beat random. The fixed linear switching controller reaches the 500-step cap on all 30 inspected rollouts; that is a controller result, not an RL solve. Priorities are learning-curve diagnosis, validation-only protocol changes and a later evaluation on new reset seeds.

The fixed controller is a **task-aware engineering reference**, with a knowledge advantage over model-free learning. CartPole uses a declared linear switching rule; Pendulum uses known simulator physics for energy shaping and a fixed local PD region. All constants were fixed before the controller's first evaluation. They were not tuned against these returns. Input/action conventions follow [Gymnasium CartPole](https://gymnasium.farama.org/environments/classic_control/cart_pole/) and [Pendulum](https://gymnasium.farama.org/environments/classic_control/pendulum/). The complete implementation is in [benchmark_controller.py](benchmark_controller.py).

```sh
python train.py --episodes 200 --seeds 0,1,2 --output runs/extended
python benchmark_controller.py --benchmark runs/extended/benchmark.json --output runs/controller
python train.py --evaluate-checkpoint results/gymnasium-200/checkpoint-seed-0.pt --output runs/extended-reload
```

## Pinned local reference installation

[requirements-reference-macos.txt](requirements-reference-macos.txt) pins the installed runtime/test dependency closure for Python 3.14.3 on macOS ARM64. It accompanies the observed environment record. It is a platform-specific reference snapshot, not a Linux/CUDA lock; a new empty-environment installation has not been independently certified. The GitHub checks use their separately declared Linux environment.

```sh
python3.14 -m venv .venv-reference
.venv-reference/bin/python -m pip install -r requirements-reference-macos.txt
```

Use the recorded exact Python patch version when comparing numerical results. Notebook tooling and optional FID dependencies are outside this runtime/test snapshot.
