# Reproduction and benchmark scope

A new CartPole-v1 path trains the released agent, selects checkpoints using separate validation reset seeds, and evaluates against random and zero-action baselines on disjoint final reset seeds. Three100-episode runs and reloadable checkpoints are included.

## Execute a recorded profile

```sh
python run_reproduction.py --profile checks --output reproduced/checks
```

Use a new output directory for each run. The driver writes command logs, source/input SHA-256 values, versions and an explicit outcome. It does not silently download data or turn generated fixtures into dataset-performance claims.

| Profile | Actual scope | Required inputs |
|---|---|---|
| `checks` | Offline behavioral regressions; not a fresh dataset benchmark | None |
| `control-benchmark` | New named Gymnasium task; three training seeds and held-out rollout seeds; no Unity solve claim | None |

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

```text
{python} smoke_test.py
```

### control-benchmark

```text
{python} train.py --output {output}/control
```

`{python}` is the current interpreter. `{output}` is the new run directory. `{input:name}` refers to a named absolute input path supplied to the driver. Install declared dependencies in a clean environment before executing. Never execute untrusted serialized models or notebooks.
