"""Chained-job checks: a run split over several Slurm jobs must equal the run done in one go.

  1. run_training stopped at given iterations and resumed gives the same weights and metric as
     one uninterrupted run, for the single, band and double nets (window LSTM).
     Runs on the CPU, where training is exactly repeatable.
  2. the whole experiment, on the GPU when there is one, under a tiny --time-budget-min reaches the end over several "jobs" with
     every Optuna trial complete and none failed, distinct learning rates, nothing left to resume.

Usage (from deep_hedging_v0/):   python -m fast_hedging.tests.test_resume
"""
import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

from fast_hedging.experiment import EXIT_RESUME, SPLIT_SEEDS, main, run_training
from fast_hedging.market import MarketSpec, default_device, get_bank
from fast_hedging.ppo import OutOfTime, RunCheckpoint, Task, TrainConfig

SIZES = {"train": 20, "val": 10, "test": 10}


class StopAt(RunCheckpoint):
    """Runs out of time once the run has done `stop` iterations in total."""

    def __init__(self, path, stop, logs):
        super().__init__(path)
        self.stop, self.logs = stop, logs

    def expired(self):
        return self.logs.get("last_global_iter", 0) >= self.stop


def same_weights(a, b):
    pairs = list(a["actor"].items())
    if a["inactor"] is not None:
        pairs += [(f"inactor.{k}", v) for k, v in a["inactor"].items()]
    other = dict(b["actor"]); other.update({f"inactor.{k}": v for k, v in (b["inactor"] or {}).items()})
    return all(torch.equal(v.cpu(), other[k].cpu()) for k, v in pairs)


def check_exact(net, stops, tmp, **kw):
    cfg = TrainConfig(net=net, paths_per_iter=2, minibatch=4096, max_iters=8, min_iters=1, eval_every=2, patience=100,
                      actor_update_every=1, lr_patience=1, lr_decays=2, **kw)  # plateau cuts happen inside the split runs
    tag = net + "".join(f" {k}={v}" for k, v in kw.items())
    spec = MarketSpec(dynamics="bsm", product="call", market="sp500")
    banks = {s: get_bank(spec, s, n, SPLIT_SEEDS[s], device="cpu") for s, n in SIZES.items()}
    train = Task(banks["train"], "call", spec.r, cfg, "cpu")
    val = Task(banks["val"], "call", spec.r, cfg, "cpu", norm=train.norm)
    tasks = (train, val, None)
    a = SimpleNamespace(product="call")
    params = {"actor_lr": 1e-3, "joint_lr": 1e-3, "inactor_lr": 1e-3}
    quiet = lambda msg: None
    whole, best_w, _, logs_w = run_training(a, cfg, tasks, params, 7, quiet)
    path = Path(tmp) / f"{net}_{abs(hash(tag))}.pt"
    for stop in stops:
        logs = {"eval": [], "train": []}
        try:
            run_training(a, cfg, tasks, params, 7, quiet, logs=logs, ckpt=StopAt(path, stop, logs))
            raise AssertionError(f"{net}: expected to stop at iteration {stop}")
        except OutOfTime:
            pass
    logs = {"eval": [], "train": []}
    split, best_s, _, logs_s = run_training(a, cfg, tasks, params, 7, quiet, logs=logs, ckpt=RunCheckpoint(path))
    assert same_weights(whole.state_dict(), split.state_dict()), f"{net}: weights differ after resume"
    assert best_w == best_s, f"{net}: best metric differs after resume: {best_w} vs {best_s}"
    assert [e["val_cvar_95"] for e in logs_w["eval"]] == [e["val_cvar_95"] for e in logs_s["eval"]], f"{net}: eval log differs"
    assert logs_w["last_global_iter"] == logs_s["last_global_iter"], f"{net}: iteration count differs"
    assert [e.get("lr_actor") for e in logs_w["train"]] == [e.get("lr_actor") for e in logs_s["train"]], f"{net}: rate cuts differ"
    cuts = sum(1 for e in logs_w["eval"] if e.get("lr_cut"))
    print(f"exact resume OK: {tag}, stopped at iterations {stops}, best {best_s:.4f}, rate cuts {cuts}", flush=True)


def check_chain(net, tmp, extra=()):
    out = Path(tmp) / f"chain_{net}"
    argv = ["--dynamics", "bsm", "--product", "call", "--net", net, "--name", f"chain_{net}", "--out", str(out),
            "--train-paths", str(SIZES["train"]), "--val-paths", str(SIZES["val"]), "--test-paths", str(SIZES["test"]),
            "--paths-per-iter", "2", "--minibatch", "4096", "--tuning-seeds", "0", "1", "--final-seeds", "1",
            "--max-iters", "6", "--min-iters", "1", "--eval-every", "2", "--patience", "100",
            "--trace-paths", "2", "--device", default_device(), "--time-budget-min", "0.1", *extra]
    jobs = 0
    while True:
        jobs += 1
        assert jobs <= 60, f"{net}: no finish after {jobs} jobs"
        try:
            main(argv)
            break
        except SystemExit as e:
            assert e.code == EXIT_RESUME, f"{net}: unexpected exit code {e.code}"
    s = json.loads((out / "summary.json").read_text())
    assert s["trials"]["FAIL"] == 0 and s["trials"]["RUNNING"] == 0 and s["trials"]["WAITING"] == 0, s["trials"]
    lrs = [t["params"]["actor_lr"] for t in s["trial_table"]]
    assert len(set(lrs)) == len(lrs) or "--actor-lr-range" in extra, f"repeated learning rates: {lrs}"
    assert not list((out / "resume").glob("*.pt")), "resume files left behind"
    print(f"chain OK: {net}, {jobs} jobs, trials {s['trials']['COMPLETE']}, learning rates {[f'{x:.2e}' for x in lrs]}", flush=True)
    return jobs


if __name__ == "__main__":
    torch.set_num_threads(2)
    with tempfile.TemporaryDirectory() as tmp:
        check_exact("single", [3], tmp)
        check_exact("band", [2, 5], tmp)
        check_exact("double", [3, 11], tmp)     # one stop in each phase
        total = check_chain("band", tmp, ("--n-trials", "3", "--prune-warmup", "1"))
        total += check_chain("double", tmp, ("--n-trials", "1", "--no-pruning"))
        total += check_chain("single", tmp, ("--n-trials", "1", "--no-pruning", "--actor-lr-range", "5.85e-4", "5.85e-4",
                                             "--value-coef", "1.0", "--init-std", "0.2"))
        assert total > 3, "the time budget never stopped a job, so the chain was not exercised"
    print("ALL PASSED")
    sys.exit(0)
