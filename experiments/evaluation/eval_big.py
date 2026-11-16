"""Evaluate the selected seed-1 single and double nets of one cell on a large fresh test bank (default 50,000 paths,
split seed 404, i.e. new paths, not the 2,000-path test set). Per-env pnl / abs_err / cost / hold_rate / n_trades to
experiments/results/main_big/<dyn>_<prod>_fee<fee>_<net>.npz and metrics to the matching .json.
usage (from the repository root): python -m experiments.evaluation.eval_big <dyn> <prod> <fee 01|1> <single run dir> <double run dir> [n_paths]"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from hedging_engine.market import MarketSpec, get_bank
from hedging_engine.ppo import Agent, Task, TrainConfig, evaluate

dyn, prod, fee, single_dir, double_dir = sys.argv[1:6]
n_paths = int(sys.argv[6]) if len(sys.argv) > 6 else 50000
BIG_SEED = 404
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path("experiments/results/main_big"); out.mkdir(parents=True, exist_ok=True)
log = lambda s: print(s, flush=True)

t0 = time.time()
spec = MarketSpec(dynamics=dyn, product=prod)
bank = get_bank(spec, "test", n_paths, BIG_SEED, device=device)
log(f"bank ready in {time.time() - t0:.0f}s: {bank.S.shape}")
for net, run_dir in (("single", single_dir), ("double", double_dir)):
    dst = out / f"{dyn}_{prod}_fee{fee}_{net}"
    if dst.with_suffix(".json").exists():
        log(f"skip {dst.name} (done)"); continue
    ck = torch.load(Path(run_dir) / "best_trial.pt", map_location=device, weights_only=True)
    cfg = TrainConfig(**ck["config"])
    agent = Agent(cfg, prod, device); agent.load_state_dict(ck["state"])
    task = Task(bank, prod, spec.r, cfg, device)
    t1 = time.time()
    m, arr = evaluate(agent, task, use_inactor=(net == "double"))
    log(f"[{dyn} {prod} fee {fee} {net}] {Path(run_dir).name}: track {m['mean_abs_tracking_error']:.3f} cost {m['mean_cost']:.2f} "
        f"cvar95 {m['cvar_95']:.1f} hold {m['hold_rate']:.3f} on {task.P} paths ({time.time() - t1:.0f}s)")
    np.savez_compressed(dst.with_suffix(".npz"), **{k: v.astype(np.float32) for k, v in arr.items()})
    dst.with_suffix(".json").write_text(json.dumps({"run": run_dir, "n_paths": int(task.P), "seed": BIG_SEED, "metrics": m}, indent=1))
log("EVAL BIG DONE")
