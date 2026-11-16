"""Out-of-sample years: evaluate saved nets on the test bank of market `sp500x` (years 2017-2023 fitted; the test split holds
2021, 2022, 2023 as assets 0, 1, 2) and report metrics per year. Same path count and split seed as the main test set.
usage (from the repository root): python -m experiments.evaluation.eval_years <dyn> <prod> <fee 01|1> <run_dir> [<run_dir> ...]
Writes experiments/results/main_years/<dyn>_<prod>_fee<fee>/<run_name>.json ({year: metrics}) and .npz (per-env arrays)."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from hedging_engine.market import MarketSpec, get_bank
from hedging_engine.ppo import Agent, Task, TrainConfig, evaluate, env_index, risk_metrics, evaluate_baseline

dyn, prod, fee = sys.argv[1:4]; run_dirs = sys.argv[4:]
MARKET, N_PATHS, SEED = "sp500x", 2000, 303
YEARS = ["2021", "2022", "2023"]
device = "cuda" if torch.cuda.is_available() else "cpu"
out = Path(f"experiments/results/main_years/{dyn}_{prod}_fee{fee}"); out.mkdir(parents=True, exist_ok=True)
log = lambda s: print(s, flush=True)

spec = MarketSpec(dynamics=dyn, product=prod, market=MARKET)
t0 = time.time(); bank = get_bank(spec, "test", N_PATHS, SEED, device=device)
log(f"bank {bank.S.shape} ready in {time.time() - t0:.0f}s")
assert bank.S.shape[1] == len(YEARS), f"expected {len(YEARS)} test years, bank has {bank.S.shape[1]} assets"

def per_year(arr, idx):
    res = {}
    for a, y in enumerate(YEARS):
        m = idx["env_asset"] == a
        r = risk_metrics(arr["pnl"][m])
        r.update({"mean_abs_tracking_error": float(arr["abs_err"][m].mean()), "mean_cost": float(arr["cost"][m].mean()),
                  "mean_trades": float(arr["n_trades"][m].mean()), "hold_rate": float(arr["hold_rate"][m].mean()) if "hold_rate" in arr else None})
        res[y] = r
    return res

# references: no hedge (every cell) and, for the call, the model delta
uf, of = {"01": (0.001, 0.002), "1": (0.01, 0.05)}[fee]
cfg0 = TrainConfig(underlying_fee=uf, option_fee=of, normalize_obs=False, reward_norm="none", agent_state=False)
task0 = Task(bank, prod, spec.r, cfg0, device); idx = env_index(task0)
refs = {}
for kind in (["no_hedge", "model_delta"] if prod == "call" else ["no_hedge"]):
    _, arr = evaluate_baseline(task0, kind)
    refs[kind] = per_year(arr, idx)
    log(f"[{dyn} {prod} fee {fee}] {kind}: " + "  ".join(f"{y} {refs[kind][y]['mean_abs_tracking_error']:.2f}" for y in YEARS))
(out / "references.json").write_text(json.dumps(refs, indent=1))

for rd in run_dirs:
    rd = Path(rd); name = rd.name
    if (out / f"{name}.json").exists():
        log(f"skip {name}"); continue
    ck = torch.load(rd / "best_trial.pt", map_location=device, weights_only=True)
    cfg = TrainConfig(**ck["config"])
    agent = Agent(cfg, prod, device); agent.load_state_dict(ck["state"])
    task = Task(bank, prod, spec.r, cfg, device)
    m, arr = evaluate(agent, task, use_inactor=(cfg.net == "double"))
    res = per_year(arr, idx)
    log(f"[{dyn} {prod} fee {fee}] {name}: " + "  ".join(f"{y} {res[y]['mean_abs_tracking_error']:.2f}/{res[y]['mean_cost']:.1f}" for y in YEARS))
    np.savez_compressed(out / f"{name}.npz", **{k: v.astype(np.float32) for k, v in arr.items()}, **idx)
    (out / f"{name}.json").write_text(json.dumps({"run": str(rd), "net": cfg.net, "years": res}, indent=1))
log("EVAL YEARS DONE")
