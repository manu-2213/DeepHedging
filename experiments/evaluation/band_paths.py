"""Per-path arrays of the model target / model band references (same test paths as the RL runs), for the paper figures.
For every cell JSON in experiments/results/main_baselines: pnl, cost, abs_err per env, plus step-by-step position,
portfolio value, trade flags for the first TRACE paths (all assets/strikes), band and daily target.
Writes experiments/results/main_baselines/<dyn>_<prod>_fee<fee>_paths.npz.
usage (from the repository root): python -m experiments.evaluation.band_paths"""
import json, sys
from pathlib import Path
import numpy as np
import torch
from hedging_engine.market import MarketSpec, get_bank
from hedging_engine.model_band import compute_targets
from hedging_engine.ppo import Task, TrainConfig

from experiments.protocol import BASELINE_PATHS as SIZES, FEES, SPLIT_SEEDS

TRACE = 500
OUT = Path("experiments/results/main_baselines")
device = "cuda" if torch.cuda.is_available() else "cpu"
log = lambda s: print(s, flush=True)


@torch.no_grad()
def run_band(task, target, dtarget, k, trace_paths):
    tgt_all = torch.as_tensor(target, device=task.device); dt_all = torch.as_tensor(dtarget, device=task.device)
    disc = torch.exp(-task.r * task.T[: task.N]).to(task.device)
    B = task.P * task.A * task.M
    X = task.X.reshape(B, task.N1, task.I); L = task.L.reshape(B, task.N1)
    tg = tgt_all.reshape(B, task.N1, task.I)[:, : task.N]; dg = dt_all.reshape(B, task.N1, task.I)[:, : task.N]
    band = None
    if k is not None:
        band = float(k) * (disc[None, :, None] * task.fees[None, None, :] * X[:, : task.N].abs() * dg**2).clamp_min(0).pow(1.0 / 3.0)
    nt = trace_paths * task.A * task.M
    pos = torch.zeros(B, task.I, device=task.device); cash = L[:, 0].clone()
    cost_tot = torch.zeros(B, device=task.device); n_tr = torch.zeros(B, device=task.device); abs_err = torch.zeros(B, device=task.device)
    tr_pos = torch.zeros(nt, task.N, task.I, device=task.device); tr_pv = torch.zeros(nt, task.N1, device=task.device)
    tr_trade = torch.zeros(nt, task.N, device=task.device); tr_err = torch.zeros(nt, task.N, device=task.device)
    tr_pv[:, 0] = cash[:nt]
    for t in range(task.N):
        final = tg[:, t]
        if band is not None:
            final = torch.maximum(torch.minimum(pos, final + band[:, t]), final - band[:, t])
        trade = final - pos; flow = trade * X[:, t]; cost = (task.fees * flow.abs()).sum(-1)
        cash = (cash - flow.sum(-1) - cost) * task.growth; pos = final
        cost_tot += cost; traded = (trade.abs().amax(-1) > 1e-6).float(); n_tr += traded
        pv = cash + (pos * X[:, t + 1]).sum(-1); err = (pv - L[:, t + 1]).abs(); abs_err += err
        tr_pos[:, t] = pos[:nt]; tr_pv[:, t + 1] = pv[:nt]; tr_trade[:, t] = traded[:nt]; tr_err[:, t] = err[:nt]
    pv = cash + (pos * X[:, task.N]).sum(-1)
    res = {"pnl": pv - L[:, task.N], "cost": cost_tot, "n_trades": n_tr, "abs_err": abs_err / task.N,
           "trace_position": tr_pos, "trace_portfolio_value": tr_pv, "trace_traded": tr_trade, "trace_tracking_error": tr_err}
    if band is not None:
        res["trace_band_width"] = band[:nt]
    return {kk: v.cpu().numpy().astype(np.float32) for kk, v in res.items()}


for f in sorted(OUT.glob("*_fee*.json")):
    r = json.load(open(f)); dyn, prod, fee = r["dynamics"], r["product"], str(r["fee"])
    out = OUT / f"{dyn}_{prod}_fee{fee}_paths.npz"
    if out.exists():
        log(f"skip {out.name}"); continue
    spec = MarketSpec(dynamics=dyn, product=prod)
    bank = get_bank(spec, "test", SIZES["test"], SPLIT_SEEDS["test"], device=device)
    target, dtarget = compute_targets(spec, "test", bank, device, log, r["bump"], r["mc_paths"], seed=SPLIT_SEEDS["test"])
    uf, of = FEES[fee]
    cfg = TrainConfig(underlying_fee=uf, option_fee=of, normalize_obs=False, reward_norm="none", agent_state=False)
    task = Task(bank, prod, spec.r, cfg, device)
    arrays = {}
    for name, k in (("model_band", r["band_scale"]), ("model_target", None)):
        res = run_band(task, target, dtarget, k, TRACE)
        arrays.update({f"{name}_{kk}": v for kk, v in res.items()})
        log(f"[{dyn} {prod} fee {fee}] {name}: track {res['abs_err'].mean():.3f} cost {res['cost'].mean():.2f} (json {r['test'][name]['mean_abs_tracking_error']:.3f})")
    nt = TRACE * task.A * task.M
    arrays["trace_target"] = np.asarray(target).reshape(-1, task.N1, task.I)[:nt].astype(np.float32)
    arrays["trace_liability"] = task.L.reshape(-1, task.N1)[:nt].cpu().numpy().astype(np.float32)
    arrays["trace_spot"] = task.S[:TRACE][:, :, None, :].expand(TRACE, task.A, task.M, task.N1).reshape(-1, task.N1).cpu().numpy().astype(np.float32)
    arrays["trace_instrument_prices"] = task.X.reshape(-1, task.N1, task.I)[:nt].cpu().numpy().astype(np.float32)
    np.savez_compressed(out, **arrays)
    log(f"wrote {out}")
log("BAND PATHS DONE")
