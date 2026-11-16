"""Collects every run into one JSON object on stdout: {"curves": [...], "runs": [...]}.

Runs where the results are (the HPC login node; standard library only). For each run under experiments/results/{main,
v2_3, v2_4} it reads log.txt (validation curve), summary.json (test results) and the launch command (settings), and adds
the model-based references from experiments/results/main_baselines. The output feeds analysis/main_table.py, the paper
figures (figures/make_figs.py) and the learning-curve dashboard.

Usage: ssh <hpc> 'cd ~/DeepHedging && python3 -' < experiments/analysis/collect_runs.py > report.json
       then save report["runs"] and report["curves"] as experiments/results/collected/runs.json and curves.json.

Works with both layouts of the repository: the current one (experiments/ at the root) and the one before the 2026-11
restructure (deep_hedging_v0/experiments/, launch scripts under simulations/slurm/<batch>/).
"""
import glob
import json
import os
import re
import subprocess

REPO = os.environ.get("DEEP_HEDGING_REPO", os.path.expanduser("~/DeepHedging"))
LEGACY = not os.path.isdir(f"{REPO}/experiments/results") and os.path.isdir(f"{REPO}/deep_hedging_v0/experiments/results")
ROOT = f"{REPO}/deep_hedging_v0" if LEGACY else REPO
RES = f"{ROOT}/experiments/results"
SLURM = f"{ROOT}/experiments/simulations/slurm" if LEGACY else f"{ROOT}/experiments/slurm"
OLD_SLURM = f"{REPO}/deep_hedging_v0/experiments/simulations/slurm"     # development batches (v2_3, v2_4), if still on disk
RUNNER = re.compile(r"fast_hedging\.experiment|experiments\.train")      # the training entry point, old and new name

# Run groups: main (the main table: single policy and Actor-Inactor), followup (band net, parity pair, proposal-aware
# gate, fee transfer; never part of the main table), and the BSM-call development batches below.
# tag -> (label, group, order). group: double (double and single nets), band (band runs still in play),
# band_old (early band probes and stopped methods)
TAGS = {
    "grpo": ("LSTM + GRPO", "band", 10), "grpostate": ("LSTM + GRPO, agent state", "band", 11),
    "grpog995": ("LSTM + GRPO, gamma 0.995", "band", 12), "grpog1": ("LSTM + GRPO, gamma 1", "band", 13),
    "tfp": ("Fast transformer", "band", 20), "tfpstate": ("Fast transformer, agent state", "band", 21),
    "tfpg995": ("Fast transformer, gamma 0.995", "band", 22), "tfpg1": ("Fast transformer, gamma 1", "band", 23),
    "tfplr3e5": ("Fast transformer, rate 3e-5", "band", 24), "tfplr3e4": ("Fast transformer, rate 3e-4", "band", 25),
    "tfpgrpo": ("Fast transformer + GRPO", "band", 26),
    "tf": ("Window transformer", "band_old", 40), "tfgrpo": ("Window transformer + GRPO", "band_old", 41),
    "tfpcvar": ("Fast transformer, CVaR reward", "band_old", 42), "grpocvar": ("LSTM + GRPO, CVaR reward", "band_old", 43),
    "batch": ("4x batch (probe)", "band_old", 44), "data": ("10x data (probe)", "band_old", 45),
    "wide": ("Wide net 128 (probe)", "band_old", 46),
    "tfpsingle": ("Single net: fast transformer", "double", 60),
    "tfpgrposingle": ("Single net: fast transformer + GRPO", "double", 61),
    "lstmsingle": ("Single net: LSTM + PPO", "double", 58), "grposingle": ("Single net: LSTM + GRPO", "double", 59),
    "tgslr3e5": ("Single net: fast tf + GRPO, rate 3e-5", "double", 62), "tgslr3e4": ("Single net: fast tf + GRPO, rate 3e-4", "double", 63),
    "tgsg8": ("Single net: fast tf + GRPO, group 8", "double", 64), "tgsg16": ("Single net: fast tf + GRPO, group 16", "double", 64), "tgsg32": ("Single net: fast tf + GRPO, group 32", "double", 64), "tgsstd02": ("Single net: fast tf + GRPO, noise 0.2", "double", 65),
    "wtgs": ("Single net: window transformer + GRPO", "double", 66),
    "p5ctl": ("Double, control (2,000 validation paths)", "double", 80), "p5init": ("Double, inactor starts at 5% holding", "double", 81),
    "p5sstate": ("Single net with agent state", "double", 82), "p5shold": ("Single net with agent state + forced holds", "double", 83),
    "p5dstate": ("Double from agent-state actor", "double", 84), "p5dhold": ("Double from forced-holds actor", "double", 85),
    "p5dholdinit": ("Double from forced-holds actor, inactor at 5% holding", "double", 86),
}


def squeue():
    try:
        out = subprocess.run(["squeue", "-u", os.environ.get("USER", "acw796"), "-h", "-o", "%j %T %r"],
                             capture_output=True, text=True, timeout=60).stdout
    except Exception:
        return {}
    st = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) >= 2:
            name = re.sub(r"^(v24_|mn_)", "", p[0])
            st[name] = "held" if (len(p) > 2 and "Held" in p[2]) else p[1].lower()
    return st


def command_for(name, ver="v2.4"):
    sub = {'v2.3': 'v2_3', 'v2.4': 'v2_4', 'main': 'main'}[ver]
    # a run's own script, if it has one: the current layout keeps the generated scripts in experiments/slurm
    for own in (f"{SLURM}/run_{name}.slurm", f"{SLURM}/{sub}/run_{name}.slurm", f"{OLD_SLURM}/{sub}/run_{name}.slurm"):
        if os.path.exists(own):
            for line in open(own):
                if RUNNER.search(line):
                    return line
    pat = re.compile(r"(--name\s+" + re.escape(name) + r"(\s|$))|(/" + re.escape(name) + r"(\s|$|/))")
    for f in [g for d in dict.fromkeys((SLURM, OLD_SLURM)) for g in glob.glob(f"{d}/v2_4/*.slurm") + glob.glob(f"{d}/v2_4/*.txt") + glob.glob(f"{d}/v2_3/*.slurm")]:
        try:
            for line in open(f):
                if (RUNNER.search(line) and pat.search(line)) or line.startswith(name + " --"):
                    return line
        except OSError:
            pass
    return ""


def flag(cmd, name, n=1, default=None):
    m = re.findall(r"--" + name + r"((?:\s+[^\s-][^\s]*|\s+-?[0-9][^\s]*){%d})" % n, cmd)
    if not m:
        return default
    v = m[-1].split()                       # the last occurrence wins, as with argparse
    return v[0] if n == 1 else v


def describe(run_dir):
    base = os.path.basename(run_dir)
    ver = "v2.3" if "/v2_3/" in run_dir else ("main" if "/main/" in run_dir else "v2.4")
    dyn, prod = "bsm", "call"
    if ver == "main":
        m = re.match(r"(bsm|heston)_(call|doc|con)_fee(01|1)_(single_lr(\w+)|double_i(\w+?)_j(\w+?)(?:_from(\w+))?|band_lr(\w+)|pair_i(\w+?)_j(\w+?)_lr(\w+?)|gate_i(\w+?)_j(\w+?)|xfer_i(\w+?))_s(?P<seed>\d)$", base)
        if not m:
            return None
        dyn, prod, fee, seed = m.group(1), m.group(2), m.group(3), int(m.group("seed"))
        lr = lambda s: s.replace("e", "e-")
        cellname = f"{dyn.upper() if dyn == 'bsm' else 'Heston'} {prod}"
        if m.group(15):
            label, group, order = f"{cellname}: frozen actor from the other fee, proposal-aware inactor {lr(m.group(15))}", "followup", 105
        elif m.group(13):
            label, group, order = f"{cellname}: Actor\u2013Inactor, proposal-aware inactor {lr(m.group(13))}, joint {lr(m.group(14))}", "followup", 104
        elif m.group(10):
            label, group, order = f"{cellname}: Actor\u2013Inactor 45+45 (from scratch), inactor {lr(m.group(10))}, joint {lr(m.group(11))}, actor {lr(m.group(12))}", "followup", 103
        elif m.group(9):
            label, group, order = f"{cellname}: band net, rate {lr(m.group(9))}", "followup", 102
        elif m.group(5):
            label, group, order = f"{cellname}: single net, rate {lr(m.group(5))}", "main", 100
        else:
            label, group, order = f"{cellname}: double net, inactor {lr(m.group(6))}, joint {lr(m.group(7))}" + (f" (actor from rate {lr(m.group(8)[2:])})" if m.group(8) else ""), "main", 101
    elif ver == "v2.3":
        m = re.match(r"bsm_call_(single|band|double)_fee(01|1)_s(\d)$", base)
        if not m:
            return None
        net = m.group(1)
        label, group, order = f"v2.3 LSTM + PPO, {net} net", ("band" if net == "band" else "double"), {"band": 0, "single": 1, "double": 2}[net]
        fee, seed = m.group(2), int(m.group(3))
    else:
        m = re.match(r"(\w+?)_fee(01|1)_s(\d)$", base)
        if not m:
            return None
        tag, fee, seed = m.group(1), m.group(2), int(m.group(3))
        d = re.match(r"dbl(lstm|grpo|tfpgrpo|tfp|tg8)_i(\w+?)_j(\w+)$", tag)
        if d:
            src = {"tfp": "fast transformer", "tfpgrpo": "fast transformer + GRPO", "tg8": "fast transformer + GRPO, group 8", "lstm": "LSTM + PPO", "grpo": "LSTM + GRPO"}[d.group(1)]
            lr = lambda s: s.replace("e", "e-")
            label, group, order = f"Double ({src}): inactor {lr(d.group(2))}, joint {lr(d.group(3))}", "double", 70
        elif tag in TAGS:
            label, group, order = TAGS[tag]
        else:
            label, group, order = tag, "band_old", 99
    cmd = command_for(base, ver)
    arch = flag(cmd, "arch", default="rnn")
    enc = {"tfpath": "fast transformer", "transformer": "window transformer"}.get(arch, "LSTM (window)")
    alr = flag(cmd, "actor-lr-range", 2) or [None]
    ilr = flag(cmd, "inactor-lr-range", 2) or [None]
    jlr = flag(cmd, "joint-lr-range", 2) or [None]
    net = flag(cmd, "net", default="?")
    return {"run": f"{ver}/{base}", "label": label, "group": group, "order": order, "net": net, "encoder": enc, "dynamics": dyn, "product": prod,
            "advantage": "GRPO" if flag(cmd, "advantage") == "grpo" else "PPO", "fee": "0.1%" if fee == "01" else "1%",
            "seed": seed, "actor_lr": alr[0], "inactor_lr": ilr[0] if net == "double" else None,
            "joint_lr": (jlr[0] if "--tune-joint-lr" in cmd or "from-single" in cmd else alr[0]) if net == "double" else None,
            "actor_init": "from single net" if "from-single" in cmd else ("trained in run" if net == "double" else None),
            "gamma": flag(cmd, "gamma", default="0.99"), "agent_state": "--no-agent-state" not in cmd,
            "reward": flag(cmd, "reward", default="?"), "select": "tracking" if flag(cmd, "metric") == "mean_abs_tracking_error" else "CVaR95"}


VAL = re.compile(r"\[(\w+)\] it +(\d+)  val (\w+)=([0-9.]+)  best=[0-9.inf]+  hold=([0-9.]+)  cost=([0-9.]+)  bad=(\d+)/(\d+).*?"
                 r"track=([0-9.]+) trades=(\d+)(?: vcvar=([0-9.]+))?")


CUT = re.compile(r"\[(\w+)\] it +\d+ +no improvement for \d+ checks: learning rates")


def parse_log(path):
    pts, offset, last_actor, cuts = [], 0, 0, {}
    for line in open(path, errors="replace"):
        c = CUT.search(line)
        if c:
            cuts[c.group(1)] = cuts.get(c.group(1), 0) + 1
            continue
        m = VAL.search(line)
        if not m:
            continue
        phase, it = m.group(1), int(m.group(2))
        if phase == "actor":
            last_actor = it
        else:
            offset = last_actor
        name, val = m.group(3), float(m.group(4))
        cvar = val if name == "cvar_95" else (float(m.group(11)) if m.group(11) else None)
        sit = re.search(r"([0-9.]+)s/it", line)
        pts.append({"sit": float(sit.group(1)) if sit else None, "phase": phase, "iteration": it + offset, "track": float(m.group(9)), "cvar": cvar,
                    "cost": float(m.group(6)), "trades": int(m.group(10)), "hold": float(m.group(5)),
                    "bad": int(m.group(7)), "patience": int(m.group(8))})
    return pts, cuts


def main():
    q = squeue()
    curves, runs, baselines, cell_baselines = [], [], {}, {}
    dirs = sorted(glob.glob(f"{RES}/v2_3/bsm_call_*")) + sorted(d for d in glob.glob(f"{RES}/v2_4/*") if not os.path.basename(d).startswith("_")) + sorted(glob.glob(f"{RES}/main/*"))
    for d in dirs:
        info = describe(d)
        if info is None or not os.path.exists(f"{d}/log.txt"):
            continue
        pts, cuts = parse_log(f"{d}/log.txt")
        if not pts:
            continue
        name = os.path.basename(d)
        summ = None
        if os.path.exists(f"{d}/summary.json"):
            try:
                summ = json.load(open(f"{d}/summary.json"))
            except ValueError:
                summ = None
        status = "done" if summ else q.get(name, "stopped")
        last_phase = pts[-1]["phase"]
        sel = [p for p in pts if p["phase"] == last_phase]
        bt = min(sel, key=lambda p: p["track"])
        cv = [p["cvar"] for p in sel if p["cvar"] is not None]
        sits = sorted(p["sit"] for p in sel if p["sit"] is not None)
        spi = sits[len(sits) // 2] if sits else None      # median s/it in the phase reached (GPU type varies between jobs)
        row = dict(info, sec_per_it=spi, hours_to_best=(bt["iteration"] * spi / 3600 if spi else None), status=status, iters=pts[-1]["iteration"], phase=last_phase,
                   bad=f"{pts[-1]['bad']}/{pts[-1]['patience']}", best_val_track=bt["track"], best_iter=bt["iteration"], cuts=cuts.get(last_phase, 0),
                   table=(info["group"] == "main") or bool(re.match(r"(tgsg8|dbltg8_i\w+_j\w+)_fee(01|1)_s1$", name)),
                   best_val_cvar=min(cv) if cv else None, val_cost=bt["cost"], trades=bt["trades"],
                   hold=bt["hold"] if info["net"] == "double" else None)
        if summ:
            t = summ["best_trial_test"]["deterministic"]
            row.update(test_track=t.get("mean_abs_tracking_error"), test_cost=t.get("mean_cost"),
                       test_cvar=t.get("cvar_95"), test_trades=t.get("mean_trades"),
                       test_hold=t.get("hold_rate") if info["net"] == "double" else None)
            b = summ.get("baselines_test", {})
            ww = b.get("ww_band", {})
            if info["select"] == "tracking" and (info["dynamics"], info["product"]) == ("bsm", "call") and "mean_abs_tracking_error" in ww:
                baselines[info["fee"]] = (b, ww)
            elif info["select"] == "tracking" and (info["dynamics"], info["product"]) != ("bsm", "call") and b:
                cell_baselines[(info["dynamics"], info["product"], info["fee"])] = b
        runs.append(row)
        best, bestc = float("inf"), float("inf")
        for i, p in enumerate(pts):
            best = min(best, p["track"])
            if p["cvar"] is not None:
                bestc = min(bestc, p["cvar"])
            if p["iteration"] % 100 and i != len(pts) - 1:
                continue
            curves.append({"method": info["label"], "group": info["group"], "order": info["order"], "fee": info["fee"], "dynamics": info["dynamics"], "product": info["product"],
                           "seed": info["seed"], "iteration": p["iteration"], "tracking_error": p["track"],
                           "best_tracking_error": round(best, 2), "cost": round(max(0.1, p["cost"]), 1), "trades": p["trades"],
                           "hold": round(p["hold"], 3)})
    # Model-based references per cell from experiments.baselines (experiments/results/main_baselines):
    # delta / delta-gamma target with a Whalley-Wilmott-style band tuned on validation. Falls back to the run summaries'
    # daily-delta baselines for cells whose JSON is missing.

    have = set()
    for f in sorted(glob.glob(f"{RES}/main_baselines/*_fee*.json")):
        r = json.load(open(f))
        dyn, prod, fee = r["dynamics"], r["product"], {"01": "0.1%", "1": "1%"}[str(r["fee"])]
        have.add((dyn, prod, fee))
        cellname = f"{dyn.upper() if dyn == 'bsm' else 'Heston'} {prod}"
        val = r.get("validation", {})
        for key, label, vt in (("model_band", f"Whalley\u2013Wilmott-style model band (k={r.get('band_scale', float('nan')):.3g}, tuned on validation)", val.get("model_band")),
                               ("model_target", "model target (delta / delta-gamma), daily", val.get("model_target")),
                               ("mv_delta", "minimum-variance delta, daily", None), ("no_hedge", "no hedge", None)):
            m = r["test"].get(key)
            if not m:
                continue
            runs.append({"run": "baseline", "label": f"{cellname}: {label}", "group": "baseline", "order": -1, "net": "\u2014", "fee": fee,
                         "dynamics": dyn, "product": prod, "status": "reference", "best_val_track": vt, "test_track": m.get("mean_abs_tracking_error"),
                         "test_cost": m.get("mean_cost"), "test_cvar": m.get("cvar_95"), "test_trades": m.get("mean_trades")})
    for fee, (b, ww) in sorted(baselines.items()):
        if ("bsm", "call", fee) in have:
            continue
        val_ww = min(ww.get("validation_scores", {}).values()) if ww.get("validation_scores") else None
        for key, label, vt in (("ww_band", f"Whalley\u2013Wilmott band (k={ww.get('param', float('nan')):.3g}, tuned on tracking)", val_ww),
                               ("model_delta", "Model delta, daily", None), ("no_hedge", "No hedge", None)):
            m = b.get(key)
            if not m:
                continue
            runs.append({"run": "baseline", "label": label, "group": "baseline", "order": -1, "net": "\u2014", "fee": fee, "dynamics": "bsm", "product": "call",
                         "status": "reference", "best_val_track": vt, "test_track": m.get("mean_abs_tracking_error"),
                         "test_cost": m.get("mean_cost"), "test_cvar": m.get("cvar_95"), "test_trades": m.get("mean_trades")})
    for (dyn, prod, fee), b in sorted(cell_baselines.items()):
        if (dyn, prod, fee) in have:
            continue
        cellname = f"{dyn.upper() if dyn == 'bsm' else 'Heston'} {prod}"
        for key, label in (("model_delta", "model delta, daily"), ("mv_delta", "minimum-variance delta, daily"), ("no_hedge", "no hedge")):
            m = b.get(key)
            if not m:
                continue
            runs.append({"run": "baseline", "label": f"{cellname}: {label}", "group": "baseline", "order": -1, "net": "\u2014", "fee": fee,
                         "dynamics": dyn, "product": prod, "status": "reference", "test_track": m.get("mean_abs_tracking_error"),
                         "test_cost": m.get("mean_cost"), "test_cvar": m.get("cvar_95"), "test_trades": m.get("mean_trades")})
    print(json.dumps({"curves": curves, "runs": runs}))


main()
