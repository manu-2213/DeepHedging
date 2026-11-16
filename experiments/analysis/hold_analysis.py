"""Does the inactor's hold decision look like a band (a function of the proposed trade size) or like something else?
Logistic regressions of hold_t on (i) the proposed trade's notional, (ii) market state, (iii) both; AUC on held-out paths.
Reads the test traces (best_trial_test_paths.npz) of the Actor-Inactor runs under experiments/results/main and v2_4.
usage (from anywhere): python experiments/analysis/hold_analysis.py"""
import numpy as np, glob, os, sys
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "results")
rng = np.random.default_rng(0)
def auc(y, s):
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(len(s))
    n1 = y.sum(); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 - 1) / 2) / (n1 * n0)
def logit_fit(X, y, l2=1e-3, it=60):
    X = np.c_[np.ones(len(X)), X]; w = np.zeros(X.shape[1])
    for _ in range(it):
        p = 1 / (1 + np.exp(-X @ w)); g = X.T @ (p - y) + l2 * w
        H = (X * (p * (1 - p))[:, None]).T @ X + l2 * np.eye(len(w)); w -= np.linalg.solve(H, g)
    return w
def zs(X): return (X - X.mean(0)) / (X.std(0) + 1e-9)
print("| run | hold rate | AUC trade size | AUC state (S/K, t, returns) | AUC both | hold rate: smallest / largest trade-size quintile |")
for d in sorted(glob.glob(f"{RES}/main/*double*")) + sorted(glob.glob(f"{RES}/v2_4/dbltg8*")):
    f = os.path.join(d, "best_trial_test_paths.npz")
    if not os.path.exists(f): continue
    z = np.load(f); H = z["trace_hold"]; P, N = H.shape
    pos, a, X, S = z["trace_position"], z["trace_proposed"], z["trace_instrument_prices"], z["trace_spot"]
    prev = np.concatenate([np.zeros_like(pos[:, :1]), pos[:, :-1]], 1)          # holding before the step
    notional = (np.abs(a - prev) * X[:, :N]).sum(-1)                           # proposed trade size in money
    K = z["env_K"][:P] if "env_K" in z.files else np.full(P, np.nan)
    t = np.broadcast_to(np.arange(N) / N, (P, N))
    ret1 = np.diff(S, axis=1)[:, :N] / S[:, :N]
    ret5 = (S[:, :N] - np.concatenate([np.repeat(S[:, :1], 5, 1), S[:, :-5]], 1)[:, :N]) / S[:, :N]
    mny = S[:, :N] / K[:, None] if np.isfinite(K).all() else S[:, :N] / S[:, :1]
    y = H.reshape(-1); keep = np.ones(P * N, bool)
    Ftrade = zs(np.c_[np.log1p(notional.reshape(-1)), notional.reshape(-1)])
    Fstate = zs(np.c_[mny.reshape(-1), (mny**2).reshape(-1), t.reshape(-1), (t**2).reshape(-1), ret1.reshape(-1), np.abs(ret1).reshape(-1), ret5.reshape(-1), np.abs(ret5).reshape(-1)])
    paths = np.repeat(np.arange(P), N); test = paths >= int(0.7 * P); tr = ~test
    res = []
    for F in (Ftrade, Fstate, np.c_[Ftrade, Fstate]):
        w = logit_fit(F[tr], y[tr]); s = np.c_[np.ones(test.sum()), F[test]] @ w; res.append(auc(y[test], s))
    q = np.quantile(notional, [0.2, 0.8]); hr_small = H[notional <= q[0]].mean(); hr_big = H[notional >= q[1]].mean()
    print(f"| {os.path.basename(d)} | {H.mean():.2f} | {res[0]:.3f} | {res[1]:.3f} | {res[2]:.3f} | {hr_small:.2f} / {hr_big:.2f} |")
