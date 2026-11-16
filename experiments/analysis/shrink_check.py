"""Does the 1% single net hold a shrunk version of the 0.1% single net's position? Same test paths, seed 1.
Per cell and instrument: OLS of the 1% position on the 0.1% position; hedge notional relative to the liability's premium;
turnover (sum |trade| x price) relative to premium for each fee.
Reads the seed-1 single nets' test traces under experiments/results/main and v2_4.
usage (from anywhere): python experiments/analysis/shrink_check.py"""
import numpy as np, glob, os
P = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiments", "results")
def load(dyn, prod, fee):
    if (dyn, prod) == ("bsm", "call"):
        d = f"{P}/v2_4/tgsg8_fee{fee}_s1"
    else:
        c = sorted(glob.glob(f"{P}/main/{dyn}_{prod}_fee{fee}_single_*_s1")); d = c[0]
    return np.load(os.path.join(d, "best_trial_test_paths.npz")), os.path.basename(d)
print("| cell | instr | slope 1% on 0.1% | R2 | mean|pos| 1% / 0.1% | hedge notional / premium (0.1%) | turnover / premium 0.1% -> 1% | trades per path 0.1% -> 1% |")
for dyn in ("bsm", "heston"):
    for prod in ("call", "doc", "con"):
        a, na = load(dyn, prod, "01"); b, nb = load(dyn, prod, "1")
        assert np.allclose(a["trace_spot"], b["trace_spot"]), "different paths"
        pa, pb, X, L = a["trace_position"], b["trace_position"], a["trace_instrument_prices"], a["trace_liability"]
        N = pa.shape[1]; prem = L[:, 0].mean()
        def turn(p):
            prev = np.concatenate([np.zeros_like(p[:, :1]), p[:, :-1]], 1)
            return (np.abs(p - prev) * X[:, :N]).sum((1, 2)).mean()
        for i in range(pa.shape[2]):
            x, y = pa[:, :, i].ravel(), pb[:, :, i].ravel()
            A = np.c_[np.ones_like(x), x]; w, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - ((y - A @ w) ** 2).sum() / ((y - y.mean()) ** 2).sum()
            notional = (np.abs(pa[:, :, i]) * X[:, :N, i]).mean() / prem
            print(f"| {dyn} {prod} | {i} | {w[1]:.2f} (+{w[0]:.2f}) | {r2:.2f} | {np.abs(pb[:,:,i]).mean()/np.abs(pa[:,:,i]).mean():.2f} | {notional:.1f}x | {turn(pa)/prem:.2f} -> {turn(pb)/prem:.2f} | {a['trace_n_trades'].mean():.0f} -> {b['trace_n_trades'].mean():.0f} |")
