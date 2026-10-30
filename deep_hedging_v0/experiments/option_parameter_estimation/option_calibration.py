"""
Heston Model Calibration using QuantLib + scipy
=================================================
Calibrates Heston stochastic volatility model parameters from market option
data for multiple equity markets (alphabet, apple, microsoft, nasdaq100).

Uses QuantLib's AnalyticHestonEngine for fast Heston pricing, combined with
scipy.optimize.differential_evolution for bounded global optimisation and
L-BFGS-B for polishing.  This avoids the unconstrained Levenberg-Marquardt
problem that produces absurd parameters (κ > 100k, ξ > 10k, etc).

Data layout (per bz2 file):
    Columns: forward_price, strike_price, risk_free_rate, tau,
             option_price, is_call (1/-1), date

Usage:
    python option_calibration.py
"""

import numpy as np
import pandas as pd
import json
import time
from pathlib import Path
from datetime import datetime
from scipy.optimize import brentq, differential_evolution, minimize
from scipy.stats import norm

import QuantLib as ql

from utils import sample_strikes

# ============================================================
# Configuration
# ============================================================

MARKETS = ["alphabet", "apple", "microsoft", "nasdaq100"]
YEARS = ["2017", "2018", "2019", "2020", "2021"]

# Filtering thresholds
MIN_TAU = 0.02          # minimum maturity in years
MAX_TAU = 2.0           # maximum maturity
MIN_MONEYNESS = 0.80    # K / S lower bound
MAX_MONEYNESS = 1.20    # K / S upper bound
MAX_HELPERS = 300       # good coverage without excessive cost per eval
N_RESTARTS = 3          # multi-restart DE for robustness

# Realistic Heston parameter bounds
#   v0     : initial variance          (√v0 in ~3–60%)
#   kappa  : mean-reversion speed      (0.1 – 10)
#   theta  : long-run variance         (√θ in ~3–60%)
#   rho    : spot–vol correlation      (-0.95 – +0.15)  slightly positive OK for some equities
#   xi     : vol-of-vol               (0.05 – 3.0)  single stocks need wider range
PARAM_BOUNDS = [
    (0.001, 0.40),   # v0
    (0.05,  10.0),   # kappa
    (0.001, 0.40),   # theta
    (-0.95,  0.15),  # rho
    (0.05,   3.0),   # xi
]
PARAM_NAMES = ["v0", "kappa", "theta", "rho", "xi"]

DATA_DIR = Path(__file__).parent / "option_data"
OUTPUT_DIR = Path(__file__).parent


# ============================================================
# Black-Scholes implied volatility
# ============================================================

def bs_price(S: float, K: float, r: float, T: float,
             sigma: float, is_call: bool = True) -> float:
    """Closed-form Black-Scholes European option price."""
    if sigma <= 0.0 or T <= 0.0:
        return max(S - K, 0.0) if is_call else max(K - S, 0.0)
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    if is_call:
        return float(S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return float(K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))


def implied_vol_single(S: float, K: float, r: float, T: float,
                       price: float, is_call: bool = True) -> float:
    """Implied volatility via Brent's method. Returns NaN on failure."""
    if price <= 0.0 or T <= 0.0 or S <= 0.0 or K <= 0.0:
        return np.nan
    intrinsic = (max(S - K * np.exp(-r * T), 0.0) if is_call
                 else max(K * np.exp(-r * T) - S, 0.0))
    if price < intrinsic - 1e-6:
        return np.nan
    try:
        return brentq(
            lambda sig: bs_price(S, K, r, T, sig, is_call) - price,
            1e-6, 5.0, xtol=1e-10, maxiter=100,
        )
    except (ValueError, RuntimeError):
        return np.nan


def batch_implied_vol(S, K, r, T, prices, is_call):
    """Compute implied vols for arrays of option data."""
    n = len(S)
    ivs = np.empty(n)
    for i in range(n):
        ivs[i] = implied_vol_single(
            float(S[i]), float(K[i]), float(r[i]),
            float(T[i]), float(prices[i]), bool(is_call[i]),
        )
    return ivs


# ============================================================
# Helper: BSM summary statistics from loaded DataFrame
# ============================================================

def bsm_summary(df: pd.DataFrame):
    """
    Replicate what utils.get_BSM_data returns, but from an
    already-loaded DataFrame (avoids re-reading the bz2 file).

    Returns (S0, annualized_return, annualized_vol, strikes_list).
    """
    S0 = float(df["forward_price"].iloc[0])

    idx = df.groupby("date")["tau"].idxmin()
    daily_spot = (df.loc[idx, ["date", "forward_price"]]
                  .set_index("date").sort_index())
    daily_spot["log_ret"] = np.log(
        daily_spot["forward_price"] / daily_spot["forward_price"].shift(1)
    )
    log_ret = daily_spot["log_ret"].dropna()
    ann_return = float(log_ret.mean() * 252)
    ann_vol = float(log_ret.std() * np.sqrt(252))
    strikes = df["strike_price"].unique().tolist()
    return S0, ann_return, ann_vol, strikes


# ============================================================
# QuantLib helpers
# ============================================================

def to_ql_date(date_str: str) -> ql.Date:
    """Convert 'YYYY-MM-DD' string to a QuantLib Date."""
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    return ql.Date(dt.day, dt.month, dt.year)


def select_calibration_date(df: pd.DataFrame) -> str:
    """Return the date with the most option contracts."""
    return df.groupby("date").size().idxmax()


# ============================================================
# Core QuantLib Heston calibration (bounded via scipy)
# ============================================================

def _build_ql_helpers(df_date, spot, risk_free_rate, calc_date_ql):
    """
    Set up QuantLib term structures and HestonModelHelpers.
    Returns (helpers, flat_ts, dividend_ts, spot_handle) or raises.
    """
    day_count = ql.Actual365Fixed()
    calendar = ql.UnitedStates(ql.UnitedStates.NYSE)
    ql.Settings.instance().evaluationDate = calc_date_ql

    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calc_date_ql, float(risk_free_rate), day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calc_date_ql, 0.0, day_count)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))

    helpers = []
    taus = []
    for _, row in df_date.iterrows():
        iv = row["implied_vol"]
        if not np.isfinite(iv) or iv <= 0.01 or iv > 3.0:
            continue

        T = float(row["tau"])
        K = float(row["strike_price"])
        maturity_days = max(int(round(T * 365)), 1)
        period = ql.Period(maturity_days, ql.Days)

        vol_quote = ql.QuoteHandle(ql.SimpleQuote(float(iv)))
        helper = ql.HestonModelHelper(
            period, calendar, float(spot), K,
            vol_quote, flat_ts, dividend_ts,
            ql.BlackCalibrationHelper.ImpliedVolError,
        )
        helpers.append(helper)
        taus.append(T)

    # Stratified subsample: keep proportional representation across maturities
    if len(helpers) > MAX_HELPERS:
        taus_arr = np.array(taus)
        n_bins = 10
        bin_edges = np.linspace(taus_arr.min(), taus_arr.max() + 1e-6, n_bins + 1)
        selected = []
        per_bin = max(MAX_HELPERS // n_bins, 1)
        rng = np.random.RandomState(42)
        for b in range(n_bins):
            in_bin = np.where((taus_arr >= bin_edges[b]) & (taus_arr < bin_edges[b+1]))[0]
            if len(in_bin) == 0:
                continue
            take = min(per_bin, len(in_bin))
            selected.extend(rng.choice(in_bin, take, replace=False).tolist())
        # Fill remaining budget randomly from leftovers
        remaining = set(range(len(helpers))) - set(selected)
        budget = MAX_HELPERS - len(selected)
        if budget > 0 and remaining:
            extra = rng.choice(list(remaining), min(budget, len(remaining)), replace=False)
            selected.extend(extra.tolist())
        helpers = [helpers[i] for i in sorted(selected)]

    return helpers, flat_ts, dividend_ts, spot_handle


def _heston_rmse(params, helpers, flat_ts, dividend_ts, spot_handle):
    """
    Compute RMSE of Heston model implied-vol errors for a given
    parameter vector [v0, kappa, theta, rho, xi].

    Rebuilds the QuantLib model/engine each call (cheap relative to
    the pricing itself).
    """
    v0, kappa, theta, rho, xi = params

    try:
        process = ql.HestonProcess(
            flat_ts, dividend_ts, spot_handle,
            float(v0), float(kappa), float(theta), float(xi), float(rho),
        )
        model = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model, 96)

        total_sq = 0.0
        n = 0
        for h in helpers:
            h.setPricingEngine(engine)
            err = h.calibrationError()
            if np.isfinite(err):
                total_sq += err * err
                n += 1

        if n == 0:
            return 1e6
        return np.sqrt(total_sq / n)

    except Exception:
        return 1e6


def calibrate_heston_ql(
    df_date: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    calc_date_ql: ql.Date,
    atm_iv: float | None = None,
) -> dict | None:
    """
    Calibrate the Heston model to a single date's option surface.

    Uses scipy.optimize.differential_evolution (bounded, global) followed
    by L-BFGS-B polishing.  All five Heston parameters are strictly
    constrained to PARAM_BOUNDS.

    Parameters
    ----------
    df_date : DataFrame  — one-date option data with implied_vol column.
    spot : float          — spot price.
    risk_free_rate : float
    calc_date_ql : ql.Date
    atm_iv : float | None — ATM implied vol (used to seed the initial guess).

    Returns
    -------
    dict with calibrated parameters, or None on failure.
    """

    helpers, flat_ts, div_ts, spot_handle = _build_ql_helpers(
        df_date, spot, risk_free_rate, calc_date_ql,
    )
    if len(helpers) < 5:
        print(f"    Too few valid helpers ({len(helpers)}), skipping.")
        return None

    print(f"    Using {len(helpers)} option helpers for calibration.")

    obj = lambda p: _heston_rmse(p, helpers, flat_ts, div_ts, spot_handle)

    # ==================================================================
    # Phase 1 — Multi-restart Differential Evolution (global search)
    # ==================================================================
    print(f"    Phase 1: Differential Evolution ({N_RESTARTS} restarts) ...")
    t0_all = time.time()
    best_de = None

    for restart in range(N_RESTARTS):
        seed = 42 + restart * 7
        de_result = differential_evolution(
            obj,
            bounds=PARAM_BOUNDS,
            seed=seed,
            maxiter=100,
            tol=1e-8,
            polish=False,
            popsize=15,
            mutation=(0.5, 1.5),
            recombination=0.85,
        )
        tag = "*" if (best_de is None or de_result.fun < best_de.fun) else " "
        if best_de is None or de_result.fun < best_de.fun:
            best_de = de_result
        print(f"      restart {restart+1}/{N_RESTARTS}  "
              f"seed={seed}  RMSE={de_result.fun:.6e} {tag}")

    print(f"    DE best RMSE = {best_de.fun:.6e}  "
          f"({time.time()-t0_all:.1f}s total)")

    # ==================================================================
    # Phase 2 — L-BFGS-B polish (gradient-based, bounded)
    # ==================================================================
    print("    Phase 2: L-BFGS-B polishing ...")
    t0 = time.time()
    lbfgs_result = minimize(
        obj,
        x0=best_de.x,
        method="L-BFGS-B",
        bounds=PARAM_BOUNDS,
        options={"maxiter": 300, "ftol": 1e-11},
    )
    print(f"    L-BFGS-B RMSE = {lbfgs_result.fun:.6e}  ({time.time()-t0:.1f}s)")

    # ==================================================================
    # Phase 3 — Nelder-Mead polish (derivative-free, can escape saddles)
    # ==================================================================
    best_so_far = lbfgs_result if lbfgs_result.fun <= best_de.fun else best_de
    print("    Phase 3: Nelder-Mead polishing ...")
    t0 = time.time()
    nm_result = minimize(
        obj,
        x0=best_so_far.x,
        method="Nelder-Mead",
        options={"maxiter": 500, "xatol": 1e-7, "fatol": 1e-9, "adaptive": True},
    )
    # Clip back into bounds (Nelder-Mead is unconstrained)
    nm_x = np.clip(nm_result.x, [b[0] for b in PARAM_BOUNDS],
                                 [b[1] for b in PARAM_BOUNDS])
    nm_fun = obj(nm_x)
    print(f"    Nelder-Mead RMSE = {nm_fun:.6e}  ({time.time()-t0:.1f}s)")

    # Pick overall best
    candidates = [
        (best_de.fun, best_de.x),
        (lbfgs_result.fun, lbfgs_result.x),
        (nm_fun, nm_x),
    ]
    best_fun, best_x = min(candidates, key=lambda c: c[0])
    v0, kappa, theta, rho, xi = best_x
    rmse = float(best_fun)
    print(f"    >>> Best RMSE = {rmse:.6e}")

    # --- Per-helper error stats ---
    # Rebuild model with best params for final error vector
    process = ql.HestonProcess(
        flat_ts, div_ts, spot_handle,
        float(v0), float(kappa), float(theta), float(xi), float(rho),
    )
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model, 96)
    errors = []
    for h in helpers:
        h.setPricingEngine(engine)
        errors.append(h.calibrationError())
    errors = np.array(errors)
    avg_abs_err = float(np.mean(np.abs(errors)))

    # --- Feller condition check ---
    feller = 2.0 * kappa * theta / (xi ** 2)
    feller_ok = bool(feller > 1.0)

    return {
        "v0": float(v0),
        "kappa": float(kappa),
        "theta": float(theta),
        "rho": float(rho),
        "xi": float(xi),
        "rmse": rmse,
        "avg_abs_error": avg_abs_err,
        "n_helpers": len(helpers),
        "feller_ratio": float(feller),
        "feller_satisfied": feller_ok,
    }


# ============================================================
# Pipeline for one (market, year)
# ============================================================

def calibrate_market_year(market: str, year: str) -> dict | None:
    """Load data ➜ compute IVs ➜ calibrate Heston for one market-year."""

    path = DATA_DIR / f"{market}_{year}.json.bz2"
    print(f"  Loading {path.name} ...")
    t0 = time.time()
    df = pd.read_json(path, compression="bz2", orient="index")
    print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

    # --- BSM summary stats (no second file read) ---
    S0, ann_ret, ann_vol, all_strikes = bsm_summary(df)

    # --- Pre-filter (before computing IVs) ---
    S = df["forward_price"].values.astype(np.float32)
    K = df["strike_price"].values.astype(np.float32)
    T = df["tau"].values.astype(np.float32)
    P = df["option_price"].values.astype(np.float32)
    moneyness = K / S

    valid = (
        (P > 0) & (S > 0) & (K > 0)
        & (T > MIN_TAU) & (T < MAX_TAU)
        & (moneyness > MIN_MONEYNESS) & (moneyness < MAX_MONEYNESS)
        & np.isfinite(P) & np.isfinite(S)
        & np.isfinite(K) & np.isfinite(T)
    )
    df = df[valid].copy()
    print(f"  After moneyness/maturity filter: {len(df):,}")

    if len(df) < 20:
        print("  ⚠️  Not enough data – skipping.")
        return None

    # --- Pick calibration date (most liquid) ---
    cal_date = select_calibration_date(df)
    df_date = df[df["date"] == cal_date].copy()
    print(f"  Calibration date: {cal_date}  ({len(df_date)} options)")

    # --- Compute implied vols only for that date ---
    print(f"  Computing implied vols for {len(df_date)} options ...")
    t0 = time.time()
    df_date["implied_vol"] = batch_implied_vol(
        df_date["forward_price"].values.astype(np.float32),
        df_date["strike_price"].values.astype(np.float32),
        df_date["risk_free_rate"].values.astype(np.float32),
        df_date["tau"].values.astype(np.float32),
        df_date["option_price"].values.astype(np.float32),
        (df_date["is_call"].values == 1),
    )
    print(f"  Implied vols computed in {time.time()-t0:.1f}s")

    # Drop failed IVs
    df_date = df_date.dropna(subset=["implied_vol"])
    df_date = df_date[
        (df_date["implied_vol"] > 0.01) & (df_date["implied_vol"] < 3.0)
    ]
    print(f"  Valid options with IVs: {len(df_date)}")

    if len(df_date) < 5:
        print("  ⚠️  Too few options with valid IVs – skipping.")
        return None

    spot = float(df_date["forward_price"].iloc[0])
    rate = float(df_date["risk_free_rate"].median())
    calc_date_ql = to_ql_date(cal_date)

    # --- Calibrate ---
    result = calibrate_heston_ql(df_date, spot, rate, calc_date_ql)
    if result is None:
        return None

    # --- Augment with BSM summary data (for downstream compatibility) ---
    K_sampled = sample_strikes(S0, all_strikes, 5, 2)
    atmK = K_sampled[len(K_sampled) // 2] if K_sampled else S0

    result.update({
        "S0": float(S0),
        "atmK": float(atmK),
        "K_sampled": [float(k) for k in K_sampled] if K_sampled else [],
        "annualized_vol": float(ann_vol),
        "annualized_return": float(ann_ret),
        "calibration_date": str(cal_date),
    })
    return result


# ============================================================
# Main
# ============================================================

def main():
    all_results = {}

    for market in MARKETS:
        print("\n" + "#" * 60)
        print(f"   CALIBRATING: {market.upper()}")
        print("#" * 60)

        all_results[market] = {}

        for year in YEARS:
            print(f"\n{'=' * 50}")
            print(f"  {market.upper()} — {year}")
            print(f"{'=' * 50}")

            result = calibrate_market_year(market, year)

            if result is not None:
                all_results[market][year] = result

                feller_str = "✓" if result["feller_satisfied"] else "✗"
                print(f"\n  ✓ Calibrated Heston parameters:")
                print(f"    v0    = {result['v0']:.6f}   "
                      f"(√v0 = {np.sqrt(result['v0'])*100:.2f}%)")
                print(f"    κ     = {result['kappa']:.4f}")
                print(f"    θ     = {result['theta']:.6f}   "
                      f"(√θ = {np.sqrt(result['theta'])*100:.2f}%)")
                print(f"    ρ     = {result['rho']:.4f}")
                print(f"    ξ     = {result['xi']:.4f}")
                print(f"    RMSE  = {result['rmse']:.6e}")
                print(f"    Feller 2κθ/ξ² = {result['feller_ratio']:.3f}  {feller_str}")
                print(f"    Realized vol = {result['annualized_vol']*100:.2f}%")
            else:
                print(f"  ✗ Calibration FAILED for {market} {year}")

        # Save per-market results
        out_path = OUTPUT_DIR / f"heston_params_QL_{market}.json"
        with open(out_path, "w") as f:
            json.dump(all_results[market], f, indent=2)
        print(f"\n  → Saved to {out_path.name}")

    # ----------------------------------------------------------
    # Summary table
    # ----------------------------------------------------------
    print("\n" + "=" * 70)
    print("  CALIBRATION SUMMARY  (QuantLib Heston)")
    print("=" * 70)

    for market in MARKETS:
        print(f"\n  {market.upper()}:")
        for year in YEARS:
            if year in all_results.get(market, {}):
                p = all_results[market][year]
                print(
                    f"    {year}:  √v0={np.sqrt(p['v0'])*100:5.1f}%  "
                    f"√θ={np.sqrt(p['theta'])*100:5.1f}%  "
                    f"ρ={p['rho']:+.2f}  "
                    f"ξ={p['xi']:.2f}  "
                    f"RMSE={p['rmse']:.2e}"
                )
            else:
                print(f"    {year}:  FAILED")

    print()


if __name__ == "__main__":
    main()
