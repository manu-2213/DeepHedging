"""
Heston Model Calibration v2 — Publication-Quality SOTA Pipeline
================================================================

State-of-the-art calibration of the Heston (1993) stochastic volatility
model from observed option prices.  Designed for publication in top-tier
finance journals (JFE, RFS, JF).

Key innovations over v1
-----------------------
1. **Vega-weighted implied-volatility objective**
   Minimises  WRMSE = sqrt[ Σ wᵢ (σᴴᵢ − σᴹᵢ)² / Σ wᵢ ]
   where wᵢ = BS-Vega(Kᵢ, Tᵢ), following Cont & Tankov (2004) and
   standard sell-side practice (Gatheral, 2006).  Illiquid deep-OTM
   options with tiny vega are down-weighted automatically.

2. **Feller-regularised loss**
   Soft penalty  λ · max(0, 1 − 2κθ/ξ²)²  encourages the variance
   process to stay strictly positive — vital for well-posed MC simulation.

3. **Logit parameter transform**
   Maps the bounded Heston space ↔ ℝ⁵ via sigmoid/logit, so gradient-based
   optimisers (L-BFGS-B) operate in a genuinely unconstrained landscape.
   This eliminates the bound-pegging pathology from v1 entirely.

4. **Four-phase optimisation**
     Phase A — Sobol-seeded Differential Evolution (global, bounded)
     Phase B — L-BFGS-B in transformed space (gradient-based, unbounded)
     Phase C — Nelder-Mead polish in transformed space (derivative-free)
     Phase D — Best-of-three selection

5. **Multi-date robustness**
   Calibrates to the top-N most liquid dates per market-year, keeps the
   result with the lowest WRMSE.

6. **2-D stratified sampling**
   Helpers are subsampled across a moneyness × maturity grid, with
   within-cell priority given to higher-vega instruments.

7. **Comprehensive diagnostics**
   Per-maturity and per-moneyness RMSE buckets, Feller ratio, parameter
   bound-proximity check, fully saved in JSON for paper tables.

8. **Resume capability**
   Already-calibrated (market, year) pairs are skipped automatically,
   so an interrupted run can be re-launched without losing progress.

Dependencies:  QuantLib >= 1.30, scipy >= 1.10, numpy, pandas

References
----------
[1] Heston S (1993). A closed-form solution for options with stochastic
    volatility. RFS 6(2), 327–343.
[2] Gatheral J (2006). The Volatility Surface. Wiley Finance.
[3] Cont R & Tankov P (2004). Financial Modelling with Jump Processes.
[4] Cui Y, Del Baño Rollin S & Germano G (2017). Full and fast Heston
    calibration. EJOR 263(2), 625–638.
[5] Fang F & Oosterlee CW (2008). A novel pricing method via Fourier-
    cosine series. SIAM J Sci Comput.

Usage
-----
    python option_calibration_v2.py
"""

from __future__ import annotations

import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import QuantLib as ql
from scipy.optimize import brentq, differential_evolution, minimize
from scipy.stats import norm

from utils import sample_strikes

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  1.  CONFIGURATION                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝

MARKETS = ["sp500", "alphabet", "apple", "microsoft", "nasdaq100"]
YEARS   = ["2017", "2018", "2019", "2020", "2021"]

# ── data filtering ────────────────────────────────────────────────────
MIN_TAU        = 0.04       # ~2 weeks  (exclude weeklies — noisy IVs)
MAX_TAU        = 1.50       # 18 months (long-dated tend to be illiquid)
MIN_MONEYNESS  = 0.85       # K/S lower bound
MAX_MONEYNESS  = 1.15       # K/S upper bound
MIN_IV         = 0.02       # floor on implied vol  (2 %)
MAX_IV         = 2.50       # cap   on implied vol  (250 %)

# ── calibration ───────────────────────────────────────────────────────
MAX_HELPERS       = 500     # helpers per calibration date
N_CANDIDATE_DATES = 3       # calibrate top-N liquid dates, keep best
N_DE_RESTARTS     = 3       # multi-restart Differential Evolution
DE_POPSIZE        = 15      # DE population size
DE_MAXITER        = 120     # DE generations per restart
LBFGSB_MAXITER    = 500     # L-BFGS-B iterations
NM_MAXITER        = 1000    # Nelder-Mead iterations
GL_QUADRATURE     = 128     # Gauss–Laguerre quadrature points

# ── regularisation ────────────────────────────────────────────────────
FELLER_LAMBDA = 0.01        # Feller soft-penalty weight

# ── parameter bounds ──────────────────────────────────────────────────
#   v0    : initial variance         √v0  ∈ [3 %, 55 %]
#   kappa : mean-reversion speed     typical equity range
#   theta : long-run variance        √θ   ∈ [3 %, 55 %]
#   rho   : spot–vol correlation     strongly negative for equities
#   xi    : vol-of-vol               moderate range
PARAM_NAMES  = ["v0", "kappa", "theta", "rho", "xi"]
PARAM_BOUNDS = np.array([
    [0.001,  0.30],     # v0
    [0.05,   8.00],     # kappa
    [0.001,  0.30],     # theta
    [-0.95,  0.10],     # rho
    [0.05,   2.50],     # xi
])
BOUNDS_LO    = PARAM_BOUNDS[:, 0]
BOUNDS_HI    = PARAM_BOUNDS[:, 1]
SCIPY_BOUNDS = list(map(tuple, PARAM_BOUNDS))

DATA_DIR   = Path(__file__).parent / "option_data"
OUTPUT_DIR = Path(__file__).parent


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  2.  LOGIT / SIGMOID PARAMETER TRANSFORM                            ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
#   θ ∈ [lo, hi]  ⟺  x ∈ ℝ
#   to_internal:   x  = logit( (θ − lo) / (hi − lo) )
#   to_external:   θ  = lo + (hi − lo) · σ(x)
#
# The transform is C∞-smooth and bijective, letting gradient-based
# optimisers explore the entire feasible region without hard walls.
# ────────────────────────────────────────────────────────────────────

_TR_EPS = 1e-8


def to_internal(theta: np.ndarray) -> np.ndarray:
    """Bounded θ → unconstrained x  (logit)."""
    theta = np.clip(theta, BOUNDS_LO + _TR_EPS, BOUNDS_HI - _TR_EPS)
    u = (theta - BOUNDS_LO) / (BOUNDS_HI - BOUNDS_LO)
    return np.log(u / (1.0 - u))


def to_external(x: np.ndarray) -> np.ndarray:
    """Unconstrained x → bounded θ  (sigmoid)."""
    u = 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))
    return BOUNDS_LO + (BOUNDS_HI - BOUNDS_LO) * u


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  3.  BLACK-SCHOLES FUNCTIONS                                         ║
# ╚══════════════════════════════════════════════════════════════════════╝

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


def bs_vega(S: float, K: float, r: float, T: float,
            sigma: float) -> float:
    """BS vega:  ∂C/∂σ = S √T φ(d₁).  Same for puts."""
    if sigma <= 0.0 or T <= 0.0:
        return 0.0
    sqrtT = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrtT)
    return float(S * sqrtT * norm.pdf(d1))


def implied_vol_single(S: float, K: float, r: float, T: float,
                       price: float, is_call: bool = True) -> float:
    """Implied vol via Brent's method.  Returns NaN on failure."""
    if price <= 0.0 or T <= 0.0 or S <= 0.0 or K <= 0.0:
        return np.nan
    intrinsic = (max(S - K * np.exp(-r * T), 0.0) if is_call
                 else max(K * np.exp(-r * T) - S, 0.0))
    if price < intrinsic - 1e-6:
        return np.nan
    try:
        return brentq(
            lambda sig: bs_price(S, K, r, T, sig, is_call) - price,
            1e-6, 5.0, xtol=1e-10, maxiter=200,
        )
    except (ValueError, RuntimeError):
        return np.nan


def batch_implied_vol(S, K, r, T, prices, is_call) -> np.ndarray:
    """Vectorised implied-vol computation."""
    n = len(S)
    out = np.empty(n)
    for i in range(n):
        out[i] = implied_vol_single(
            float(S[i]), float(K[i]), float(r[i]),
            float(T[i]), float(prices[i]), bool(is_call[i]),
        )
    return out


def batch_bs_vega(S, K, r, T, sigma) -> np.ndarray:
    """Vectorised BS vega."""
    n = len(S)
    out = np.empty(n)
    for i in range(n):
        out[i] = bs_vega(float(S[i]), float(K[i]), float(r[i]),
                         float(T[i]), float(sigma[i]))
    return out


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  4.  DATA PREPARATION                                               ║
# ╚══════════════════════════════════════════════════════════════════════╝

def bsm_summary(df: pd.DataFrame):
    """Replicate utils.get_BSM_data from a loaded DataFrame (avoids re-read)."""
    S0 = float(df["forward_price"].iloc[0])
    idx = df.groupby("date")["tau"].idxmin()
    daily = df.loc[idx, ["date", "forward_price"]].set_index("date").sort_index()
    daily["log_ret"] = np.log(
        daily["forward_price"] / daily["forward_price"].shift(1)
    )
    lr = daily["log_ret"].dropna()
    ann_ret = float(lr.mean() * 252)
    ann_vol = float(lr.std() * np.sqrt(252))
    strikes = df["strike_price"].unique().tolist()
    return S0, ann_ret, ann_vol, strikes


def to_ql_date(date_str: str) -> ql.Date:
    """'YYYY-MM-DD ...' → QuantLib Date."""
    dt = datetime.strptime(str(date_str)[:10], "%Y-%m-%d")
    return ql.Date(dt.day, dt.month, dt.year)


def select_calibration_dates(df: pd.DataFrame,
                             n: int = N_CANDIDATE_DATES) -> list[str]:
    """Return the *n* dates with the most traded option contracts."""
    counts = df.groupby("date").size().sort_values(ascending=False)
    return list(counts.index[:n])


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  5.  QUANTLIB HELPER CONSTRUCTION                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝

def build_ql_helpers(
    df_cal: pd.DataFrame,
    spot: float,
    risk_free_rate: float,
    calc_date_ql: ql.Date,
) -> tuple:
    """
    Build QuantLib HestonModelHelpers plus normalised vega weights.

    Returns
    -------
    helpers      : list[ql.HestonModelHelper]
    vega_weights : np.ndarray  (sums to 1)
    flat_ts, div_ts, spot_handle : QuantLib term-structure objects
    meta         : dict with 'tau', 'moneyness', 'strike' arrays
    """
    day_count = ql.Actual365Fixed()
    calendar  = ql.UnitedStates(ql.UnitedStates.NYSE)
    ql.Settings.instance().evaluationDate = calc_date_ql

    flat_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calc_date_ql, float(risk_free_rate), day_count)
    )
    div_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(calc_date_ql, 0.0, day_count)
    )
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(float(spot)))

    # --- pre-compute BS vegas ---
    S_arr  = df_cal["forward_price"].values.astype(np.float32)
    K_arr  = df_cal["strike_price"].values.astype(np.float32)
    r_arr  = df_cal["risk_free_rate"].values.astype(np.float32)
    T_arr  = df_cal["tau"].values.astype(np.float32)
    iv_arr = df_cal["implied_vol"].values.astype(np.float32)
    vegas  = batch_bs_vega(S_arr, K_arr, r_arr, T_arr, iv_arr)
    money  = K_arr / S_arr

    # --- build every valid helper ---
    all_helpers   : list = []
    all_vegas     : list = []
    all_taus      : list = []
    all_moneyness : list = []
    all_strikes   : list = []

    for i, (_, row) in enumerate(df_cal.iterrows()):
        iv = float(row["implied_vol"])
        v  = vegas[i]
        if not (np.isfinite(iv) and MIN_IV < iv < MAX_IV
                and np.isfinite(v) and v > 0):
            continue

        T = float(row["tau"])
        K = float(row["strike_price"])
        mat_days = max(int(round(T * 365)), 1)
        period   = ql.Period(mat_days, ql.Days)
        vol_q    = ql.QuoteHandle(ql.SimpleQuote(iv))

        helper = ql.HestonModelHelper(
            period, calendar, float(spot), K,
            vol_q, flat_ts, div_ts,
            ql.BlackCalibrationHelper.ImpliedVolError,
        )
        all_helpers.append(helper)
        all_vegas.append(v)
        all_taus.append(T)
        all_moneyness.append(money[i])
        all_strikes.append(K)

    # --- 2-D stratified subsample (moneyness × maturity) ---
    if len(all_helpers) > MAX_HELPERS:
        tau_a   = np.array(all_taus)
        mon_a   = np.array(all_moneyness)
        veg_a   = np.array(all_vegas)
        n_tbins = 5
        n_mbins = 5
        t_edges = np.linspace(tau_a.min(), tau_a.max() + 1e-6, n_tbins + 1)
        m_edges = np.linspace(mon_a.min(), mon_a.max() + 1e-6, n_mbins + 1)
        per_cell = max(MAX_HELPERS // (n_tbins * n_mbins), 1)
        rng = np.random.RandomState(42)
        selected: list[int] = []

        for ti in range(n_tbins):
            for mi in range(n_mbins):
                in_cell = np.where(
                    (tau_a >= t_edges[ti]) & (tau_a < t_edges[ti + 1])
                    & (mon_a >= m_edges[mi]) & (mon_a < m_edges[mi + 1])
                )[0]
                if len(in_cell) == 0:
                    continue
                take = min(per_cell, len(in_cell))
                # preferentially select higher-vega options within cell
                cell_v = veg_a[in_cell]
                prob   = cell_v / cell_v.sum()
                chosen = rng.choice(in_cell, take, replace=False, p=prob)
                selected.extend(chosen.tolist())

        # fill remaining budget, vega-preferential
        remaining = sorted(set(range(len(all_helpers))) - set(selected))
        budget    = MAX_HELPERS - len(selected)
        if budget > 0 and remaining:
            rem_v = veg_a[remaining]
            prob  = rem_v / rem_v.sum()
            extra = rng.choice(remaining,
                               min(budget, len(remaining)),
                               replace=False, p=prob)
            selected.extend(extra.tolist())

        selected = sorted(set(selected))
        all_helpers   = [all_helpers[i]   for i in selected]
        all_vegas     = [all_vegas[i]     for i in selected]
        all_taus      = [all_taus[i]      for i in selected]
        all_moneyness = [all_moneyness[i] for i in selected]
        all_strikes   = [all_strikes[i]   for i in selected]

    # normalise vega weights  →  Σ wᵢ = 1
    vega_arr     = np.array(all_vegas)
    vega_weights = vega_arr / vega_arr.sum()

    meta = {
        "tau":       np.array(all_taus),
        "moneyness": np.array(all_moneyness),
        "strike":    np.array(all_strikes),
    }
    return all_helpers, vega_weights, flat_ts, div_ts, spot_handle, meta


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  6.  OBJECTIVE FUNCTIONS                                             ║
# ╚══════════════════════════════════════════════════════════════════════╝

def _heston_wrmse(
    params: np.ndarray,
    helpers: list,
    vega_weights: np.ndarray,
    flat_ts,
    div_ts,
    spot_handle,
) -> float:
    """
    Vega-weighted implied-vol RMSE  +  Feller soft penalty.

        obj = WRMSE  +  λ · max(0, 1 − 2κθ/ξ²)²

    QuantLib's HestonModelHelper.calibrationError() returns
    σ_model − σ_market  when constructed with ImpliedVolError,
    which is exactly the IV error we want.
    """
    v0, kappa, theta, rho, xi = params

    try:
        process = ql.HestonProcess(
            flat_ts, div_ts, spot_handle,
            float(v0), float(kappa), float(theta),
            float(xi), float(rho),             # QL order: sigma then rho
        )
        model  = ql.HestonModel(process)
        engine = ql.AnalyticHestonEngine(model, GL_QUADRATURE)

        wsq   = 0.0
        wtot  = 0.0
        for i, h in enumerate(helpers):
            h.setPricingEngine(engine)
            err = h.calibrationError()
            if np.isfinite(err):
                w    = vega_weights[i]
                wsq += w * err * err
                wtot += w

        if wtot < 1e-15:
            return 1e6

        wrmse = np.sqrt(wsq / wtot)

        # Feller soft penalty
        feller = 2.0 * kappa * theta / (xi * xi)
        penalty = FELLER_LAMBDA * max(0.0, 1.0 - feller) ** 2

        return wrmse + penalty

    except Exception:
        return 1e6


def _obj_external(params, helpers, vw, fts, dts, sh):
    """Objective in original (bounded) parameter space."""
    return _heston_wrmse(params, helpers, vw, fts, dts, sh)


def _obj_internal(x, helpers, vw, fts, dts, sh):
    """Objective in logit-transformed (unconstrained) space."""
    return _heston_wrmse(to_external(x), helpers, vw, fts, dts, sh)


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  7.  CALIBRATION ENGINE  (4-phase)                                   ║
# ╚══════════════════════════════════════════════════════════════════════╝

def calibrate_heston_v2(
    helpers: list,
    vega_weights: np.ndarray,
    flat_ts, div_ts, spot_handle,
) -> tuple[float, np.ndarray]:
    """
    Four-phase state-of-the-art calibration.

    Phase A — Multi-restart Sobol-seeded Differential Evolution (global)
    Phase B — L-BFGS-B in logit-transformed space (gradient-based)
    Phase C — Adaptive Nelder-Mead in logit-transformed space
    Phase D — Best-of-three selection

    Returns (best_wrmse, best_params_array).
    """
    obj_e = lambda p: _obj_external(p, helpers, vega_weights,
                                     flat_ts, div_ts, spot_handle)
    obj_i = lambda x: _obj_internal(x, helpers, vega_weights,
                                     flat_ts, div_ts, spot_handle)

    candidates: list[tuple[float, np.ndarray]] = []

    # ─────────────────────────────────────────────────────────────
    # Phase A: Multi-restart Differential Evolution
    # ─────────────────────────────────────────────────────────────
    print(f"    Phase A: Differential Evolution "
          f"({N_DE_RESTARTS} restarts, pop={DE_POPSIZE}, "
          f"maxiter={DE_MAXITER}) ...")
    t0_all = time.time()
    best_de = None

    for restart in range(N_DE_RESTARTS):
        seed = 42 + restart * 13
        # Try Sobol init (scipy ≥ 1.10), fall back to Latin Hypercube
        try:
            de = differential_evolution(
                obj_e, bounds=SCIPY_BOUNDS, seed=seed,
                maxiter=DE_MAXITER, tol=1e-10, polish=False,
                popsize=DE_POPSIZE, mutation=(0.5, 1.5),
                recombination=0.85, init="sobol",
            )
        except TypeError:
            de = differential_evolution(
                obj_e, bounds=SCIPY_BOUNDS, seed=seed,
                maxiter=DE_MAXITER, tol=1e-10, polish=False,
                popsize=DE_POPSIZE, mutation=(0.5, 1.5),
                recombination=0.85, init="latinhypercube",
            )

        tag = ""
        if best_de is None or de.fun < best_de.fun:
            best_de = de
            tag = " ← best"
        print(f"      restart {restart + 1}/{N_DE_RESTARTS}  "
              f"WRMSE = {de.fun:.6e}{tag}")

    de_time = time.time() - t0_all
    print(f"    DE done: best = {best_de.fun:.6e}  ({de_time:.1f}s)")
    candidates.append((best_de.fun, best_de.x.copy()))

    # ─────────────────────────────────────────────────────────────
    # Phase B: L-BFGS-B in logit-transformed space
    # ─────────────────────────────────────────────────────────────
    print(f"    Phase B: L-BFGS-B (transformed space, "
          f"maxiter={LBFGSB_MAXITER}) ...")
    t0 = time.time()
    x0 = to_internal(best_de.x)
    lbfgs = minimize(obj_i, x0=x0, method="L-BFGS-B",
                     options={"maxiter": LBFGSB_MAXITER,
                              "ftol": 1e-14, "gtol": 1e-10})
    lbfgs_p = to_external(lbfgs.x)
    lbfgs_v = obj_e(lbfgs_p)
    print(f"    L-BFGS-B WRMSE = {lbfgs_v:.6e}  ({time.time() - t0:.1f}s)")
    candidates.append((lbfgs_v, lbfgs_p))

    # ─────────────────────────────────────────────────────────────
    # Phase C: Adaptive Nelder-Mead polish
    # ─────────────────────────────────────────────────────────────
    best_c_val, best_c_p = min(candidates, key=lambda c: c[0])
    x0_nm = to_internal(best_c_p)
    print(f"    Phase C: Nelder-Mead (adaptive, maxiter={NM_MAXITER}) ...")
    t0 = time.time()
    nm = minimize(obj_i, x0=x0_nm, method="Nelder-Mead",
                  options={"maxiter": NM_MAXITER,
                           "xatol": 1e-9, "fatol": 1e-12,
                           "adaptive": True})
    nm_p = to_external(nm.x)
    nm_v = obj_e(nm_p)
    print(f"    Nelder-Mead WRMSE = {nm_v:.6e}  ({time.time() - t0:.1f}s)")
    candidates.append((nm_v, nm_p))

    # ─────────────────────────────────────────────────────────────
    # Phase D: Selection
    # ─────────────────────────────────────────────────────────────
    best_val, best_p = min(candidates, key=lambda c: c[0])
    print(f"    >>> Overall best WRMSE = {best_val:.6e}")
    return best_val, best_p


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  8.  DIAGNOSTICS                                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def compute_diagnostics(
    best_params: np.ndarray,
    helpers: list,
    vega_weights: np.ndarray,
    meta: dict,
    flat_ts, div_ts, spot_handle,
) -> dict:
    """Per-bucket RMSE, Feller check, bound proximity."""
    v0, kappa, theta, rho, xi = best_params

    process = ql.HestonProcess(
        flat_ts, div_ts, spot_handle,
        float(v0), float(kappa), float(theta),
        float(xi), float(rho),
    )
    model  = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model, GL_QUADRATURE)

    errors = np.empty(len(helpers))
    for i, h in enumerate(helpers):
        h.setPricingEngine(engine)
        errors[i] = h.calibrationError()

    abs_err = np.abs(errors)
    wrmse   = float(np.sqrt(np.sum(vega_weights * errors ** 2)
                             / np.sum(vega_weights)))
    rmse    = float(np.sqrt(np.mean(errors ** 2)))
    mae     = float(np.mean(abs_err))
    max_err = float(np.max(abs_err))

    # ── per-maturity buckets ──
    tau = meta["tau"]
    TAU_EDGES  = [0, 0.08, 0.25, 0.50, 1.00, 2.00]
    TAU_LABELS = ["< 1M", "1–3M", "3–6M", "6–12M", "12M+"]
    tau_bkts: dict = {}
    for j, lbl in enumerate(TAU_LABELS):
        mask = (tau >= TAU_EDGES[j]) & (tau < TAU_EDGES[j + 1])
        if mask.sum() > 0:
            e = errors[mask]
            tau_bkts[lbl] = {
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "mae":  float(np.mean(np.abs(e))),
                "n":    int(mask.sum()),
            }

    # ── per-moneyness buckets ──
    money = meta["moneyness"]
    MON_EDGES  = [0.80, 0.90, 0.95, 1.00, 1.05, 1.10, 1.20]
    MON_LABELS = ["deep-OTM-P", "OTM-P", "ATM-P", "ATM-C", "OTM-C",
                  "deep-OTM-C"]
    mon_bkts: dict = {}
    for j, lbl in enumerate(MON_LABELS):
        mask = (money >= MON_EDGES[j]) & (money < MON_EDGES[j + 1])
        if mask.sum() > 0:
            e = errors[mask]
            mon_bkts[lbl] = {
                "rmse": float(np.sqrt(np.mean(e ** 2))),
                "mae":  float(np.mean(np.abs(e))),
                "n":    int(mask.sum()),
            }

    # ── Feller condition ──
    feller = 2.0 * kappa * theta / (xi ** 2)

    # ── bound proximity (0 = at bound, 0.5 = centred) ──
    prox: dict = {}
    for k, name in enumerate(PARAM_NAMES):
        lo, hi = BOUNDS_LO[k], BOUNDS_HI[k]
        val = best_params[k]
        prox[name] = float(min((val - lo), (hi - val)) / (hi - lo))

    return {
        "wrmse":             wrmse,
        "rmse":              rmse,
        "mae":               mae,
        "max_abs_error":     max_err,
        "n_helpers":         len(helpers),
        "feller_ratio":      float(feller),
        "feller_satisfied":  bool(feller > 1.0),
        "tau_buckets":       tau_bkts,
        "moneyness_buckets": mon_bkts,
        "bound_proximity":   prox,
    }


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  9.  PIPELINE  (one market-year)                                     ║
# ╚══════════════════════════════════════════════════════════════════════╝

def calibrate_market_year(market: str, year: str) -> dict | None:
    """Load → filter → IV → vega → calibrate → diagnose."""

    path = DATA_DIR / f"{market}_{year}.json.bz2"
    print(f"\n  Loading {path.name} ...")
    t0 = time.time()
    df = pd.read_json(path, compression="bz2", orient="index")
    print(f"  {len(df):,} rows loaded in {time.time() - t0:.1f}s")

    S0, ann_ret, ann_vol, all_strikes = bsm_summary(df)

    # ── pre-filter ──
    S = df["forward_price"].values.astype(np.float32)
    K = df["strike_price"].values.astype(np.float32)
    T = df["tau"].values.astype(np.float32)
    P = df["option_price"].values.astype(np.float32)
    mon = K / S
    mask = (
        (P > 0) & (S > 0) & (K > 0)
        & (T > MIN_TAU) & (T < MAX_TAU)
        & (mon > MIN_MONEYNESS) & (mon < MAX_MONEYNESS)
        & np.isfinite(P) & np.isfinite(S)
        & np.isfinite(K) & np.isfinite(T)
    )
    df = df[mask].copy()
    print(f"  After filter: {len(df):,} options")
    if len(df) < 50:
        print("  ⚠  Insufficient data — skipping.")
        return None

    # ── candidate dates ──
    cal_dates = select_calibration_dates(df, N_CANDIDATE_DATES)
    print(f"  Candidate dates ({len(cal_dates)}): "
          + ", ".join(str(d)[:10] for d in cal_dates))

    best_result: dict | None = None
    best_wrmse = float("inf")

    for di, cal_date in enumerate(cal_dates):
        print(f"\n  ── date {di + 1}/{len(cal_dates)}: "
              f"{str(cal_date)[:10]} ──")

        df_d = df[df["date"] == cal_date].copy()
        print(f"  {len(df_d)} options on this date")
        if len(df_d) < 20:
            print("  Too few — skip.")
            continue

        # implied vols
        t0 = time.time()
        df_d["implied_vol"] = batch_implied_vol(
            df_d["forward_price"].values.astype(np.float32),
            df_d["strike_price"].values.astype(np.float32),
            df_d["risk_free_rate"].values.astype(np.float32),
            df_d["tau"].values.astype(np.float32),
            df_d["option_price"].values.astype(np.float32),
            (df_d["is_call"].values == 1),
        )
        print(f"  IVs computed in {time.time() - t0:.1f}s")

        df_d = df_d.dropna(subset=["implied_vol"])
        df_d = df_d[(df_d["implied_vol"] > MIN_IV)
                     & (df_d["implied_vol"] < MAX_IV)]
        print(f"  Valid IVs: {len(df_d)}")
        if len(df_d) < 10:
            print("  Too few valid IVs — skip.")
            continue

        spot = float(df_d["forward_price"].iloc[0])
        rate = float(df_d["risk_free_rate"].median())
        ql_date = to_ql_date(cal_date)

        helpers, vw, fts, dts, sh, meta = build_ql_helpers(
            df_d, spot, rate, ql_date,
        )
        if len(helpers) < 10:
            print(f"  Only {len(helpers)} helpers — skip.")
            continue
        print(f"  {len(helpers)} vega-weighted helpers")

        t0 = time.time()
        wrmse, best_p = calibrate_heston_v2(helpers, vw, fts, dts, sh)
        cal_time = time.time() - t0
        print(f"  Calibration wall-time: {cal_time:.1f}s")

        if wrmse < best_wrmse:
            best_wrmse = wrmse
            diag = compute_diagnostics(best_p, helpers, vw, meta,
                                       fts, dts, sh)

            v0, kappa, theta, rho, xi = best_p
            best_result = {
                "v0":    float(v0),
                "kappa": float(kappa),
                "theta": float(theta),
                "rho":   float(rho),
                "xi":    float(xi),
                **diag,
                "S0":                float(S0),
                "annualized_vol":    float(ann_vol),
                "annualized_return": float(ann_ret),
                "calibration_date":  str(cal_date),
                "calibration_time_s": float(cal_time),
            }

            # K_sampled (for downstream compatibility)
            K_samp = sample_strikes(S0, all_strikes, 5, 2)
            if K_samp:
                best_result["atmK"]      = float(K_samp[len(K_samp) // 2])
                best_result["K_sampled"] = [float(k) for k in K_samp]
            else:
                best_result["atmK"]      = float(S0)
                best_result["K_sampled"] = []

    if best_result:
        print(f"\n  ✓ Best date: "
              f"{best_result['calibration_date'][:10]}")
    return best_result


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  10.  MAIN                                                           ║
# ╚══════════════════════════════════════════════════════════════════════╝

def main():
    total_t0 = time.time()
    all_results: dict[str, dict] = {}

    for market in MARKETS:
        print("\n" + "╔" + "═" * 58 + "╗")
        print(f"║  CALIBRATING: {market.upper():<43}║")
        print("╚" + "═" * 58 + "╝")

        # ── resume: load existing results ──
        out_path = OUTPUT_DIR / f"heston_params_QL_v2_{market}.json"
        existing: dict = {}
        if out_path.exists():
            with open(out_path) as f:
                existing = json.load(f)

        all_results[market] = dict(existing)

        for year in YEARS:
            # skip if we already have this year
            if year in existing:
                print(f"\n  {market.upper()} — {year}:  already calibrated "
                      f"(WRMSE = {existing[year].get('wrmse', '?')}) — skip.")
                continue

            print(f"\n{'─' * 55}")
            print(f"  {market.upper()} — {year}")
            print(f"{'─' * 55}")

            result = calibrate_market_year(market, year)

            if result is not None:
                all_results[market][year] = result

                fk = "✓" if result["feller_satisfied"] else "✗"
                print(f"\n  ✓ Fitted Heston parameters:")
                print(f"    v0    = {result['v0']:.6f}   "
                      f"(√v0 = {np.sqrt(result['v0']) * 100:.2f}%)")
                print(f"    κ     = {result['kappa']:.4f}")
                print(f"    θ     = {result['theta']:.6f}   "
                      f"(√θ = {np.sqrt(result['theta']) * 100:.2f}%)")
                print(f"    ρ     = {result['rho']:.4f}")
                print(f"    ξ     = {result['xi']:.4f}")
                print(f"    WRMSE = {result['wrmse']:.6e}  (vega-weighted)")
                print(f"    RMSE  = {result['rmse']:.6e}  (unweighted)")
                print(f"    Feller 2κθ/ξ² = "
                      f"{result['feller_ratio']:.3f}  {fk}")

                # bound-proximity warnings
                for pn, px in result["bound_proximity"].items():
                    if px < 0.05:
                        print(f"    ⚠  {pn} within 5 % of its bounds")

                # per-maturity RMSE
                if result.get("tau_buckets"):
                    print("    Maturity-bucket RMSE:")
                    for bkt, st in result["tau_buckets"].items():
                        print(f"      {bkt:>8s}:  {st['rmse']:.4e}  "
                              f"(n={st['n']})")
            else:
                print(f"  ✗ FAILED for {market} {year}")

            # ── incremental save after each year ──
            with open(out_path, "w") as f:
                json.dump(all_results[market], f, indent=2)

        print(f"\n  → Results in {out_path.name}")

    # ──────────────────────────────────────────────────────────────
    #  Summary table
    # ──────────────────────────────────────────────────────────────
    elapsed = time.time() - total_t0
    print("\n" + "═" * 74)
    print("  CALIBRATION SUMMARY   (Heston v2 — Vega-Weighted, "
          "Feller-Regularised)")
    print("═" * 74)

    for market in MARKETS:
        print(f"\n  {market.upper()}:")
        for year in YEARS:
            p = all_results.get(market, {}).get(year)
            if p:
                fk = "✓" if p["feller_satisfied"] else "✗"
                print(
                    f"    {year}:  √v0={np.sqrt(p['v0']) * 100:5.1f}%  "
                    f"κ={p['kappa']:5.2f}  "
                    f"√θ={np.sqrt(p['theta']) * 100:5.1f}%  "
                    f"ρ={p['rho']:+.3f}  "
                    f"ξ={p['xi']:.3f}  "
                    f"WRMSE={p['wrmse']:.2e}  "
                    f"Feller={p['feller_ratio']:.2f}{fk}"
                )
            else:
                print(f"    {year}:  FAILED")

    print(f"\n  Total elapsed: {elapsed / 60:.1f} min\n")


if __name__ == "__main__":
    main()
