"""Path-level paper figures from the per-path test outputs (seed 1 of every cell, per-seed-selected runs, listed in
experiments/analysis/selected_runs.json) and the model-band path dumps, all read from experiments/results (the runs'
*_test_paths.npz, main_baselines/*_paths.npz from evaluation/band_paths.py, main_big/*.npz from evaluation/eval_big.py).
Output: paper_notes/figures/*.pdf and *.png (PAPER=1: paper_notes/figures/paper).
usage (from anywhere): python experiments/figures/make_path_figs.py   FIG_OUT=<folder> overrides the output folder"""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = f"{REPO}/experiments/results"
OUT = os.environ.get("FIG_OUT", f"{REPO}/paper_notes/figures")
BLUE, ORANGE, AQUA, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#b5b4ae"
COL = {"single": BLUE, "double": ORANGE, "band": AQUA, "nohedge": GREY, "target": INK}
NAME = {"single": "single net", "double": "double net", "band": "model band", "nohedge": "no hedge", "target": "model target, daily"}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "sans-serif", "pdf.fonttype": 42})
CELLS = [('bsm', 'call', 'BSM call'), ('bsm', 'doc', 'BSM down-and-out call'), ('bsm', 'con', 'BSM cash-or-nothing'),
         ('heston', 'call', 'Heston call'), ('heston', 'doc', 'Heston down-and-out call'), ('heston', 'con', 'Heston cash-or-nothing')]
FEEDIR = {'0.1%': '01', '1%': '1'}
SEL = json.load(open(f"{REPO}/experiments/analysis/selected_runs.json"))

def rl(dyn, prod, fee, net, seed=1):
    run = SEL[f"{dyn}|{prod}|{fee}|{seed}"][net].replace("v2.4/", "v2_4/")
    return np.load(f"{RES}/{run}/best_trial_test_paths.npz")

def nohedge(dyn, prod, fee):
    for net in ("double", "single"):
        run = SEL[f"{dyn}|{prod}|{fee}|1"][net].replace("v2.4/", "v2_4/")
        p = f"{RES}/{run}/baselines_test_paths.npz"
        if os.path.exists(p):
            return np.load(p)
    return None

def bandf(dyn, prod, fee):
    p = f"{RES}/main_baselines/{dyn}_{prod}_fee{FEEDIR[fee]}_paths.npz"
    return np.load(p) if os.path.exists(p) else None

PAPER = os.environ.get("PAPER") == "1"      # PAPER=1: paper wording, no in-figure super-titles, output to figures/paper/
if PAPER:
    OUT = OUT + "/paper"; os.makedirs(OUT, exist_ok=True)
RENAME = [("single net", "single policy"), ("double net (joint phase)", "Actor\u2013Inactor (joint stage)"), ("double net", "Actor\u2013Inactor"),
          ("Double net", "Actor\u2013Inactor"), ("Single net", "Single policy"), ("double / single", "ratio"),
          ("model band (Whalley\u2013Wilmott-style)", "model band"), ("the inaction net", "the inaction network"), ("inactor", "inaction network")]

def _paperise(fig):
    if PAPER and fig._suptitle is not None:
        fig._suptitle.set_visible(False)
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label] + list(ax.texts)
        if ax.get_legend():
            texts += ax.get_legend().get_texts()
    for lg in fig.legends:
        texts += lg.get_texts()
    for t in texts:
        s = t.get_text()
        for a, b in RENAME:
            s = s.replace(a, b)
        t.set_text(s)

def save(fig, name):
    _paperise(fig)
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight"); fig.savefig(f"{OUT}/{name}.png", bbox_inches="tight"); plt.close(fig)
    print("wrote", name)

def kde(x, grid, bw=None):
    x = np.asarray(x, float); n = len(x)
    bw = bw or 1.06 * min(x.std(), (np.percentile(x, 75) - np.percentile(x, 25)) / 1.34) * n ** (-0.2)
    # binned KDE for speed
    h, e = np.histogram(x, bins=400, range=(grid[0] - 4 * bw, grid[-1] + 4 * bw))
    c = 0.5 * (e[1:] + e[:-1])
    d = np.exp(-0.5 * ((grid[:, None] - c[None, :]) / bw) ** 2).dot(h) / (n * bw * np.sqrt(2 * np.pi))
    return d

# ------------------------------------------------------------------ Fig A: terminal P&L histograms (fee 1%)
def bigf(dyn, prod, fee, net):
    p = f"{RES}/main_big/{dyn}_{prod}_fee{FEEDIR[fee]}_{net}.npz"
    return np.load(p) if os.path.exists(p) else None

def fig_track_hist(fee, name):
    """Per-path tracking error (mean |portfolio - liability| over the path's steps), single vs double net."""
    fig, axes = plt.subplots(2, 3, figsize=(7.4, 4.6))
    npaths = set()
    for ax, (d, p, n) in zip(axes.ravel(), CELLS):
        series = {}
        for net in ("single", "double"):
            b = bigf(d, p, fee, net)
            series[net] = b["abs_err"] if b is not None else rl(d, p, fee, net)["deterministic_abs_err"]
            npaths.add(len(series[net]) // 5)
        allx = np.concatenate([series["single"], series["double"]])
        lo, hi = 0.0, np.percentile(allx, 99)
        bins = np.linspace(lo, hi, 61)
        for key in ("single", "double"):
            x = series[key]
            ax.hist(x, bins=bins, histtype='stepfilled', color=COL[key], alpha=0.18, lw=0, density=True)
            ax.hist(x, bins=bins, histtype='step', color=COL[key], lw=1.3, label=NAME[key], density=True)
            ax.axvline(np.mean(x), color=COL[key], lw=1.0, ls='--')
            ax.plot([np.percentile(x, 95)], [0], marker='|', color=COL[key], ms=9, mew=1.5, clip_on=False, zorder=5)
        ax.set_xlim(bins[0], bins[-1]); ax.set_ylim(0)
        ax.set_yticks([]); ax.spines['left'].set_visible(False)
        ax.set_title(n, loc='left'); ax.set_xlabel('tracking error of the path', fontsize=8)
    h, l = axes.ravel()[0].get_legend_handles_labels()
    fig.supxlabel('per-path tracking error = mean over the 250 steps of |hedge portfolio − liability|; dashed: mean over paths (the table\'s number); tick: 95th percentile', fontsize=8, y=-0.02)
    fig.legend(h, l, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"fee {fee}, {'/'.join(f'{k:,}' for k in sorted(npaths))} test paths × 5 strikes (window: up to the 99th percentile)", x=0.01, ha='left', fontsize=9, color=INK)
    fig.tight_layout(); save(fig, name)

fig_track_hist('1%', 'fig_track_hist_fee1')

# ------------------------------------------------------------------ Fig B: tracking error and cumulative cost over the life
def step_cost(z, fees):
    pos = z["trace_position"]; X = z["trace_instrument_prices"][:, :pos.shape[1]]
    prev = np.concatenate([np.zeros_like(pos[:, :1]), pos[:, :-1]], axis=1)
    return (np.abs((pos - prev) * X) * fees[None, None, :]).sum(-1)

def fees_of(fee, I):
    uf, of = (0.001, 0.002) if fee == '0.1%' else (0.01, 0.05)
    return np.array([uf] if I == 1 else [of] * I)

def fig_life(fee, name):
    fig, axes = plt.subplots(2, 6, figsize=(10.5, 3.9))
    for j, (d, p, n) in enumerate(CELLS):
        S = {"single": rl(d, p, fee, "single"), "double": rl(d, p, fee, "double")}
        b = bandf(d, p, fee)
        ax, ax2 = axes[0, j], axes[1, j]
        for key, z in S.items():
            err = np.abs(z["trace_tracking_error"]); N = err.shape[1]; t = np.arange(1, N + 1) / N
            ax.plot(t, np.median(err, 0), color=COL[key], lw=1.3, label=NAME[key])
            ax.fill_between(t, np.percentile(err, 25, 0), np.percentile(err, 75, 0), color=COL[key], alpha=0.15, lw=0)
            c = np.cumsum(step_cost(z, fees_of(fee, z["trace_position"].shape[2])), axis=1)
            ax2.plot(t, c.mean(0), color=COL[key], lw=1.3)
        if b is not None:
            err = np.abs(b["model_band_trace_tracking_error"]); N = err.shape[1]; t = np.arange(1, N + 1) / N
            ax.plot(t, np.median(err, 0), color=COL["band"], lw=1.3, ls='--', label=NAME["band"])
            ax.fill_between(t, np.percentile(err, 25, 0), np.percentile(err, 75, 0), color=COL["band"], alpha=0.12, lw=0)
            zz = {"trace_position": b["model_band_trace_position"], "trace_instrument_prices": b["trace_instrument_prices"]}
            c = np.cumsum(step_cost(zz, fees_of(fee, zz["trace_position"].shape[2])), axis=1)
            ax2.plot(t, c.mean(0), color=COL["band"], lw=1.3, ls='--')
        ax.set_title(n, loc='left', fontsize=8); ax.grid(True, axis='y'); ax2.grid(True, axis='y')
        ax.set_xticks([0, 0.5, 1]); ax2.set_xticks([0, 0.5, 1]); ax.set_xticklabels([]); ax2.set_xlabel('fraction of life', fontsize=8)
        ax.set_yscale('log'); ax.tick_params(labelsize=7); ax2.tick_params(labelsize=7)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    axes[0, 0].set_ylabel('|portfolio − liability|\nmedian, interquartile band', fontsize=8)
    axes[1, 0].set_ylabel('cumulative transaction cost\nmean over paths', fontsize=8)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"fee {fee}, 500 test paths × 5 strikes", x=0.01, ha='left', fontsize=9, color=INK)
    fig.tight_layout(); save(fig, name)

fig_life('1%', 'fig_life_fee1')
fig_life('0.1%', 'fig_life_fee01')

# ------------------------------------------------------------------ Fig C: one illustrative path per BSM product at 1%
def pick_path(z, prod):
    """an env whose spot ends near the strike (digital/call) or that approaches the barrier (doc)"""
    S = z["trace_spot"]; K = z["env_K"][: S.shape[0]]
    m = S[:, -1] / K
    if prod == 'doc':
        score = -np.abs(S.min(1) / K - 0.92)       # closest to 0.92K from above: near miss of the 0.9K barrier
        alive = S.min(1) / K > 0.9
        score[~alive] = -9
    else:
        score = -np.abs(m - 1.0)
    return int(np.argmax(score))

fig, axes = plt.subplots(3, 3, figsize=(10.5, 7.2))
for i, (d, p, n) in enumerate(CELLS[:3]):
    fee = '1%'
    zs, zd = rl(d, p, fee, "single"), rl(d, p, fee, "double"); b = bandf(d, p, fee)
    e = pick_path(zd, p); K = float(zd["env_K"][e]); S = zd["trace_spot"][e]; N = len(S) - 1; t = np.arange(N + 1) / N
    L = zd["trace_liability"][e]
    ax = axes[i, 0]
    ax.plot(t, S / K, color=INK, lw=1.2)
    ax.axhline(1.0, color='#8a8984', lw=0.7, ls=':', label='strike')
    if p == 'doc':
        ax.axhline(0.9, color='#e34948', lw=0.9, ls='--', label='barrier')
    ax.set_title(f"{n}: spot / strike", loc='left'); ax.legend(frameon=False, loc='best', fontsize=7)
    ax = axes[i, 1]
    ax.plot(t, L, color=INK, lw=1.4, label='liability')
    ax.plot(t, zs["trace_portfolio_value"][e], color=COL['single'], lw=1.1, label='single net portfolio')
    ax.plot(t, zd["trace_portfolio_value"][e], color=COL['double'], lw=1.1, label='double net portfolio')
    if b is not None:
        ax.plot(t, b["model_band_trace_portfolio_value"][e], color=COL['band'], lw=1.1, ls='--', label='model band portfolio')
    ax.set_title('liability and hedge portfolios', loc='left')
    if i == 0: ax.legend(frameon=False, fontsize=7)
    ax = axes[i, 2]
    hold = zd["trace_hold"][e] > 0.5
    ax.fill_between(t[:-1], 0, 1, where=hold, color=COL['double'], alpha=0.10, lw=0, transform=ax.get_xaxis_transform(), step='post', label='double net holds')
    I = zd["trace_position"].shape[2]
    inst = ['underlying'] if I == 1 else ['call', 'put']
    for k in range(I):
        ls = '-' if k == 0 else ':'
        ax.plot(t[:-1], zs["trace_position"][e, :, k], color=COL['single'], lw=1.1, ls=ls, label=f'single net, {inst[k]}')
        ax.plot(t[:-1], zd["trace_position"][e, :, k], color=COL['double'], lw=1.1, ls=ls, label=f'double net, {inst[k]}')
        if b is not None:
            ax.plot(t[:-1], b["trace_target"][e, :-1, k], color=COL['target'], lw=0.9, ls=ls, alpha=0.8, label=f'model target, {inst[k]}')
    ax.set_title('positions' + (' (solid: call, dotted: put)' if I == 2 else ''), loc='left')
    for ax in axes[i]:
        ax.grid(True, axis='y'); ax.set_xlim(0, 1)
for ax in axes[-1]:
    ax.set_xlabel('fraction of life')
fig.suptitle("One test path per product, BSM, fee 1% (seed 1 nets); shaded steps: the inaction net holds", x=0.01, ha='left', fontsize=9, color=INK)
hh, ll = axes[2, 2].get_legend_handles_labels()
fig.legend(hh, ll, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.05))
fig.tight_layout(); save(fig, 'fig_sample_paths')

# ------------------------------------------------------------------ Fig D: where the double net holds (moneyness x time)
cmap = LinearSegmentedColormap.from_list('or', ['#efeeea', '#f8c9b4', ORANGE, '#8a2e0c']); cmap.set_bad('white')
fig, axes = plt.subplots(2, 3, figsize=(7.6, 4.6), sharex=True, sharey=True)
for k, fee in enumerate(FEES := ['0.1%', '1%']):
    for j, (d, p, n) in enumerate(CELLS[:3]):
        z = rl(d, p, fee, "double"); ax = axes[k, j]
        S = z["trace_spot"][:, :-1]; K = z["env_K"][: S.shape[0]][:, None]; N = S.shape[1]
        m = (S / K).ravel(); tt = np.broadcast_to(np.arange(N)[None, :] / N, S.shape).ravel(); h = z["trace_hold"].ravel()
        mb = np.linspace(0.85, 1.15, 31); tb = np.linspace(0, 1, 26)
        num, _, _ = np.histogram2d(m, tt, bins=[mb, tb], weights=h); den, _, _ = np.histogram2d(m, tt, bins=[mb, tb])
        with np.errstate(invalid='ignore'):
            rate = np.where(den >= 20, num / den, np.nan)
        im = ax.pcolormesh(tb, mb, rate, cmap=cmap, vmin=0, vmax=1, shading='flat')
        if p == 'doc':
            ax.axhline(0.9, color='#e34948', lw=0.9, ls='--')
        ax.axhline(1.0, color='#8a8984', lw=0.7, ls=':')
        if k == 0:
            ax.set_title(n, loc='left')
        if j == 0:
            ax.set_ylabel(f'fee {fee}\nspot / strike')
for ax in axes[-1]:
    ax.set_xlabel('fraction of life')
cb = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02); cb.set_label('hold probability of the inaction net')
fig.suptitle("Where the double net does not trade (BSM, seed 1, 500 test paths × 5 strikes); white: fewer than 20 observations", x=0.01, ha='left', fontsize=9, color=INK)
save(fig, 'fig_hold_map')

# ------------------------------------------------------------------ Fig E: learned position vs model target (calls, 1%)
if bandf('bsm', 'call', '1%') is not None and bandf('heston', 'call', '1%') is not None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    cm = LinearSegmentedColormap.from_list('bl', ['#cde2fb', '#2a78d6', '#0d366b'])
    for ax, (d, p, n) in zip(axes, [CELLS[0], CELLS[3]]):
        z = rl(d, p, '1%', 'single'); b = bandf(d, p, '1%')
        pos = z["trace_position"][:, :, 0]; tgt = b["trace_target"][:, :-1, 0]; N = pos.shape[1]
        tt = np.broadcast_to(np.arange(N)[None, :] / N, pos.shape)
        sel = np.random.default_rng(0).choice(pos.size, 20000, replace=False)
        sc = ax.scatter(tgt.ravel()[sel], pos.ravel()[sel], c=tt.ravel()[sel], cmap=cm, s=2, alpha=0.35, lw=0, rasterized=True)
        ax.plot([0, 1], [0, 1], color='#8a8984', lw=0.8, ls='--')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect('equal'); ax.grid(True); ax.set_axisbelow(True)
        ax.set_title(f"{n}, fee 1%", loc='left'); ax.set_xlabel('model delta'); ax.set_ylabel('single net position in the underlying')
    cb = fig.colorbar(sc, ax=axes, fraction=0.03, pad=0.02); cb.set_label('fraction of life elapsed')
    save(fig, 'fig_position_vs_delta')
else:
    print("skip fig_position_vs_delta (band paths missing)")

# ------------------------------------------------------------------ Fig E2: position vs delta with OLS fits, both fees (compact)
fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.12, "wspace": 0.08})
for i, fee in enumerate(['0.1%', '1%']):
    for j, (d, name) in enumerate((('bsm', 'BSM call'), ('heston', 'Heston call'))):
        ax = axes[i, j]
        b = bandf(d, 'call', fee)
        D = b["trace_target"][:, :-1, 0]
        fits = {}
        for net in ('single', 'double'):
            z = rl(d, 'call', fee, net); P = z["trace_position"][:, :, 0]; n = min(len(D), len(P))
            x, y = D[:n].ravel(), P[:n].ravel()
            X = np.column_stack([np.ones_like(x), x]); beta = np.linalg.lstsq(X, y, rcond=None)[0]
            fits[net] = (beta, x, y)
        x, y = fits['single'][1], fits['single'][2]
        sel = np.random.default_rng(0).choice(x.size, 6000, replace=False)
        ax.scatter(x[sel], y[sel], s=1.2, color=GREY, alpha=0.35, lw=0, rasterized=True)
        ax.plot([0, 1], [0, 1], color='#8a8984', lw=0.7, ls='--')
        g = np.linspace(0, 1, 2)
        for net, col in (('single', COL['single']), ('double', COL['double'])):
            b0, b1 = fits[net][0]
            ax.plot(g, b0 + b1 * g, color=col, lw=1.4)
        ax.text(0.03, 0.97, f"slope {fits['single'][0][1]:.2f} / {fits['double'][0][1]:.2f}", transform=ax.transAxes, ha='left', va='top', fontsize=6, color=INK)
        ax.set_xlim(0, 1); ax.set_ylim(-0.05, 1); ax.set_aspect('equal'); ax.grid(True); ax.set_axisbelow(True)
        ax.tick_params(labelsize=6, length=2); ax.set_xticks([0, 0.5, 1]); ax.set_yticks([0, 0.5, 1])
        if i == 0: ax.set_title(name, fontsize=7.5, pad=3)
        if j == 0: ax.set_ylabel(f"fee {fee}\nposition", fontsize=7)
        if i == 1: ax.set_xlabel('model delta', fontsize=7)
h = [plt.Line2D([], [], color=COL['single'], label='single policy, OLS'), plt.Line2D([], [], color=COL['double'], label='Actor–Inactor, OLS'),
     plt.Line2D([], [], color='#8a8984', ls='--', label='position = delta')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, fontsize=6, bbox_to_anchor=(0.5, -0.06), handlelength=1.6, columnspacing=1.0)
save(fig, 'fig_delta_ols')
