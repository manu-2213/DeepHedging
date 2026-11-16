"""Paper figures from the collected run tables (experiments/results/collected/runs.json and curves.json, written by
experiments/analysis/collect_runs.py). Output: paper_notes/figures/*.pdf and *.png (PAPER=1: paper_notes/figures/paper).
usage (from anywhere): python experiments/figures/make_figs.py        FIG_OUT=<folder> overrides the output folder"""
import json, re, statistics as st, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CUR = f"{REPO}/experiments/results/collected"
OUT = os.environ.get("FIG_OUT", f"{REPO}/paper_notes/figures")
os.makedirs(OUT, exist_ok=True)
R = json.load(open(f"{CUR}/runs.json"))
C = json.load(open(f"{CUR}/curves.json"))

# palette (dataviz reference instance, slots 1-3 + secondary ink)
BLUE, ORANGE, AQUA, INK, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e", "#b5b4ae"
COL = {"single": BLUE, "double": ORANGE, "band": AQUA, "nohedge": INK}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9.5, "axes.labelsize": 9, "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "axes.labelcolor": "#0b0b0b", "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "figure.dpi": 150, "savefig.dpi": 300,
                     "font.family": "sans-serif", "pdf.fonttype": 42})

MAIN = {('bsm','call','0.1%'):('1e-4','3e-5'),('bsm','call','1%'):('1e-4','3e-5'),('bsm','doc','0.1%'):('1e-3','3e-3'),('bsm','doc','1%'):('3e-4','1e-4'),
        ('bsm','con','0.1%'):('3e-3','1e-2'),('bsm','con','1%'):('1e-3','3e-5'),('heston','call','0.1%'):('3e-3','1e-3'),('heston','call','1%'):('3e-3','1e-3'),
        ('heston','doc','0.1%'):('3e-3','1e-3'),('heston','doc','1%'):('3e-4','3e-3'),('heston','con','0.1%'):('3e-3','1e-3'),('heston','con','1%'):('3e-3','1e-3')}
CELLS = [('bsm','call','BSM call'),('bsm','doc','BSM down-and-out call'),('bsm','con','BSM cash-or-nothing'),
         ('heston','call','Heston call'),('heston','doc','Heston down-and-out call'),('heston','con','Heston cash-or-nothing')]
FEES = ['0.1%', '1%']
T = [r for r in R if r.get('table') and r['run'] != 'baseline']
B = [r for r in R if r['group'] == 'baseline']

def fromlr(r):
    m = re.search(r'_fromlr(\d)e(\d)_s\d', r['run']); return f"{m.group(1)}e-{m.group(2)}" if m else None

def selected(dyn, prod, fee):
    """Per-seed selection: (single row, double row) per seed."""
    r1, r2 = MAIN[(dyn, prod, fee)]
    cr = [r for r in T if r['dynamics'] == dyn and r['product'] == prod and r['fee'] == fee and r['status'] == 'done']
    out = []
    for seed in range(1, 6):
        sg = [r for r in cr if r['net'] == 'single' and r['seed'] == seed and r['actor_lr'] in (r1, r2)]
        best = min(sg, key=lambda r: r['best_val_track'])
        dd = [r for r in cr if r['net'] == 'double' and r['seed'] == seed and ((fromlr(r) == best['actor_lr']) if best['actor_lr'] != r1 else not fromlr(r))]
        bd = min(dd, key=lambda r: r['best_val_track'])
        out.append((best, bd))
    return out

def band(dyn, prod, fee):
    b = [r for r in B if r['dynamics'] == dyn and r['product'] == prod and r['fee'] == fee]
    ww = next(r for r in b if 'Whalley' in r['label']); nh = next(r for r in b if 'no hedge' in r['label'].lower())
    return ww, nh

SEL = {(d, p, f): selected(d, p, f) for d, p, _ in CELLS for f in FEES}
BND = {(d, p, f): band(d, p, f) for d, p, _ in CELLS for f in FEES}

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

# ---------------------------------------------------------------- Fig 1: main results, tracking error per cell
def fig_main(metric, name, xlabel):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)
    ylab = [n for _, _, n in CELLS]
    for ax, fee in zip(axes, FEES):
        for i, (d, p, n) in enumerate(CELLS):
            y = len(CELLS) - 1 - i
            s = [r[metric] for r, _ in SEL[(d, p, fee)]]; dbl = [b[metric] for _, b in SEL[(d, p, fee)]]
            ww, nh = BND[(d, p, fee)]
            ax.plot([min(s), max(s)], [y + 0.17] * 2, color=COL['single'], lw=1.2, solid_capstyle='round')
            ax.plot([min(dbl), max(dbl)], [y - 0.17] * 2, color=COL['double'], lw=1.2, solid_capstyle='round')
            ax.plot(st.mean(s), y + 0.17, 'o', ms=5.5, color=COL['single'], mec='white', mew=0.8, zorder=3)
            ax.plot(st.mean(dbl), y - 0.17, 'o', ms=5.5, color=COL['double'], mec='white', mew=0.8, zorder=3)
            ax.plot(ww[metric], y, 'D', ms=4.5, color=COL['band'], mec='white', mew=0.6, zorder=3)
            if metric == 'test_track':
                ax.plot(nh[metric], y, '|', ms=8, color=COL['nohedge'], mew=1.2, zorder=2)
        ax.set_xscale('log'); ax.grid(True, axis='x'); ax.set_axisbelow(True)
        ax.set_yticks(range(len(CELLS))); ax.set_yticklabels(ylab[::-1]); ax.tick_params(axis='y', length=0)
        ax.set_title(f"fee {fee}", loc='left')
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in '235' else ""))
        ax.spines['left'].set_visible(False)
    fig.supxlabel(xlabel, fontsize=9, y=-0.01)
    h = [plt.Line2D([], [], marker='o', color=COL['single'], ls='-', ms=5, label='single net (5 seeds: mean, range)'),
         plt.Line2D([], [], marker='o', color=COL['double'], ls='-', ms=5, label='double net (5 seeds: mean, range)'),
         plt.Line2D([], [], marker='D', color=COL['band'], ls='', ms=4.5, label='model band (Whalley–Wilmott-style)')]
    if metric == 'test_track':
        h.append(plt.Line2D([], [], marker='|', color=COL['nohedge'], ls='', ms=8, label='no hedge'))
    fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.16))
    fig.tight_layout(); save(fig, name)

fig_main('test_track', 'fig_main_tracking', 'test tracking error: mean |portfolio value − liability| per step (log scale)')
fig_main('test_cost', 'fig_main_cost', 'test transaction cost per path (log scale)')

# ---------------------------------------------------------------- Fig 2: cost vs tracking, single -> double arrows
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))
for ax, fee in zip(axes, FEES):
    for i, (d, p, n) in enumerate(CELLS):
        sm = (st.mean([r['test_track'] for r, _ in SEL[(d, p, fee)]]), st.mean([r['test_cost'] for r, _ in SEL[(d, p, fee)]]))
        dm = (st.mean([b['test_track'] for _, b in SEL[(d, p, fee)]]), st.mean([b['test_cost'] for _, b in SEL[(d, p, fee)]]))
        ww, _ = BND[(d, p, fee)]
        ax.annotate('', xy=dm, xytext=sm, arrowprops=dict(arrowstyle='-|>', color=INK, lw=0.9, shrinkA=3, shrinkB=3, mutation_scale=8))
        ax.plot(*sm, 'o', color=COL['single'], ms=5.5, mec='white', mew=0.8, zorder=3)
        ax.plot(*dm, 'o', color=COL['double'], ms=5.5, mec='white', mew=0.8, zorder=3)
        ax.plot(ww['test_track'], ww['test_cost'], 'D', color=COL['band'], ms=4.5, mec='white', mew=0.6, zorder=3)
        off = {(0, '1%'): (-6, 8), (3, '1%'): (6, -6), (1, '0.1%'): (6, -4), (4, '0.1%'): (6, -6)}.get((i, fee), (4, -3))
        ax.annotate(n, dm, xytext=off, textcoords='offset points', fontsize=6.5, color=INK, ha='left' if off[0] > 0 else 'right', va='top' if off[1] < 0 else 'bottom')
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True); ax.set_axisbelow(True)
    ax.set_title(f"fee {fee}", loc='left'); ax.set_xlabel('test tracking error (log)'); ax.set_ylabel('test transaction cost (log)')
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        a.set_minor_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}" if str(v)[0] in '235' else ""))
h = [plt.Line2D([], [], marker='o', color=COL['single'], ls='', ms=5, label='single net'),
     plt.Line2D([], [], marker='o', color=COL['double'], ls='', ms=5, label='double net (arrow: single → double)'),
     plt.Line2D([], [], marker='D', color=COL['band'], ls='', ms=4.5, label='model band')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); save(fig, 'fig_cost_vs_tracking')

# ---------------------------------------------------------------- Fig 3: double/single ratios per seed
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
for ax, metric, ttl in zip(axes, ['test_track', 'test_cost'], ['tracking error', 'transaction cost']):
    for i, (d, p, n) in enumerate(CELLS):
        y = len(CELLS) - 1 - i
        for k, fee in enumerate(FEES):
            ratios = [b[metric] / r[metric] for r, b in SEL[(d, p, fee)]]
            yy = y + (0.17 if k == 0 else -0.17)
            col = GREY if k == 0 else INK
            ax.plot(ratios, [yy] * 5, 'o', ms=3.5, color=col, alpha=0.55, mec='none')
            ax.plot(st.mean(ratios), yy, 'o', ms=6, color=col, mec='white', mew=0.8, zorder=3)
    ax.axvline(1, color='#8a8984', lw=0.8, ls='--')
    ax.set_yticks(range(len(CELLS))); ax.set_yticklabels([n for _, _, n in CELLS][::-1]); ax.tick_params(axis='y', length=0)
    ax.grid(True, axis='x'); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)
    ax.set_title(f"double / single: {ttl}", loc='left'); ax.set_xlabel('ratio on test (per seed, and mean)')
h = [plt.Line2D([], [], marker='o', color=GREY, ls='', ms=6, label='fee 0.1%'), plt.Line2D([], [], marker='o', color=INK, ls='', ms=6, label='fee 1%')]
fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(); save(fig, 'fig_double_over_single')

# ---------------------------------------------------------------- Fig 4: hold rate of the double net
fig, ax = plt.subplots(figsize=(3.6, 3.2))
for i, (d, p, n) in enumerate(CELLS):
    y = len(CELLS) - 1 - i
    for k, fee in enumerate(FEES):
        hs = [b.get('test_hold') if b.get('test_hold') is not None else b.get('hold') for _, b in SEL[(d, p, fee)]]
        hs = [h for h in hs if h is not None]
        col = GREY if k == 0 else INK
        ax.plot(hs, [y] * len(hs), 'o', ms=3.5, color=col, alpha=0.55, mec='none')
        ax.plot(st.mean(hs), y, 'o', ms=6, color=col, mec='white', mew=0.8, zorder=3)
ax.set_yticks(range(len(CELLS))); ax.set_yticklabels([n for _, _, n in CELLS][::-1]); ax.tick_params(axis='y', length=0)
ax.set_xlim(-0.02, 1.02); ax.grid(True, axis='x'); ax.set_axisbelow(True); ax.spines['left'].set_visible(False)
ax.set_xlabel('hold rate of the inaction net on test')
fig.legend(handles=h, loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.6, -0.04))
fig.tight_layout(); save(fig, 'fig_hold_rate')

# ---------------------------------------------------------------- Fig 5: learning curves, 12 small multiples
fig, axes = plt.subplots(3, 4, figsize=(7.4, 5.6), sharex=False)
byrow = {}
for c in C:
    byrow.setdefault((c['method'], c['fee'], c['seed'], c['dynamics'], c['product']), []).append(c)
for i, (d, p, n) in enumerate(CELLS):
    for k, fee in enumerate(FEES):
        ax = axes[i // 2, (i % 2) * 2 + k]
        ww, _ = BND[(d, p, fee)]
        ymin, ymax = 1e9, 0
        for seed, (s, b) in enumerate(SEL[(d, p, fee)], 1):
            for r, col in ((s, COL['single']), (b, COL['double'])):
                pts = sorted(byrow.get((r['label'], fee, seed, d, p), []), key=lambda q: q['iteration'])
                if r['net'] == 'double':
                    j = next((ix for ix, q in enumerate(pts) if q['hold'] > 0), 0)
                    pts = pts[j:]
                if not pts:
                    continue
                x = [q['iteration'] for q in pts]; y = [q['best_tracking_error'] for q in pts]
                ax.plot(x, y, color=col, lw=0.9, alpha=0.85)
                ymin = min(ymin, min(y)); ymax = max(ymax, y[0])
        if ww.get('best_val_track'):
            ax.axhline(ww['best_val_track'], color=COL['band'], lw=0.9, ls='--')
            ymin = min(ymin, ww['best_val_track'])
        ax.set_yscale('log'); ax.set_ylim(ymin * 0.85, min(ymax, ymin * 12) * 1.1)
        ax.set_title(f"{n} · {fee}", loc='left', fontsize=7.5)
        ax.grid(True, axis='y'); ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if v else "0"))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.tick_params(labelsize=6.5)
for ax in axes[-1]:
    ax.set_xlabel('training iteration', fontsize=7.5)
for ax in axes[:, 0]:
    ax.set_ylabel('best validation tracking error', fontsize=7)
h = [plt.Line2D([], [], color=COL['single'], label='single net, 5 seeds'), plt.Line2D([], [], color=COL['double'], label='double net (joint phase), 5 seeds'),
     plt.Line2D([], [], color=COL['band'], ls='--', label='model band on validation')]
fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(); save(fig, 'fig_learning_curves')

# ---------------------------------------------------------------- Fig 6: per-seed rate selection (validation of rate 1 vs rate 2)
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
for ax, fee in zip(axes, FEES):
    for i, (d, p, n) in enumerate(CELLS):
        r1, r2 = MAIN[(d, p, fee)]
        cr = [r for r in T if r['dynamics'] == d and r['product'] == p and r['fee'] == fee and r['net'] == 'single' and r['status'] == 'done']
        xs, ys = [], []
        for seed in range(1, 6):
            a = [r for r in cr if r['seed'] == seed and r['actor_lr'] == r1]; b = [r for r in cr if r['seed'] == seed and r['actor_lr'] == r2]
            if a and b:
                xs.append(a[0]['best_val_track']); ys.append(b[0]['best_val_track'])
        ax.plot(xs, ys, 'o', ms=4.5, mec='white', mew=0.5, label=f"{n} ({r1} vs {r2})", color=plt.cm.tab10(i) if False else [BLUE, ORANGE, AQUA, '#eda100', '#6250d6', '#e87ba4'][i])
    lim = [3, 200]
    ax.plot(lim, lim, color='#8a8984', lw=0.8, ls='--'); ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xscale('log'); ax.set_yscale('log'); ax.grid(True); ax.set_axisbelow(True)
    ax.set_title(f"fee {fee}", loc='left'); ax.set_xlabel('validation tracking error, first rate'); ax.set_ylabel('validation tracking error, second rate')
    for a in (ax.xaxis, ax.yaxis):
        a.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.legend(frameon=False, fontsize=6.5, loc='upper left')
fig.tight_layout(); save(fig, 'fig_rate_selection')

# ---------------------------------------------------------------- Fig 7: seed-1 tuning curves: actor rate sweep, two cells
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
for ax, (d, p, n, fee) in zip(axes, [('bsm', 'call', 'BSM call', '1%'), ('heston', 'con', 'Heston cash-or-nothing', '1%')]):
    cr = sorted([r for r in T if r['dynamics'] == d and r['product'] == p and r['fee'] == fee and r['net'] == 'single' and r['seed'] == 1 and r['run'].startswith('main/')],
                key=lambda r: float(r['actor_lr']))
    if (d, p) == ('bsm', 'call'):
        cr = sorted([r for r in T if r['dynamics'] == d and r['product'] == p and r['fee'] == fee and r['net'] == 'single' and r['seed'] == 1], key=lambda r: float(r['actor_lr']))
    shades = ['#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b', '#eb6834', '#e34948']
    for j, r in enumerate(cr):
        pts = sorted(byrow.get((r['label'], fee, 1, d, p), []), key=lambda q: q['iteration'])
        if not pts:
            continue
        ax.plot([q['iteration'] for q in pts], [q['best_tracking_error'] for q in pts], color=shades[min(j, len(shades) - 1)], lw=1.2, label=f"rate {r['actor_lr']}")
    ax.set_yscale('log'); ax.grid(True, axis='y'); ax.set_axisbelow(True)
    ax.set_title(f"{n} · {fee}, seed 1", loc='left'); ax.set_xlabel('training iteration'); ax.set_ylabel('best validation tracking error')
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:g}k" if v else "0"))
    ax.set_yticks([20, 30, 50, 100, 200]); ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v:g}")); ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.legend(frameon=False, fontsize=7)
fig.tight_layout(); save(fig, 'fig_rate_sweep')

# ---------------------------------------------------------------- Fig 1b: per-cell panels with their own linear axis
def fig_main_zoom(metric, name, xlabel):
    """3 rows (products) x 4 columns (BSM 0.1%, BSM 1%, Heston 0.1%, Heston 1%), each panel with its own linear axis."""
    PRODUCTS = [('call', 'European call'), ('doc', 'down-and-out call'), ('con', 'cash-or-nothing')]
    COLS = [('bsm', '0.1%'), ('bsm', '1%'), ('heston', '0.1%'), ('heston', '1%')]
    fig, axes = plt.subplots(3, 4, figsize=(7.4, 3.0), sharey=True, gridspec_kw={"hspace": 0.7, "wspace": 0.12, "left": 0.12})
    for k, (d, fee) in enumerate(COLS):
        axes[0, k].annotate(f"{'BSM' if d == 'bsm' else 'Heston'} \u00b7 fee {fee}", xy=(0.5, 1.45), xycoords='axes fraction',
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        for i, (p, pname) in enumerate(PRODUCTS):
            ax = axes[i, k]
            s = [r[metric] for r, _ in SEL[(d, p, fee)]]; dbl = [b[metric] for _, b in SEL[(d, p, fee)]]
            ww, nh = BND[(d, p, fee)]
            vals = s + dbl
            band_in = min(vals) * 0.6 <= ww[metric] <= max(vals) * 1.5
            if band_in:
                vals = vals + [ww[metric]]
            lo, hi = min(vals), max(vals); pad = 0.12 * (hi - lo) + 1e-9
            for j, (xs, col) in enumerate(((s, COL['single']), (dbl, COL['double']))):
                y = 0.5 - j
                ax.plot(xs, [y] * len(xs), 'o', ms=2.6, color=col, alpha=0.5, mec='none')
                ax.plot(st.mean(xs), y, 'o', ms=5.5, color=col, mec='white', mew=0.7, zorder=3)
            if band_in:
                ax.axvline(ww[metric], color=COL['band'], lw=1.2, ls='--', zorder=2)
            else:
                side = ww[metric] > hi
                ax.annotate(f"band {ww[metric]:.3g} " + ("\u25b8" if side else ""), xy=(1.0 if side else 0.0, 0.5), xycoords='axes fraction',
                            ha='right' if side else 'left', va='center', fontsize=5.8, color=COL['band'],
                            xytext=(-2 if side else 2, 0), textcoords='offset points')
            ax.set_xlim(lo - pad, hi + pad); ax.set_ylim(-1.1, 1.1); ax.set_yticks([])
            ax.spines['left'].set_visible(False); ax.grid(True, axis='x'); ax.set_axisbelow(True)
            ax.tick_params(labelsize=6, length=2, pad=1)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            if k == 0:
                ax.set_ylabel(pname, rotation=0, ha='right', va='center', fontsize=7, labelpad=6)
            if metric == 'test_track':
                ax.text(1.0, 1.02, f"no hedge {nh[metric]:.0f}", transform=ax.transAxes, ha='right', va='bottom', fontsize=5.8, color=INK)
    h = [plt.Line2D([], [], marker='o', color=COL['single'], ls='', ms=5, label='single policy (seeds, mean)'),
         plt.Line2D([], [], marker='o', color=COL['double'], ls='', ms=5, label='Actor\u2013Inactor (seeds, mean)'),
         plt.Line2D([], [], color=COL['band'], ls='--', label='model band')]
    fig.legend(handles=h, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.1), fontsize=7)
    fig.supxlabel(xlabel, fontsize=7.5, y=-0.02)
    save(fig, name)

fig_main_zoom('test_track', 'fig_main_tracking_zoom', 'test tracking error (each panel has its own linear axis)')
fig_main_zoom('test_cost', 'fig_main_cost_zoom', 'test transaction cost per path (each panel has its own linear axis)')
