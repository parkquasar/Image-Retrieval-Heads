"""
Set-partition appendix view: separate table + figure showing
random + overlap_set + attn_only_set + retr_only_set only.
Does NOT merge with the main top-K sweep view.

Usage:
    python eval/build_setpartition_view.py
    python eval/build_setpartition_view.py --out paper/.../setpartition_table.tex

Reads the same JSONL files but on a different SRC_LIST and writes to a
distinct TeX/PDF artifact.
"""
import os, sys, statistics, argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import ATTN_ROOT, MODEL_CONFIGS
# Reuse the loaders from the main builder.
from eval.build_topk_sweep_table import (load_cell, load_baseline,
                                         random_mean, MODEL_ORDER,
                                         RANDOM_SEEDS)

import matplotlib.pyplot as plt

HS_LIST = [2, 3, 5, 10, 20, 50]
K_LIST  = [5, 10, 15, 20, 25, 30]

# Mode is set in main() before SOURCES/STYLE are read.
# 'catpos' => paper ablation: output-zero on o_proj input
#             (default ABL_TAG in build_topk_sweep_table.py; no suffix needed).
# 'qzero'  => LEGACY Q-zero on q_proj output (uses '_qzero' source suffix to
#             switch load_cell's tag; kept for prior-experiment reproduction).
MODE = 'catpos'
INTERVENTION_LABEL = 'output-zero'   # cosmetic, set with MODE


def _src_keys():
    """Return [(display, key)] tuples honoring MODE.

    PAPER: catpos (output-zero) — default, no suffix needed.
    LEGACY: qzero — adds `_qzero` suffix so load_cell loads the Q-zero jsonl.
    """
    suf = '' if MODE == 'catpos' else '_qzero'
    return [
        ('pointer & writer', f'overlap_set_attn{suf}'),
        ('pointer-only',     f'attn_only_set{suf}'),
        ('writer-only',      f'retr_only_set{suf}'),
    ]


def _style():
    suf = '' if MODE == 'catpos' else '_qzero'
    return {
        f'overlap_set_attn{suf}': dict(label='pointer & writer', color='#2ca02c', marker='D'),
        f'attn_only_set{suf}':    dict(label='pointer-only',     color='#d62728', marker='o'),
        f'retr_only_set{suf}':    dict(label='writer-only',      color='#1f77b4', marker='s'),
    }


# These are populated in main() so that argparse can flip MODE first.
SOURCES = _src_keys()
STYLE   = _style()


def emit_model_table(key, label):
    cfg = MODEL_CONFIGS[key]
    md = cfg['model_dir']
    baselines = {hs: load_baseline(md, hs) for hs in HS_LIST}

    rows = []
    base_row = '\\multirow{1}{*}{Baseline} & ' + ' & '.join(
        '\\multicolumn{2}{c}{' + (f'{baselines[hs]:.1f}' if baselines[hs] is not None else 'TBD') + '}'
        for hs in HS_LIST) + ' \\\\'
    rows.append(base_row)
    rows.append('\\midrule')

    # Random row (mean of 3 seeds) per K
    for K in K_LIST:
        cells = []
        for hs in HS_LIST:
            res = random_mean(md, hs, K)
            base = baselines.get(hs)
            if res is None or base is None:
                cells.append('\\multicolumn{2}{c}{TBD}'); continue
            mean, std, n = res
            drop = base - mean
            delta = (f'\\dpos{{0.0}}' if abs(drop) < 0.05 else
                     f'\\drop{{{drop:.1f}}}' if drop > 0 else
                     f'\\dpos{{{-drop:.1f}}}')
            std_part = f'\\,$\\pm$\\,{std:.1f}' if n > 1 else ''
            cells.append(f'{mean:.1f}{std_part} & {delta}')
        rows.append(f'random-{K} & ' + ' & '.join(cells) + ' \\\\')
    rows.append('\\midrule')

    # The 3 set-partition rows
    for label_src, src in SOURCES:
        for K in K_LIST:
            cells = []
            for hs in HS_LIST:
                acc = load_cell(md, hs, K, src)
                base = baselines.get(hs)
                if acc is None or base is None:
                    cells.append('\\multicolumn{2}{c}{TBD}'); continue
                drop = base - acc
                delta = (f'\\dpos{{0.0}}' if abs(drop) < 0.05 else
                         f'\\drop{{{drop:.1f}}}' if drop > 0 else
                         f'\\dpos{{{-drop:.1f}}}')
                cells.append(f'{acc:.1f} & {delta}')
            rows.append(f'{label_src}-{K} & ' + ' & '.join(cells) + ' \\\\')
        rows.append('\\midrule')
    if rows[-1] == '\\midrule':
        rows.pop()

    body = '\n'.join(rows)
    intervention_loc = ('on $o\\_proj$ input' if MODE == 'catpos'
                        else 'on $q\\_proj$ output')
    label_suf = '' if MODE == 'catpos' else '_qzero'
    label_key = key.replace('-', '_')
    col_spec = 'l ' + ' '.join(['r@{}l'] * len(HS_LIST))
    header_cells = ' & '.join(f'\\multicolumn{{2}}{{c}}{{hs={hs}}}' for hs in HS_LIST)
    return f"""\\begin{{table}}[h]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\caption{{Set-partition ablation on {label} (VHs, $n=1{{,}}000$, {INTERVENTION_LABEL} {intervention_loc}). At top-$N$=100 of the union of attn-score-top and retriever-top rankings, the union is split into three disjoint subsets: \\emph{{overlap}} (both), \\emph{{attn-only}} (in attn-top-100, not in retr-top-100), and \\emph{{retr-only}} (in retr-top-100, not in attn-top-100). For overlap we report two intra-subset orderings: max-rank (earliest to enter both top-N) and attn-rank. Each row ablates the top-$K$ heads of the corresponding subset. Random rows ablate $K$ uniformly drawn heads. Cells show accuracy and $\\Delta$ from the in-pipeline baseline.}}
\\label{{tab:setpart_{label_key}{label_suf}}}
\\begin{{tabular}}{{{col_spec}}}
\\toprule
& {header_cells} \\\\
\\addlinespace[2pt]
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def plot_one_model(key, label, out_path):
    cfg = MODEL_CONFIGS[key]
    md = cfg['model_dir']
    fig, axes = plt.subplots(1, len(HS_LIST), figsize=(2.8 * len(HS_LIST), 3.2), sharey=False)

    has_any = False
    for i, hs in enumerate(HS_LIST):
        ax = axes[i]
        baseline = load_baseline(md, hs)
        if baseline is not None:
            ax.axhline(baseline, ls='--', lw=1, color='black',
                       label='baseline' if i == 0 else None, alpha=0.6)

        # Set sources
        for label_src, src in SOURCES:
            xs, ys = ([0], [baseline]) if baseline is not None else ([], [])
            for K in K_LIST:
                acc = load_cell(md, hs, K, src)
                if acc is not None:
                    xs.append(K); ys.append(acc); has_any = True
            if len(xs) > 1:
                style = STYLE[src]
                ax.plot(xs, ys, color=style['color'], marker=style['marker'],
                        markersize=4, lw=1.4,
                        label=style['label'] if i == 0 else None)

        # random (mean ± std)
        xs = [0] if baseline is not None else []
        means = [baseline] if baseline is not None else []
        lo = [baseline] if baseline is not None else []
        hi = [baseline] if baseline is not None else []
        for K in K_LIST:
            res = random_mean(md, hs, K)
            if res is None: continue
            m, sd, n = res
            xs.append(K); means.append(m); lo.append(m-sd); hi.append(m+sd)
            has_any = True
        if len(xs) > 1:
            ax.plot(xs, means, color='#7f7f7f', marker='^', markersize=4, lw=1.4,
                    label='random-$K$ (mean)' if i == 0 else None)
            if any(hi[k]-lo[k] != 0 for k in range(len(xs))):
                ax.fill_between(xs, lo, hi, color='#7f7f7f', alpha=0.15, lw=0)

        ax.set_title(f'hs={hs}')
        ax.set_xlabel('$K$ (heads ablated)')
        if i == 0:
            ax.set_ylabel('accuracy (%)')
        ax.set_xticks([0] + K_LIST)
        ax.grid(True, alpha=0.25, lw=0.5)

    if not has_any:
        plt.close(fig); return False

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    # (figure-level suptitle removed; described in paper text instead)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}', file=sys.stderr)
    return True


def plot_avg_across_models(out_path):
    """Single figure: average drop (baseline - acc) across 4 models, 5 panels per hs.
    Each model contributes its own (baseline - acc); average across models.
    Random shows mean ± std across models (and across the 3 random seeds).
    """
    fig, axes = plt.subplots(1, len(HS_LIST), figsize=(2.8 * len(HS_LIST), 3.2), sharey=False)

    for i, hs in enumerate(HS_LIST):
        ax = axes[i]
        # Per-source averaged drop curves
        for label_src, src in SOURCES:
            xs, mean_drops, lo, hi = [0], [0.0], [0.0], [0.0]
            for K in K_LIST:
                drops = []
                for key, _ in MODEL_ORDER:
                    md = MODEL_CONFIGS[key]['model_dir']
                    base = load_baseline(md, hs)
                    acc = load_cell(md, hs, K, src)
                    if base is None or acc is None: continue
                    drops.append(base - acc)
                if not drops: continue
                m = sum(drops) / len(drops)
                # std across models
                if len(drops) > 1:
                    var = sum((d - m)**2 for d in drops) / (len(drops) - 1)
                    sd = var ** 0.5
                else:
                    sd = 0.0
                xs.append(K); mean_drops.append(m); lo.append(m-sd); hi.append(m+sd)
            if len(xs) > 1:
                style = STYLE[src]
                ax.plot(xs, mean_drops, color=style['color'], marker=style['marker'],
                        markersize=4, lw=1.6, ls=style.get('ls', '-'),
                        label=style['label'] if i == 0 else None)
                ax.fill_between(xs, lo, hi, color=style['color'], alpha=0.10, lw=0)

        # Random baseline drop
        xs, mean_drops, lo, hi = [0], [0.0], [0.0], [0.0]
        for K in K_LIST:
            drops = []
            for key, _ in MODEL_ORDER:
                md = MODEL_CONFIGS[key]['model_dir']
                base = load_baseline(md, hs)
                res = random_mean(md, hs, K)
                if base is None or res is None: continue
                drops.append(base - res[0])
            if not drops: continue
            m = sum(drops) / len(drops)
            if len(drops) > 1:
                var = sum((d - m)**2 for d in drops) / (len(drops) - 1)
                sd = var ** 0.5
            else:
                sd = 0.0
            xs.append(K); mean_drops.append(m); lo.append(m-sd); hi.append(m+sd)
        if len(xs) > 1:
            ax.plot(xs, mean_drops, color='#7f7f7f', marker='^', markersize=4, lw=1.6,
                    label='random' if i == 0 else None)
            ax.fill_between(xs, lo, hi, color='#7f7f7f', alpha=0.10, lw=0)

        ax.axhline(0, ls='--', lw=0.8, color='black', alpha=0.5,
                   label='baseline (no drop)' if i == 0 else None)
        ax.set_title(f'hs={hs}')
        ax.set_xlabel('$K$ (heads ablated)')
        if i == 0:
            ax.set_ylabel('average drop (pp)')
        ax.set_xticks([0] + K_LIST)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.invert_yaxis()  # drops go DOWN as K increases — flip so larger drops = lower

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    # (figure-level suptitle removed; described in paper text instead)
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}', file=sys.stderr)


def plot_avg_across_hs(out_path):
    """Single figure: 4 panels (one per model). Each panel shows the average
    drop across the 5 hs values per K (so each curve is hs-averaged within one
    model). Bands are ±1std across hs values within that model.
    """
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.2), sharey=False)

    for i, (key, label) in enumerate(MODEL_ORDER):
        ax = axes[i]
        md = MODEL_CONFIGS[key]['model_dir']
        for label_src, src in SOURCES:
            xs, mean_drops, lo, hi = [0], [0.0], [0.0], [0.0]
            for K in K_LIST:
                drops = []
                for hs in HS_LIST:
                    base = load_baseline(md, hs)
                    acc = load_cell(md, hs, K, src)
                    if base is None or acc is None: continue
                    drops.append(base - acc)
                if not drops: continue
                m = sum(drops) / len(drops)
                if len(drops) > 1:
                    sd = (sum((d-m)**2 for d in drops) / (len(drops)-1)) ** 0.5
                else:
                    sd = 0.0
                xs.append(K); mean_drops.append(m); lo.append(m-sd); hi.append(m+sd)
            if len(xs) > 1:
                style = STYLE[src]
                ax.plot(xs, mean_drops, color=style['color'], marker=style['marker'],
                        markersize=4, lw=1.6, ls=style.get('ls', '-'),
                        label=style['label'] if i == 0 else None)
                ax.fill_between(xs, lo, hi, color=style['color'], alpha=0.10, lw=0)

        # random
        xs, mean_drops, lo, hi = [0], [0.0], [0.0], [0.0]
        for K in K_LIST:
            drops = []
            for hs in HS_LIST:
                base = load_baseline(md, hs)
                res = random_mean(md, hs, K)
                if base is None or res is None: continue
                drops.append(base - res[0])
            if not drops: continue
            m = sum(drops) / len(drops)
            if len(drops) > 1:
                sd = (sum((d-m)**2 for d in drops) / (len(drops)-1)) ** 0.5
            else:
                sd = 0.0
            xs.append(K); mean_drops.append(m); lo.append(m-sd); hi.append(m+sd)
        if len(xs) > 1:
            ax.plot(xs, mean_drops, color='#7f7f7f', marker='^', markersize=4, lw=1.6,
                    label='random' if i == 0 else None)
            ax.fill_between(xs, lo, hi, color='#7f7f7f', alpha=0.10, lw=0)

        ax.axhline(0, ls='--', lw=0.8, color='black', alpha=0.5,
                   label='baseline (no drop)' if i == 0 else None)
        ax.set_title(label)
        ax.set_xlabel('$K$ (heads ablated)')
        if i == 0:
            ax.set_ylabel('average drop across hs (pp)')
        ax.set_xticks([0] + K_LIST)
        ax.grid(True, alpha=0.25, lw=0.5)
        ax.invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.05))
    # (figure-level suptitle removed; described in paper text instead)
    fig.tight_layout(rect=[0, 0.04, 1, 1.0])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}', file=sys.stderr)


def plot_all_models_grid(out_path):
    """Per-model dose-response grid in the same form as the main-paper avg
    Figure: rows = LVLMs, cols = haystack sizes; Y-axis = accuracy drop in
    percentage points relative to the unablated baseline; Y-axis inverted so
    larger drops go DOWN visually.
    """
    nrows, ncols = len(MODEL_ORDER), len(HS_LIST)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(14, 2.0 * nrows),
                              sharey='row')
    import numpy as _np
    axes = _np.atleast_2d(axes)
    if axes.shape == (1, ncols) and nrows == ncols:
        axes = axes.reshape(nrows, ncols)

    for r, (key, label) in enumerate(MODEL_ORDER):
        md = MODEL_CONFIGS[key]['model_dir']
        for c, hs in enumerate(HS_LIST):
            ax = axes[r, c]
            baseline = load_baseline(md, hs)

            # Per-source drop curves
            for label_src, src in SOURCES:
                xs, drops = [0], [0.0]
                for K in K_LIST:
                    acc = load_cell(md, hs, K, src)
                    if baseline is None or acc is None: continue
                    xs.append(K); drops.append(baseline - acc)
                if len(xs) > 1:
                    style = STYLE[src]
                    ax.plot(xs, drops, color=style['color'], marker=style['marker'],
                            markersize=3.5, lw=1.3,
                            label=style['label'] if (r == 0 and c == 0) else None)

            # Random-control drop
            xs, drops = [0], [0.0]
            for K in K_LIST:
                res = random_mean(md, hs, K)
                if baseline is None or res is None: continue
                xs.append(K); drops.append(baseline - res[0])
            if len(xs) > 1:
                ax.plot(xs, drops, color='#7f7f7f', marker='^', markersize=3.5,
                        lw=1.3,
                        label='random' if (r == 0 and c == 0) else None)

            ax.axhline(0, ls='--', lw=0.8, color='black', alpha=0.5,
                       label='baseline (no drop)' if (r == 0 and c == 0) else None)
            if r == 0:
                ax.set_title(f'hs={hs}')
            if c == 0:
                ax.set_ylabel(f'{label}\ndrop (pp)')
            if r == nrows - 1:
                ax.set_xlabel('$K$ (heads ablated)')
            ax.set_xticks([0] + K_LIST)
            ax.grid(True, alpha=0.25, lw=0.5)
            ax.invert_yaxis()  # larger drops go DOWN, matching main paper figure

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=len(handles),
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.03, 1, 1.0])
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}', file=sys.stderr)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['catpos', 'qzero'], default='catpos',
                        help='catpos = output-zero (o_proj input); qzero = Q-zero (q_proj output).')
    parser.add_argument('--out', default=None,
                        help='Default: setpartition_auto.tex (catpos) or setpartition_auto_qzero.tex (qzero).')
    parser.add_argument('--figs-dir', default='paper/69df6529e7624f32262d432f/figs')
    args = parser.parse_args()

    # Set MODE early, then refresh SOURCES/STYLE so the rest of the module sees them.
    global MODE, INTERVENTION_LABEL, SOURCES, STYLE
    MODE = args.mode
    INTERVENTION_LABEL = 'output-zero' if MODE == 'catpos' else 'Q-zero'
    SOURCES = _src_keys()
    STYLE   = _style()

    if args.out is None:
        args.out = ('paper/69df6529e7624f32262d432f/setpartition_auto.tex'
                    if MODE == 'catpos'
                    else 'paper/69df6529e7624f32262d432f/setpartition_auto_qzero.tex')
    fig_suffix = '' if MODE == 'catpos' else '_qzero'

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Nimbus Roman', 'Times New Roman', 'Times',
                        'Liberation Serif', 'DejaVu Serif'],
        'mathtext.fontset': 'stix',
        # source figsize ~14" wide, rendered at \linewidth=5.5".
        # Sizes chosen so labels are legible without squeezing the data plots.
        'font.size': 12,
        'axes.titlesize': 13,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
    })

    # table
    chunks = [f'% Auto-generated by eval/build_setpartition_view.py --mode {MODE} — do not edit by hand.\n']
    for key, label in MODEL_ORDER:
        chunks.append(f'% --- {label} ---')
        chunks.append(emit_model_table(key, label))
    text = '\n'.join(chunks)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w') as f:
        f.write(text)
    print(f'Wrote {args.out}', file=sys.stderr)

    # per-model figures (kept for reference; no longer used in the paper)
    os.makedirs(args.figs_dir, exist_ok=True)
    for key, label in MODEL_ORDER:
        out = os.path.join(args.figs_dir, f'setpartition_{key.replace("-", "_")}{fig_suffix}.pdf')
        plot_one_model(key, label, out)

    # combined per-model grid (used in the appendix to replace the four 1x5 figures)
    grid_out = os.path.join(args.figs_dir, f'setpartition_per_model_grid{fig_suffix}.pdf')
    plot_all_models_grid(grid_out)

    # averaged figures (main paper)
    avg_models_out = os.path.join(args.figs_dir, f'setpartition_avg4models{fig_suffix}.pdf')
    plot_avg_across_models(avg_models_out)
    avg_hs_out = os.path.join(args.figs_dir, f'setpartition_avgHs_perModel{fig_suffix}.pdf')
    plot_avg_across_hs(avg_hs_out)


if __name__ == '__main__':
    main()
