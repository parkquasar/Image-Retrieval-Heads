"""
Read ablation JSONL files for the top-K + random-seed sweep (Phase 2) and emit
LaTeX tables for the paper appendix. One table per model, rows = (source, K),
cols = hs values. Cells show accuracy and Δ from same-pipeline baseline; for
random source, mean ± std across the seeds we have.

Usage:
  python eval/build_topk_sweep_table.py
  python eval/build_topk_sweep_table.py --markdown   # also emit markdown table
  python eval/build_topk_sweep_table.py --out paper/.../appendix_topk.tex

Reads from $ATTN_ROOT/{model_dir}/ablation/*.jsonl.
"""
import os, sys, json, csv, glob, re, argparse, statistics

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_utils import MODEL_CONFIGS
from utils.model_utils import ATTN_ROOT  # $ATTN_ROOT/{model_dir}/ablation/*.jsonl


HS_LIST = [2, 3, 5, 10, 20, 50]
K_LIST = [5, 10, 20, 30, 40, 50]
RANDOM_SEEDS = [42, 1, 7]
PHASE = 'cattok'
# Default ablation tag = paper output-zero (`--ablation-mode catpos
# --ablation-pos cattok_onwards` in eval_vhs_head_ablation.py).
# `load_cell` accepts `_outputzero` source suffix as a no-op alias for
# this default, and `_qzero` suffix to switch to the legacy Q-zero tag
# (kept for prior-experiment reproduction).
ABL_TAG = 'catpos_cattok_onwards'

MODEL_ORDER = [
    ('internvl-8b', 'InternVL3.5-8B'),
    ('internvl-4b', 'InternVL3.5-4B'),
    ('qwen3vl-8b',  'Qwen3-VL-8B'),
    ('qwen3vl-4b',  'Qwen3-VL-4B'),
]


def jsonl_acc(path):
    """Return (correct, total) by counting JSONL rows."""
    if not os.path.exists(path):
        return None
    correct = total = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            total += 1
            if r.get('correct'):
                correct += 1
    if total == 0:
        return None
    return correct, total


def load_cell(model_dir, hs, k, source, seed=None):
    """Return accuracy in % for one ablation cell, or None if missing/incomplete.

    Default tag is the paper output-zero ablation
    (`catpos_cattok_onwards`). Source can carry one of two suffixes that
    override the tag:
      - `_outputzero` (alias for the default; kept for backward
         compatibility with older calls)
      - `_qzero` (legacy Q-zero variant, NOT used in the paper)

    For sources that are intervention-independent (baseline, random,
    random_seed*), the on-disk filenames use the legacy `qzero_catpos_cattok_onwards`
    tag — those files were generated alongside the legacy ablation runs
    and were not duplicated under the catpos tag because the underlying
    forward (no head zeroed for `baseline`, random heads zeroed for
    `random*`) is the same regardless of which paper-vs-legacy
    intervention is being measured. We fall back to that legacy tag if
    the catpos-tagged file is missing.
    """
    base = os.path.join(ATTN_ROOT, model_dir, 'ablation')
    abl_tag = ABL_TAG
    src = source
    if src.endswith('_outputzero'):
        src = src[:-len('_outputzero')]
        abl_tag = 'catpos_cattok_onwards'
    elif src.endswith('_qzero'):
        src = src[:-len('_qzero')]
        abl_tag = 'qzero_catpos_cattok_onwards'
    if seed is not None and seed != 42:
        src_tag = f'random_seed{seed}'
    else:
        src_tag = src
    fn = f'hs{hs}_top{k}_{PHASE}_{src_tag}_{abl_tag}.jsonl'
    res = jsonl_acc(os.path.join(base, fn))
    if res is None and abl_tag == 'catpos_cattok_onwards' and src_tag == 'baseline':
        # Only the unablated baseline may fall back to the legacy tag. The random fallback was removed:
        # random controls under qzero and catpos differ by up to 4.2 pp, so they must not be mixed.
        legacy_fn = f'hs{hs}_top{k}_{PHASE}_{src_tag}_qzero_catpos_cattok_onwards.jsonl'
        res = jsonl_acc(os.path.join(base, legacy_fn))
        if res is not None:
            print(f'[fallback] using the legacy baseline tag: {model_dir} hs{hs}')
    if res is None:
        return None
    correct, total = res
    if total < 1000:        # require completed run
        return None
    return 100.0 * correct / total


def load_baseline(model_dir, hs):
    return load_cell(model_dir, hs, 0, 'baseline')


def fmt_acc_drop(acc, baseline):
    if acc is None:
        return ' & ', 'TBD'
    if baseline is None:
        return f'{acc:.1f}', '?'
    delta = acc - baseline
    if abs(delta) < 0.05:
        delta_str = '\\dpos{0.0}'
    elif delta < 0:
        delta_str = f'\\drop{{{-delta:.1f}}}'
    else:
        delta_str = f'\\dpos{{{delta:.1f}}}'
    return f'{acc:.1f}', delta_str


def random_mean(model_dir, hs, k):
    """Return (mean_acc, std_acc, n_seeds) across available seeds, or None if zero seeds."""
    accs = []
    for seed in RANDOM_SEEDS:
        a = load_cell(model_dir, hs, k, 'random', seed=seed)
        if a is not None:
            accs.append(a)
    if not accs:
        return None
    if len(accs) == 1:
        return accs[0], 0.0, 1
    return statistics.mean(accs), statistics.stdev(accs), len(accs)


def emit_model_table(key, label):
    cfg = MODEL_CONFIGS[key]
    model_dir = cfg['model_dir']
    baselines = {hs: load_baseline(model_dir, hs) for hs in HS_LIST}

    rows = []
    rows.append(f'\\multirow{{1}}{{*}}{{Baseline}} & ' +
                ' & '.join('\\multicolumn{2}{c}{' + (f'{baselines[hs]:.1f}' if baselines[hs] is not None else 'TBD') + '}' for hs in HS_LIST) + ' \\\\')
    rows.append('\\midrule')

    SRC_LIST = [('attn-score (Q-zero)',       'attn_score'),
                ('retriever (Q-zero)',        'retriever'),
                ('retriever (output-zero)',   'retriever_outputzero'),
                # ('retriever-gen',    'retriever_gen'),     # hidden: gen-phase variants too unstable
                # ('overlap',          'overlap'),           # keeping only overlap_avg as the combined criterion
                # ('overlap-gen',      'overlap_gen'),
                ('overlap-avg',      'overlap_avg'),
                # ('overlap-avg-gen',  'overlap_avg_gen'),
                # ('score-sum',        'score_sum'),
                # ('interleave',       'interleave'),
                # ('interleave-gen',   'interleave_gen')
               ]
    for src_label, src_key in SRC_LIST:
        # If no cell for this source has data, skip the whole block (avoids
        # polluting tables for models we haven't run that source on yet).
        any_data = any(load_cell(model_dir, hs, K, src_key) is not None
                       for hs in HS_LIST for K in K_LIST)
        if not any_data:
            continue
        for K in K_LIST:
            cells = []
            for hs in HS_LIST:
                acc = load_cell(model_dir, hs, K, src_key)
                num, delta = fmt_acc_drop(acc, baselines.get(hs))
                if acc is None:
                    cells.append('\\multicolumn{2}{c}{TBD}')
                else:
                    cells.append(f'{num} & {delta}')
            rows.append(f'{src_label} top-{K} & ' + ' & '.join(cells) + ' \\\\')
        rows.append('\\midrule')

    for K in K_LIST:
        cells = []
        for hs in HS_LIST:
            res = random_mean(model_dir, hs, K)
            if res is None:
                cells.append('\\multicolumn{2}{c}{TBD}')
            else:
                mean, std, n = res
                base = baselines.get(hs)
                drop = (mean - base) if base is not None else 0
                if abs(drop) < 0.05:
                    delta = '\\dpos{0.0}'
                elif drop < 0:
                    delta = f'\\drop{{{-drop:.1f}}}'
                else:
                    delta = f'\\dpos{{{drop:.1f}}}'
                if n == 1:
                    cells.append(f'{mean:.1f} & {delta}')
                else:
                    cells.append(f'{mean:.1f}\\,$\\pm$\\,{std:.1f} & {delta}')
        rows.append(f'random-{K} & ' + ' & '.join(cells) + ' \\\\')

    body = '\n'.join(rows)
    table = f"""\\begin{{table}}[h]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{2.5pt}}
\\caption{{Top-$K$ dose-response on {label} (VHs, $n=1{{,}}000$). Rows: head selection (attn-score top, retriever top, overlap top, interleave top where run, random seed-mean). Overlap-$K$ ranks heads by $\\max(\\text{{rank}}_\\text{{attn}}, \\text{{rank}}_\\text{{retr}})$ (heads on which the two criteria agree most); interleave-$K$ takes a round-robin merge of the two rankings (attn \\#1, retr \\#1, attn \\#2, retr \\#2, \\ldots) skipping duplicates. Columns: haystack size. Cells show accuracy and $\\Delta$ from the in-pipeline baseline (top row). Random rows report mean$\\pm$std across {len(RANDOM_SEEDS)} seeds when all are available; otherwise the single available seed.}}
\\label{{tab:topk_{key.replace('-', '_')}}}
\\begin{{tabular}}{{l r@{{}}l r@{{}}l r@{{}}l r@{{}}l r@{{}}l r@{{}}l}}
\\toprule
& \\multicolumn{{2}}{{c}}{{hs=2}} & \\multicolumn{{2}}{{c}}{{hs=3}} & \\multicolumn{{2}}{{c}}{{hs=5}} & \\multicolumn{{2}}{{c}}{{hs=10}} & \\multicolumn{{2}}{{c}}{{hs=20}} & \\multicolumn{{2}}{{c}}{{hs=50}} \\\\
\\addlinespace[2pt]
{body}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='-', help='Output file (or - for stdout)')
    args = parser.parse_args()

    chunks = []
    chunks.append('% Auto-generated by eval/build_topk_sweep_table.py — do not edit by hand.\n'
                  '% Re-run after Phase 2 ablation jobs finish to refresh the cells.\n')
    for key, label in MODEL_ORDER:
        chunks.append(f'% --- {label} ---')
        chunks.append(emit_model_table(key, label))

    text = '\n'.join(chunks)
    if args.out == '-':
        sys.stdout.write(text)
    else:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            f.write(text)
        print(f'Wrote {args.out}', file=sys.stderr)


if __name__ == '__main__':
    main()
