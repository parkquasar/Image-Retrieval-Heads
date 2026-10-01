"""
Compute attn_score-based head rankings from H5 files (no GPU needed).

Supports two sources:
  - vhs:      VHS H5 files (cattok_attn_scores/layer_X per sample)
  - coco_mcq: COCO MCQ H5 files (attn_score/scores_sum per sample)

Output: CSV with columns layer,head,r1,avg_target_attn,n_samples
        saved to {model_dir}/head_rankings/attn_score_{source}/

Usage:
    # VHS-based ranking (all models)
    python training/rank_heads_attn_score.py --model-dir $ATTN_ROOT/internvl3_5_8b --source vhs
    python training/rank_heads_attn_score.py --model-dir $ATTN_ROOT/internvl3_5_4b --source vhs
    python training/rank_heads_attn_score.py --model-dir $ATTN_ROOT/qwen3_vl_8b --source vhs
    python training/rank_heads_attn_score.py --model-dir $ATTN_ROOT/qwen3_vl_4b --source vhs

    # COCO MCQ-based ranking (models with attn_score in H5)
    python training/rank_heads_attn_score.py --model-dir $ATTN_ROOT/qwen3_vl_8b --source coco_mcq
"""

import os, argparse, csv
import numpy as np
import h5py
from tqdm import tqdm
from glob import glob


def rank_from_vhs(model_dir, hs_list, phases):
    """Compute per-head R@1 from VHS H5 cattok/generation attn_scores."""
    vhs_dir = os.path.join(model_dir, 'vhs')
    results = {}  # (phase, hs) -> {r1: (L,H), avg_attn: (L,H), n: int}

    for hs in hs_list:
        h5_path = os.path.join(vhs_dir, f'vhs_hs{hs}.h5')
        if not os.path.exists(h5_path):
            print(f'  [SKIP] {h5_path} not found')
            continue

        f = h5py.File(h5_path, 'r')
        keys = sorted(f.keys())
        if not keys:
            f.close()
            continue

        # Detect dimensions from first sample
        g0 = f[keys[0]]
        for phase in phases:
            attn_key = f'{phase}_attn_scores'
            if attn_key not in g0:
                print(f'  [SKIP] {attn_key} not in {h5_path}')
                continue

            layer_keys = sorted(g0[attn_key].keys(), key=lambda x: int(x.split('_')[1]))
            num_layers = len(layer_keys)
            num_heads = g0[attn_key][layer_keys[0]].shape[0]

            r1_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
            attn_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
            n = 0

            for key in tqdm(keys, desc=f'VHS hs={hs} {phase}', leave=False):
                g = f[key]
                if attn_key not in g:
                    continue
                pos_idx = int(g.attrs.get('pos_idx', -1))
                if pos_idx < 0:
                    continue

                n += 1
                for li, lk in enumerate(layer_keys):
                    scores = g[attn_key][lk][:]  # (num_heads, n_images)
                    # R@1: is target image the highest-attended?
                    argmax = scores.argmax(axis=1)  # (num_heads,)
                    r1_sum[li] += (argmax == pos_idx).astype(np.float64)
                    # Avg attention to target
                    attn_sum[li] += scores[:, pos_idx].astype(np.float64)

            if n > 0:
                results[(phase, hs)] = {
                    'r1': r1_sum / n,
                    'avg_attn': attn_sum / n,
                    'n': n,
                    'num_layers': num_layers,
                    'num_heads': num_heads,
                }
        f.close()

    return results


def rank_from_coco_mcq(model_dir, phases):
    """Compute per-head R@1 from COCO MCQ H5 attn_score/scores_sum."""
    mcq_dir = os.path.join(model_dir, 'coco_mcq')
    h5_files = sorted(glob(os.path.join(mcq_dir, '**', 'head_outputs*.h5'), recursive=True))
    # Exclude failure files
    h5_files = [f for f in h5_files if 'failure' not in os.path.basename(f)]

    if not h5_files:
        print(f'  No COCO MCQ H5 files found in {mcq_dir}')
        return {}

    r1_sum = None
    attn_sum = None
    n = 0

    for h5_path in h5_files:
        print(f'  Reading {h5_path}')
        f = h5py.File(h5_path, 'r')
        keys = sorted(f.keys())

        for key in tqdm(keys, desc=os.path.basename(h5_path), leave=False):
            g = f[key]
            if 'attn_score' not in g:
                continue
            target_idx = int(g.attrs.get('target_index', -1))
            if target_idx < 0:
                continue

            scores_sum = g['attn_score']['scores_sum'][:]  # (n_layers, n_heads, n_images)
            num_layers, num_heads, _ = scores_sum.shape

            if r1_sum is None:
                r1_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
                attn_sum = np.zeros((num_layers, num_heads), dtype=np.float64)

            n += 1
            argmax = scores_sum.argmax(axis=2)  # (n_layers, n_heads)
            r1_sum += (argmax == target_idx).astype(np.float64)
            attn_sum += scores_sum[:, :, target_idx].astype(np.float64)

        f.close()

    if n == 0:
        return {}

    # COCO MCQ doesn't have separate phases — attn_score is at cattok position
    results = {}
    results[('cattok', 'all')] = {
        'r1': r1_sum / n,
        'avg_attn': attn_sum / n,
        'n': n,
        'num_layers': r1_sum.shape[0],
        'num_heads': r1_sum.shape[1],
    }
    return results


def save_rankings(results, out_dir):
    """Save per-(phase, subset) ranking CSVs."""
    os.makedirs(out_dir, exist_ok=True)

    for (phase, subset), data in results.items():
        r1 = data['r1']
        avg_attn = data['avg_attn']
        n = data['n']
        num_layers = data['num_layers']
        num_heads = data['num_heads']

        rows = []
        for l in range(num_layers):
            for h in range(num_heads):
                rows.append({
                    'layer': l,
                    'head': h,
                    'r1': round(r1[l, h], 6),
                    'avg_target_attn': round(avg_attn[l, h], 6),
                    'n_samples': n,
                })
        rows.sort(key=lambda x: -x['r1'])

        fname = f'{phase}_hs{subset}.csv' if subset != 'all' else f'{phase}.csv'
        out_path = os.path.join(out_dir, fname)
        with open(out_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['layer', 'head', 'r1', 'avg_target_attn', 'n_samples'])
            w.writeheader()
            w.writerows(rows)

        top = rows[0]
        print(f'  {fname}: top-1 = L{top["layer"]}H{top["head"]} (R@1={top["r1"]:.4f}), n={n}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True,
                        help='Model directory (e.g., $ATTN_ROOT/internvl3_5_8b)')
    parser.add_argument('--source', choices=['vhs', 'coco_mcq'], required=True)
    parser.add_argument('--output-dir', default=None,
                        help='Where to write the ranking CSVs (default: <model-dir>/head_rankings/attn_score_<source>)')
    parser.add_argument('--hs', nargs='+', type=int, default=[2, 3, 5, 10, 20],
                        help='HS sizes for VHS source')
    parser.add_argument('--phases', nargs='+', default=['cattok', 'generation'],
                        help='Phases for VHS source')
    args = parser.parse_args()

    out_dir = args.output_dir or os.path.join(args.model_dir, 'head_rankings', f'attn_score_{args.source}')
    print(f'Source: {args.source}')
    print(f'Output: {out_dir}')

    if args.source == 'vhs':
        results = rank_from_vhs(args.model_dir, args.hs, args.phases)
    else:
        results = rank_from_coco_mcq(args.model_dir, args.phases)

    if not results:
        print('No results computed.')
        return

    save_rankings(results, out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
