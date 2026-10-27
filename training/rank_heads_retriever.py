"""
Full-pool Recall@1/Recall@5 evaluation for all saved head checkpoints (MCQ format).

Strategy: open the HDF5 once, load the whole test set into memory, then
evaluate every (layer, head, phase, emb_type) combination with pure PyTorch ops (fast).

Dedup: samples that point to the same image are clustered so that the N-way retrieval is fair.
Generation phase: only ansA of the test set is used (correct=True on every verified sample).

Usage (model-dir mode, preferred):
    python training/rank_heads_retriever.py \
        --model-dir $ATTN_ROOT/internvl3_5_8b \
        --retriever-name standard_b256_lr1e-3_norm \
        --gpu_id 1

    # Auto-resolves: h5 from {model_dir}/coco_mcq/
    #   checkpoints from {model_dir}/retriever/{name}/
    #   output to {model_dir}/head_rankings/{name}/

Usage (explicit paths, legacy):
    python training/rank_heads_retriever.py \
        --checkpoint_dir $ATTN_ROOT/internvl3_5_8b/retriever/standard_b256_lr1e-3_norm \
        --h5_paths $ATTN_ROOT/internvl3_5_8b/coco_mcq/head_outputs_w0.h5 \
                   $ATTN_ROOT/internvl3_5_8b/coco_mcq/head_outputs_w1.h5 \
        --output_dir $ATTN_ROOT/internvl3_5_8b/head_rankings/standard_b256_lr1e-3_norm \
        --gpu_id 1
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import glob
import re
import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py

PHASES    = ['cattok', 'generation']
EMB_TYPES = ['patch_mean', 'proj_mean']


class ContrastiveHeadRetriever(nn.Module):
    def __init__(self, image_dim=1024, head_dim=128,
                 projection_dim=256, hidden_dim=512, num_layers=3,
                 arch='standard', use_norm=True):
        super().__init__()
        self.arch     = arch
        self.use_norm = use_norm

        def _mlp(in_dim, out_dim):
            layers = []
            d = in_dim
            for _ in range(num_layers - 1):
                layers += [nn.Linear(d, hidden_dim), nn.ReLU(), nn.Dropout(0.1)]
                d = hidden_dim
            layers.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers)

        if arch == 'standard':
            self.image_projector = _mlp(image_dim, projection_dim)
            self.head_projector  = _mlp(head_dim,  projection_dim)
        else:  # direct: image_dim → head_dim via Sequential(Linear) to match checkpoint keys
            self.image_projector = nn.Sequential(nn.Linear(image_dim, head_dim))

    def _norm(self, x):
        return F.normalize(x, p=2, dim=-1) if self.use_norm else x

    def forward(self, image_embedding=None, head_output=None):
        image_proj = head_proj = None
        if image_embedding is not None:
            image_proj = self._norm(self.image_projector(image_embedding))
        if head_output is not None:
            if self.arch == 'standard':
                head_proj = self._norm(self.head_projector(head_output))
            else:  # direct: head output used as-is
                head_proj = self._norm(head_output)
        return image_proj, head_proj


def get_test_sample_ids(h5_paths, train_ratio=0.8, seed=42):
    """
    Determine which sample_ids belong to the test split.
    Matches CombinedHeadOutputDataset split logic exactly.
    """
    all_ids = set()
    for path in h5_paths:
        with h5py.File(path, 'r') as f:
            for name in f.keys():
                all_ids.add(int(name.split('_')[1]))

    unique_ids = sorted(all_ids)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(unique_ids))
    n_train = int(len(unique_ids) * train_ratio)
    return set(unique_ids[i] for i in perm[n_train:])


def preload_test_data(h5_paths, test_ids, num_layers, num_heads, emb_type):
    """
    Load all test samples into memory.

    For each sample, loads:
      - gt_embedding: emb[target_index]          → (image_dim,)
      - head_outputs[cattok]:    (num_layers, num_heads, head_dim)
      - head_outputs[generation]: (num_layers, num_heads, head_dim) using ansA

    Returns:
        gt_embs:         np.ndarray (N, image_dim)
        cattok_outputs:  np.ndarray (N, num_layers, num_heads, head_dim)
        gen_outputs:     np.ndarray (N, num_layers, num_heads, head_dim)
        sample_ids:      List[int]
    """
    # Collect (file_idx, sample_name) for test split
    test_items = []
    h5_files = [h5py.File(p, 'r') for p in h5_paths]
    for file_idx, h5f in enumerate(h5_files):
        for sample_name in sorted(h5f.keys()):
            sid = int(sample_name.split('_')[1])
            if sid not in test_ids:
                continue
            grp = h5f[sample_name]
            if 'image_embeddings' not in grp or emb_type not in grp['image_embeddings']:
                continue
            if 'cattok' not in grp:
                continue
            if 'ansA' not in grp or 'generation' not in grp['ansA']:
                continue
            test_items.append((file_idx, sample_name))

    N = len(test_items)
    if N == 0:
        raise ValueError(f"No test samples found for emb_type={emb_type}")

    # Determine dims from first sample
    fi0, sn0 = test_items[0]
    s0 = h5_files[fi0][sn0]
    image_dim = s0['image_embeddings'][emb_type].shape[1]
    head_dim  = s0['cattok']['layer_0'].shape[2]

    gt_embs        = np.zeros((N, image_dim), dtype=np.float32)
    cattok_outputs = np.zeros((N, num_layers, num_heads, head_dim), dtype=np.float32)
    gen_outputs    = np.zeros((N, num_layers, num_heads, head_dim), dtype=np.float32)
    sample_ids     = []

    print(f"Preloading {N} test samples × {num_layers}L × {num_heads}H × {head_dim}d "
          f"× 2 phases ({emb_type}) ...")
    print(f"Memory: {(cattok_outputs.nbytes + gen_outputs.nbytes) / 1e9:.2f} GB")

    for i, (fi, sn) in enumerate(test_items):
        s = h5_files[fi][sn]
        emb = s['image_embeddings'][emb_type][:]
        tgt = int(s.attrs['target_index'])
        gt_embs[i] = emb[tgt]

        for l in range(num_layers):
            cattok_outputs[i, l] = s['cattok'][f'layer_{l}'][0]
            gen_outputs[i, l]    = s['ansA']['generation'][f'layer_{l}'][0]

        sample_ids.append(int(s.attrs['sample_id']))

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{N} loaded")

    for h5f in h5_files:
        h5f.close()

    print("Preload done.")
    return gt_embs, cattok_outputs, gen_outputs, sample_ids


def build_dedup_clusters(gt_embs_np, sim_threshold=0.9999):
    """
    Cluster GT embeddings by near-exact similarity (same image → same cluster).
    Returns:
        cluster_labels: List[int] length N
        cluster_reps:   List[int] — representative index per cluster
    """
    gt_t = torch.from_numpy(gt_embs_np).float()
    norm = gt_t / gt_t.norm(dim=1, keepdim=True)
    sim  = torch.mm(norm, norm.T)

    N        = gt_t.size(0)
    assigned = [-1] * N
    reps     = []
    cid      = 0
    for i in range(N):
        if assigned[i] == -1:
            dups = (sim[i] > sim_threshold).nonzero(as_tuple=True)[0].tolist()
            for j in dups:
                assigned[j] = cid
            reps.append(i)
            cid += 1
    return assigned, reps


def main(args):
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Detect num_layers / num_heads from first h5 file
    with h5py.File(args.h5_paths[0], 'r') as f:
        sn = sorted(f.keys())[0]
        grp = f[sn]
        cattok_layers = [k for k in grp['cattok'].keys() if k.startswith('layer_')]
        num_layers = len(cattok_layers)
        num_heads  = grp['cattok']['layer_0'].shape[1]
    print(f"Model: {num_layers} layers × {num_heads} heads")

    test_ids = get_test_sample_ids(args.h5_paths, train_ratio=args.train_ratio, seed=args.seed)
    print(f"Test split: {len(test_ids)} unique sample_ids")

    os.makedirs(args.output_dir, exist_ok=True)

    # Evaluate each (phase, emb_type) combo that has checkpoints
    for emb_type in args.emb_types:
        # Preload data for this emb_type (both phases share the same embeddings)
        gt_embs_np, cattok_np, gen_np, sample_ids = preload_test_data(
            args.h5_paths, test_ids, num_layers, num_heads, emb_type
        )
        N = len(sample_ids)
        gt_embs = torch.from_numpy(gt_embs_np).to(device)

        print(f"\nBuilding dedup clusters for emb_type={emb_type} ...")
        cluster_labels, cluster_reps = build_dedup_clusters(gt_embs_np)
        n_unique       = len(cluster_reps)
        correct_cluster = torch.tensor(cluster_labels, device=device)
        rep_idx         = torch.tensor(cluster_reps,   device=device)
        print(f"Dedup pool: {n_unique} unique images (from {N} samples)")

        for phase in args.phases:
            head_outputs_np = cattok_np if phase == 'cattok' else gen_np

            # Find checkpoints for this combo
            pattern   = os.path.join(
                args.checkpoint_dir,
                f'contrastive_layer??_head??_{phase}_{emb_type}.pt'
            )
            ckpt_paths = sorted(glob.glob(pattern))
            print(f"\n[{phase}/{emb_type}] Found {len(ckpt_paths)} checkpoints")
            if not ckpt_paths:
                continue

            def parse_lh(path):
                m = re.search(r'layer(\d+)_head(\d+)', os.path.basename(path))
                return int(m.group(1)), int(m.group(2))

            sample_ckpt = torch.load(ckpt_paths[0], map_location='cpu')
            proj_dim    = sample_ckpt['projection_dim']
            hidden_dim  = sample_ckpt.get('hidden_dim', 512)
            num_mlp     = sample_ckpt.get('num_mlp_layers', 3)
            image_dim   = sample_ckpt['image_dim']
            head_dim    = sample_ckpt['head_dim']
            arch        = sample_ckpt.get('arch', 'standard')
            use_norm    = sample_ckpt.get('use_norm', True)

            results = []
            for ckpt_path in ckpt_paths:
                layer, head = parse_lh(ckpt_path)
                try:
                    ckpt = torch.load(ckpt_path, map_location='cpu')
                    model = ContrastiveHeadRetriever(
                        image_dim=image_dim, head_dim=head_dim,
                        projection_dim=proj_dim, hidden_dim=hidden_dim, num_layers=num_mlp,
                        arch=arch, use_norm=use_norm,
                    ).to(device)
                    model.load_state_dict(ckpt['model_state_dict'])
                    model.eval()

                    with torch.no_grad():
                        h_out = torch.from_numpy(
                            head_outputs_np[:, layer, head, :]
                        ).to(device)
                        gt_proj, head_proj = model(image_embedding=gt_embs, head_output=h_out)

                        # Full-pool R@1
                        sim_full = torch.mm(head_proj, gt_proj.T)
                        labels   = torch.arange(N, device=device)
                        r1_full  = (sim_full.argmax(dim=1) == labels).float().mean().item()

                        # Dedup R@1 / R@5 / R@10
                        gt_proj_dedup = gt_proj[rep_idx]
                        sim_dedup     = torch.mm(head_proj, gt_proj_dedup.T)
                        r1_dedup  = (sim_dedup.argmax(dim=1) == correct_cluster).float().mean().item()
                        top10     = sim_dedup.topk(min(10, n_unique), dim=1).indices
                        r5_dedup  = (top10[:, :5] == correct_cluster.unsqueeze(1)).any(dim=1).float().mean().item()
                        r10_dedup = (top10       == correct_cluster.unsqueeze(1)).any(dim=1).float().mean().item()

                    results.append({
                        'layer': layer, 'head': head,
                        'r1_dedup': r1_dedup, 'r5_dedup': r5_dedup, 'r10_dedup': r10_dedup,
                        'r1_full': r1_full,
                        'n_test': N, 'n_unique': n_unique,
                        'train_r1': ckpt.get('recall_at_1'), 'ckpt_epoch': ckpt.get('epoch', -1),
                    })

                except Exception as e:
                    print(f"  ERROR L{layer:02d}H{head:02d}: {e}")
                    results.append({
                        'layer': layer, 'head': head,
                        'r1_dedup': None, 'r5_dedup': None, 'r10_dedup': None, 'r1_full': None,
                        'n_test': N, 'n_unique': n_unique, 'train_r1': None, 'ckpt_epoch': None,
                    })

            results_valid = sorted(
                [r for r in results if r['r1_dedup'] is not None],
                key=lambda x: x['r1_dedup'], reverse=True
            )

            # Save CSV
            csv_path = os.path.join(args.output_dir, f'{phase}_{emb_type}_fullpool.csv')
            fieldnames = ['layer', 'head', 'r1_dedup', 'r5_dedup', 'r10_dedup', 'r1_full',
                          'n_test', 'n_unique', 'train_r1', 'ckpt_epoch']
            with open(csv_path, 'w', newline='') as csvf:
                writer = csv.DictWriter(csvf, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(results_valid)
            print(f"Saved → {csv_path}")

            # Print top-20
            print(f"\nTop-20 [{phase}/{emb_type}] (dedup {n_unique}-way):")
            print(f"{'Rank':>4}  {'Head':>8}  {'R@1':>7}  {'R@5':>7}  {'R@10':>7}  "
                  f"{'R@1_full':>8}  {'train_R@1':>9}")
            print("-" * 72)
            for rank, r in enumerate(results_valid[:20], 1):
                tr = f"{r['train_r1']:.3f}" if r['train_r1'] is not None else "  N/A"
                print(f"{rank:4d}  L{r['layer']:02d}H{r['head']:02d}    "
                      f"{r['r1_dedup']:.3f}    {r['r5_dedup']:.3f}    {r['r10_dedup']:.3f}    "
                      f"{r['r1_full']:.3f}    {tr}")

            # Heatmap
            if args.plot:
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt

                    grid = np.full((num_layers, num_heads), np.nan)
                    for r in results_valid:
                        grid[r['layer'], r['head']] = r['r1_dedup']

                    fig, ax = plt.subplots(figsize=(18, 10))
                    valid = grid[~np.isnan(grid)]
                    im = ax.imshow(grid, aspect='auto', cmap='RdYlGn',
                                   vmin=valid.min(), vmax=valid.max())
                    plt.colorbar(im, ax=ax, label='Dedup R@1')
                    ax.set_xlabel('Head index', fontsize=12)
                    ax.set_ylabel('Layer index', fontsize=12)
                    ax.set_title(f'Head Retriever — Dedup R@1 ({n_unique}-way)\n'
                                 f'{phase}/{emb_type}, {N} test items', fontsize=13)
                    ax.set_xticks(range(num_heads))
                    ax.set_yticks(range(num_layers))
                    ax.set_xticklabels(range(num_heads), fontsize=7)
                    ax.set_yticklabels(range(num_layers), fontsize=7)
                    for rank, r in enumerate(results_valid[:10], 1):
                        ax.text(r['head'], r['layer'], str(rank),
                                ha='center', va='center', fontsize=7, fontweight='bold')
                    plt.tight_layout()
                    plot_path = csv_path.replace('.csv', '_heatmap.png')
                    plt.savefig(plot_path, dpi=150)
                    print(f"Heatmap saved → {plot_path}")
                    plt.close()
                except Exception as e:
                    print(f"Heatmap failed: {e}")

        # Free memory before next emb_type
        del gt_embs, gt_embs_np, cattok_np, gen_np
        torch.cuda.empty_cache()


def _find_h5_files(model_dir):
    """Auto-detect COCO MCQ h5 files from model directory."""
    coco_dir = os.path.join(model_dir, 'coco_mcq')
    h5_files = sorted(glob.glob(os.path.join(coco_dir, '**', 'head_outputs*.h5'), recursive=True))
    h5_files = [f for f in h5_files if 'failure' not in os.path.basename(f)]
    if not h5_files:
        raise FileNotFoundError(f'No head_outputs*.h5 found in {coco_dir}')
    return h5_files


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model directory (e.g., $ATTN_ROOT/internvl3_5_8b). '
                             'Use with --retriever-name to auto-resolve all paths.')
    parser.add_argument('--retriever-name', type=str, default=None,
                        help='Retriever config name (e.g., standard_b256_lr1e-3_norm). '
                             'Resolves checkpoint/output dirs under --model-dir.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Explicit checkpoint dir (overrides --model-dir)')
    parser.add_argument('--h5_paths', type=str, nargs='+', default=None,
                        help='Explicit h5 paths (overrides --model-dir)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Explicit output dir (overrides --model-dir)')
    parser.add_argument('--gpu_id',    type=int, default=1)
    parser.add_argument('--phases',    nargs='+', default=PHASES,   choices=PHASES)
    parser.add_argument('--emb_types', nargs='+', default=EMB_TYPES, choices=EMB_TYPES)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--plot', action='store_true', default=True)
    args = parser.parse_args()

    # Resolve paths from --model-dir when explicit paths not given
    if args.model_dir and not args.h5_paths:
        args.h5_paths = _find_h5_files(args.model_dir)
        print(f'Auto-detected {len(args.h5_paths)} h5 files from {args.model_dir}/coco_mcq/')
    if args.h5_paths is None:
        parser.error('Either --model-dir or --h5_paths is required')
    if args.model_dir and args.retriever_name:
        if not args.checkpoint_dir:
            args.checkpoint_dir = os.path.join(args.model_dir, 'retriever', args.retriever_name)
            print(f'Checkpoint dir: {args.checkpoint_dir}')
        if not args.output_dir:
            args.output_dir = os.path.join(args.model_dir, 'head_rankings', args.retriever_name)
            print(f'Output dir: {args.output_dir}')
    if args.checkpoint_dir is None:
        parser.error('Either --model-dir + --retriever-name or --checkpoint_dir is required')
    if args.output_dir is None:
        parser.error('Either --model-dir + --retriever-name or --output_dir is required')

    main(args)
