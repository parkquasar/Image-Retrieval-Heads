"""
Vectorized head retriever training — all 32 heads per layer trained simultaneously.

Key optimisations over train_all_v2.py:
  - Task granularity: (layer, phase, emb_type) — 32 heads share one HDF5 read
  - VectorizedHeadRetriever: all 32 heads in a single torch.bmm forward pass
  - Loss: batch-wide stored negatives — for each sample i, negative pool is
      127 in-batch gt's  +  all 128 × (4~6) stored negatives from the batch
    → 640~770 negatives total (vs 4~5 previously), fully vectorized (no Python loops)

Checkpoint format compatible with rank_heads_retriever.py.

Usage (model-dir mode — preferred):
    python training/train_retrieval_heads.py \
        --model-dir $ATTN_ROOT/internvl3_5_8b \
        --gpu_ids 1 3 --processes_per_gpu 2

    # Auto-detects h5 from {model_dir}/coco_mcq/**/head_outputs*.h5
    # Saves to {model_dir}/retriever/{arch}_b{bs}_lr{lr}_{norm|nonorm}/

Usage (explicit paths — legacy):
    python training/train_retrieval_heads.py \
        --h5_paths $ATTN_ROOT/internvl3_5_8b/coco_mcq/head_outputs_w0.h5 \
                   $ATTN_ROOT/internvl3_5_8b/coco_mcq/head_outputs_w1.h5 \
        --output_dir $ATTN_ROOT/internvl3_5_8b/retriever/standard_b256_lr1e-3_norm \
        --gpu_ids 0 1 2 3 --processes_per_gpu 2
"""

import os, sys, argparse, multiprocessing as mp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

PHASES    = ['cattok', 'generation']
EMB_TYPES = ['patch_mean', 'proj_mean']


# ── Vectorized model ───────────────────────────────────────────────────────────

class VectorizedHeadRetriever(nn.Module):
    """
    32 independent ContrastiveHeadRetrievers fused into batched matmul.

    arch='standard': both image and head projectors → shared proj_dim space
    arch='direct':   only image projector (image_dim → head_dim);
                     head output used directly (no head projector)

    use_norm: whether to L2-normalize projections before similarity
    """
    def __init__(self, n_heads: int, image_dim: int, head_dim: int,
                 proj_dim: int = 256, hidden_dim: int = 512, n_mlp_layers: int = 3,
                 arch: str = 'standard', use_norm: bool = True):
        super().__init__()
        self.n_heads      = n_heads
        self.proj_dim     = proj_dim
        self.n_mlp_layers = n_mlp_layers
        self.arch         = arch
        self.use_norm     = use_norm

        def _make_params(dims):
            Ws, Bs = nn.ParameterList(), nn.ParameterList()
            for i in range(len(dims) - 1):
                W = nn.Parameter(torch.empty(n_heads, dims[i], dims[i+1]))
                b = nn.Parameter(torch.zeros(n_heads, 1, dims[i+1]))
                nn.init.kaiming_uniform_(W.reshape(n_heads * dims[i], dims[i+1]))
                Ws.append(W); Bs.append(b)
            return Ws, Bs

        if arch == 'standard':
            img_dims  = [image_dim] + [hidden_dim] * (n_mlp_layers - 1) + [proj_dim]
            head_dims = [head_dim]  + [hidden_dim] * (n_mlp_layers - 1) + [proj_dim]
            self.img_W,  self.img_b  = _make_params(img_dims)
            self.head_W, self.head_b = _make_params(head_dims)
        else:  # direct: image_dim → head_dim only, no head projector
            self.img_W,  self.img_b  = _make_params([image_dim, head_dim])
            self.head_W, self.head_b = nn.ParameterList(), nn.ParameterList()

    def _mlp(self, x, Ws, Bs):
        # x: (n_heads, N, in_dim)
        for i, (W, b) in enumerate(zip(Ws, Bs)):
            x = torch.bmm(x, W) + b
            if i < self.n_mlp_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.1, training=self.training)
        return x

    def _norm(self, x):
        return F.normalize(x, p=2, dim=-1) if self.use_norm else x

    def project_image(self, emb):
        """emb: (N, image_dim) → (n_heads, N, out_dim) optionally normalised"""
        x = emb.unsqueeze(0).expand(self.n_heads, -1, -1)
        return self._norm(self._mlp(x, self.img_W, self.img_b))

    def project_head(self, head_outputs):
        """head_outputs: (B, n_heads, head_dim) → (n_heads, B, out_dim) optionally normalised"""
        x = head_outputs.permute(1, 0, 2).contiguous()
        if self.arch == 'standard':
            return self._norm(self._mlp(x, self.head_W, self.head_b))
        else:  # direct: head is used as-is
            return self._norm(x)


# ── Loss ──────────────────────────────────────────────────────────────────────

def info_nce_batch_negatives(head_proj, img_proj, neg_proj, neg_mask, temperature):
    """
    Fully vectorized InfoNCE over all 32 heads simultaneously.

    For each sample i, the denominator contains:
        - all B in-batch gt projections (positive at index i)
        - all B × max_neg stored negative projections (masked for padding)

    Args:
        head_proj: (n_heads, B, proj_dim)   — normalised
        img_proj:  (n_heads, B, proj_dim)   — normalised gt embeddings
        neg_proj:  (n_heads, B*max_neg, proj_dim) — normalised stored negatives
        neg_mask:  (B*max_neg,) bool        — True for valid (non-padded) entries
        temperature: float

    Returns:
        loss: scalar (mean over n_heads × B)
    """
    n_heads, B, proj_dim = head_proj.shape

    # In-batch similarity: (n_heads, B, B)
    inbatch_sim = torch.bmm(head_proj, img_proj.transpose(1, 2)) / temperature

    # Stored negative similarity: (n_heads, B, B*max_neg)
    neg_sim = torch.bmm(head_proj, neg_proj.transpose(1, 2)) / temperature
    # Mask padding positions to -inf so they don't contribute to softmax
    neg_sim = neg_sim.masked_fill(
        ~neg_mask.unsqueeze(0).unsqueeze(1), float('-inf')
    )

    # Concatenate: (n_heads, B, B + B*max_neg)
    # Positive for sample i is at column i in inbatch_sim
    all_logits = torch.cat([inbatch_sim, neg_sim], dim=2)

    # Labels: position of positive = i (diagonal of in-batch block)
    labels = torch.arange(B, device=head_proj.device)  # (B,)

    # Reshape for cross_entropy: (n_heads*B, B + B*max_neg)
    loss = F.cross_entropy(
        all_logits.reshape(n_heads * B, -1),
        labels.unsqueeze(0).expand(n_heads, -1).reshape(-1),
    )
    return loss


# ── Training ──────────────────────────────────────────────────────────────────

def train_layer_combo(args_tuple):
    layer, phase, emb_type, gpu_id, base_args = args_tuple
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    if not base_args['verbose']:
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    try:
        device = torch.device('cuda:0')
        from head_dataloader import get_combined_dataloader, get_h5_dimensions

        dims      = get_h5_dimensions(base_args['h5_paths'][0], phase=phase, emb_type=emb_type)
        n_heads   = dims['num_heads']
        image_dim = dims['image_dim']
        head_dim  = dims['head_dim']
        proj_dim  = base_args['projection_dim']

        # Check which heads already have checkpoints
        done_heads = set()
        for head in range(n_heads):
            ckpt_path = os.path.join(
                base_args['output_dir'],
                f'contrastive_layer{layer:02d}_head{head:02d}_{phase}_{emb_type}.pt'
            )
            if os.path.exists(ckpt_path):
                done_heads.add(head)

        if len(done_heads) == n_heads:
            recalls = []
            for head in range(n_heads):
                ckpt_path = os.path.join(
                    base_args['output_dir'],
                    f'contrastive_layer{layer:02d}_head{head:02d}_{phase}_{emb_type}.pt'
                )
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                recalls.append(ckpt.get('recall_at_1', 0.0))
            return (layer, phase, emb_type, recalls, True, "SKIPPED")

        # Load data ONCE — head_idx=None returns all heads
        train_dl = get_combined_dataloader(
            h5_paths=base_args['h5_paths'], layer_idx=layer, head_idx=None,
            phase=phase, emb_type=emb_type,
            batch_size=base_args['batch_size'], shuffle=True,
            split='train', train_ratio=base_args['train_ratio'],
            seed=base_args['seed'], drop_last=True, verbose=False,
        )
        test_dl = get_combined_dataloader(
            h5_paths=base_args['h5_paths'], layer_idx=layer, head_idx=None,
            phase=phase, emb_type=emb_type,
            batch_size=base_args['batch_size'], shuffle=False,
            split='test', train_ratio=base_args['train_ratio'],
            seed=base_args['seed'], drop_last=False, verbose=False,
        )

        arch     = base_args['arch']
        use_norm = base_args['use_norm']
        model = VectorizedHeadRetriever(
            n_heads=n_heads, image_dim=image_dim, head_dim=head_dim,
            proj_dim=proj_dim, hidden_dim=base_args['hidden_dim'],
            n_mlp_layers=base_args['num_mlp_layers'],
            arch=arch, use_norm=use_norm,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=base_args['lr'],
            weight_decay=base_args['weight_decay'],
        )
        lr_schedule    = base_args.get('lr_schedule', 'cosine')
        warmup_epochs  = base_args.get('warmup_epochs', 0)
        n_epochs       = base_args['epochs']
        if lr_schedule == 'flat':
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda e: 1.0)
        elif warmup_epochs > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs - warmup_epochs, eta_min=base_args['lr'] * 0.01)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=n_epochs, eta_min=base_args['lr'] * 0.01
            )

        temperature     = base_args['temperature']
        best_recalls    = [-1.0] * n_heads
        best_losses     = [float('inf')] * n_heads
        best_img_W      = [None] * n_heads
        best_img_b      = [None] * n_heads
        best_head_W     = [None] * n_heads
        best_head_b     = [None] * n_heads

        for epoch in range(base_args['epochs']):
            # ── Train ──────────────────────────────────────────────────────
            model.train()
            for batch in train_dl:
                head_outputs = batch['head_outputs'].to(device)   # (B, n_heads, head_dim)
                gt_emb       = batch['gt_embeddings'].to(device)  # (B, image_dim)
                neg_list     = batch['negative_embeddings']        # List[B] of (N_i, image_dim)

                # Project gt embeddings for all heads: (n_heads, B, proj_dim)
                img_proj  = model.project_image(gt_emb)
                head_proj = model.project_head(head_outputs)

                # Stack and pad stored negatives: (B, max_neg, image_dim)
                max_neg    = max(n.size(0) for n in neg_list)
                neg_padded = torch.zeros(len(neg_list), max_neg, image_dim, device=device)
                neg_mask   = torch.zeros(len(neg_list), max_neg, dtype=torch.bool, device=device)
                for i, n in enumerate(neg_list):
                    neg_padded[i, :n.size(0)] = n.to(device)
                    neg_mask[i, :n.size(0)]   = True

                # Project all stored negatives at once: (n_heads, B*max_neg, proj_dim)
                B = gt_emb.size(0)
                neg_proj = model.project_image(neg_padded.reshape(B * max_neg, image_dim))
                # neg_proj: (n_heads, B*max_neg, proj_dim)

                loss = info_nce_batch_negatives(
                    head_proj, img_proj, neg_proj,
                    neg_mask.reshape(B * max_neg), temperature,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            scheduler.step()

            # ── Evaluate ───────────────────────────────────────────────────
            model.eval()
            all_head_projs = []
            all_img_projs  = []

            with torch.no_grad():
                for batch in test_dl:
                    head_outputs = batch['head_outputs'].to(device)
                    gt_emb       = batch['gt_embeddings'].to(device)
                    img_proj  = model.project_image(gt_emb)
                    head_proj = model.project_head(head_outputs)
                    all_head_projs.append(head_proj)
                    all_img_projs.append(img_proj)

            # (n_heads, N_test, proj_dim)
            all_head_cat = torch.cat(all_head_projs, dim=1)
            all_img_cat  = torch.cat(all_img_projs,  dim=1)
            N_test = all_head_cat.size(1)
            labels = torch.arange(N_test, device=device)

            # Per-head recall@1
            sim_all = torch.bmm(all_head_cat, all_img_cat.transpose(1, 2))  # (n_heads, N, N)
            for h in range(n_heads):
                r1 = (sim_all[h].argmax(dim=1) == labels).float().mean().item()
                tl = F.cross_entropy(sim_all[h] / temperature, labels).item()
                if r1 > best_recalls[h] or (r1 == best_recalls[h] and tl < best_losses[h]):
                    best_recalls[h] = r1
                    best_losses[h]  = tl
                    # Snapshot this head's slice of each parameter
                    with torch.no_grad():
                        best_img_W[h]  = [W[h].cpu().clone() for W in model.img_W]
                        best_img_b[h]  = [b[h].cpu().clone() for b in model.img_b]
                        if arch == 'standard':
                            best_head_W[h] = [W[h].cpu().clone() for W in model.head_W]
                            best_head_b[h] = [b[h].cpu().clone() for b in model.head_b]

        # ── Save per-head checkpoints (compatible with rank_heads_retriever.py) ──
        os.makedirs(base_args['output_dir'], exist_ok=True)
        n_mlp = base_args['num_mlp_layers']

        for head in range(n_heads):
            if head in done_heads:
                continue
            ckpt_path = os.path.join(
                base_args['output_dir'],
                f'contrastive_layer{layer:02d}_head{head:02d}_{phase}_{emb_type}.pt'
            )
            # Build ContrastiveHeadRetriever-compatible state dict
            state = {}
            if arch == 'standard':
                linear_pos = 0
                for i in range(n_mlp):
                    state[f'image_projector.{linear_pos}.weight'] = best_img_W[head][i].T.clone()
                    state[f'image_projector.{linear_pos}.bias']   = best_img_b[head][i].squeeze(0).clone()
                    state[f'head_projector.{linear_pos}.weight']  = best_head_W[head][i].T.clone()
                    state[f'head_projector.{linear_pos}.bias']    = best_head_b[head][i].squeeze(0).clone()
                    if i < n_mlp - 1:
                        linear_pos += 3
            else:  # direct: only image projector (image_dim → head_dim)
                state['image_projector.0.weight'] = best_img_W[head][0].T.clone()
                state['image_projector.0.bias']   = best_img_b[head][0].squeeze(0).clone()

            torch.save({
                'epoch':            base_args['epochs'] - 1,
                'model_state_dict': state,
                'recall_at_1':      best_recalls[head],
                'test_loss':        best_losses[head],
                'layer':            layer,
                'head':             head,
                'phase':            phase,
                'emb_type':         emb_type,
                'image_dim':        image_dim,
                'head_dim':         head_dim,
                'projection_dim':   proj_dim,
                'hidden_dim':       base_args['hidden_dim'],
                'num_mlp_layers':   n_mlp,
                'arch':             arch,
                'use_norm':         use_norm,
                'temperature':      temperature,
                'h5_paths':         base_args['h5_paths'],
            }, ckpt_path)

        return (layer, phase, emb_type, best_recalls, True, None)

    except Exception as e:
        import traceback
        return (layer, phase, emb_type, None, False, traceback.format_exc())


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    from head_dataloader import get_h5_dimensions

    dims = get_h5_dimensions(args.h5_paths[0], phase='generation', emb_type='patch_mean')
    num_layers = dims['num_layers']

    if args.end_layer is None:
        args.end_layer = num_layers

    base_args = {
        'h5_paths':       args.h5_paths,
        'batch_size':     args.batch_size,
        'epochs':         args.epochs,
        'lr':             args.lr,
        'weight_decay':   args.weight_decay,
        'hidden_dim':     args.hidden_dim,
        'num_mlp_layers': args.num_mlp_layers,
        'projection_dim': args.projection_dim,
        'temperature':    args.temperature,
        'train_ratio':    args.train_ratio,
        'seed':           args.seed,
        'output_dir':     args.output_dir,
        'verbose':        args.verbose,
        'arch':           args.arch,
        'use_norm':       not args.no_norm,
        'lr_schedule':    args.lr_schedule,
        'warmup_epochs':  args.warmup_epochs,
    }

    combos  = [(p, e) for p in args.phases for e in args.emb_types]
    gpu_ids = args.gpu_ids
    n_gpus  = len(gpu_ids)
    n_slots = n_gpus * args.processes_per_gpu  # total parallel slots

    # Assign tasks so that within each GPU, all combos rotate across its slots.
    # Build flat list ordered by combo first, then layer — then assign slot
    # round-robin.  With 4 GPUs × 2 slots = 8 slots and 4 combos:
    #   slot 0 (GPU 0): cattok/patch layers 0,4,8,...
    #   slot 1 (GPU 0): generation/proj layers 0,4,8,...
    #   slot 2 (GPU 1): cattok/proj  layers 0,4,8,...
    #   slot 3 (GPU 1): generation/patch layers 0,4,8,...
    # → each GPU handles 2 different combos simultaneously → balanced memory.
    flat = []
    for phase, emb_type in combos:
        for layer in range(args.start_layer, args.end_layer):
            flat.append((layer, phase, emb_type))

    tasks = []
    for i, (layer, phase, emb_type) in enumerate(flat):
        gpu_id = gpu_ids[(i % n_slots) // args.processes_per_gpu]
        tasks.append((layer, phase, emb_type, gpu_id, base_args))

    n_retrievers = len(tasks) * dims['num_heads']
    print(f"Training {n_retrievers} retrievers via {len(tasks)} vectorized tasks")
    print(f"  layers {args.start_layer}–{args.end_layer-1} × {len(combos)} combos, "
          f"{dims['num_heads']} heads each")
    print(f"GPUs: {gpu_ids}, {args.processes_per_gpu} proc/GPU → "
          f"{len(gpu_ids)*args.processes_per_gpu} parallel tasks")
    print(f"Batch={args.batch_size}, Epochs={args.epochs}, T={args.temperature}, LR={args.lr}")
    print(f"Output: {args.output_dir}\n")

    with mp.Pool(processes=len(gpu_ids) * args.processes_per_gpu) as pool:
        results = []
        for i, result in enumerate(pool.imap_unordered(train_layer_combo, tasks)):
            layer, phase, emb_type, recalls, success, error = result
            tag = f"L{layer:02d} {phase}/{emb_type}"
            if success:
                status = "SKIP" if error == "SKIPPED" else "OK"
                if recalls:
                    mean_r = sum(recalls) / len(recalls)
                    top3   = sorted(recalls, reverse=True)[:3]
                    print(f"[{i+1:4d}/{len(tasks)}] {status} {tag:<35} "
                          f"mean_R@1={mean_r:.3f}  top3={[f'{r:.3f}' for r in top3]}")
            else:
                short_err = (error or '')[-300:].replace('\n', ' | ')
                print(f"[{i+1:4d}/{len(tasks)}] FAIL {tag:<35} {short_err}")
            results.append(result)

    print("\n" + "=" * 70)
    ok  = [r for r in results if r[4]]
    bad = [r for r in results if not r[4]]
    print(f"Done: {len(ok)}/{len(tasks)} tasks, {len(bad)} failed")
    if ok:
        all_r = [r for res in ok if res[3] for r in res[3]]
        if all_r:
            print(f"Overall R@1 mean={sum(all_r)/len(all_r):.3f}  "
                  f"top5={sorted(all_r, reverse=True)[:5]}")
        for phase, emb_type in combos:
            sub = [r for res in ok if res[1]==phase and res[2]==emb_type and res[3]
                   for r in res[3]]
            if sub:
                print(f"  {phase}/{emb_type}: mean={sum(sub)/len(sub):.3f}  "
                      f"top3={sorted(sub, reverse=True)[:3]}")
    if bad:
        print("Failed:")
        for layer, phase, emb_type, _, _, err in bad[:3]:
            print(f"  L{layer:02d} {phase}/{emb_type}: {(err or '')[:300]}")
    print("=" * 70)


def _find_h5_files(model_dir):
    """Auto-detect COCO MCQ h5 files from model directory."""
    import glob as _glob
    coco_dir = os.path.join(model_dir, 'coco_mcq')
    h5_files = sorted(_glob.glob(os.path.join(coco_dir, '**', 'head_outputs*.h5'), recursive=True))
    h5_files = [f for f in h5_files if 'failure' not in os.path.basename(f)]
    if not h5_files:
        raise FileNotFoundError(f'No head_outputs*.h5 found in {coco_dir}')
    return h5_files


def _make_config_name(args):
    """Generate retriever config name from hyperparams."""
    lr_str = f'{args.lr:.0e}'.replace('+', '').replace('0', '')  # 1e-3 → 1e-3
    lr_str = f'{args.lr}'.rstrip('0').replace('0.', '').replace('.', '')
    # Simple format: 1e-3, 1e-4, etc.
    if args.lr == 1e-3:
        lr_str = '1e-3'
    elif args.lr == 1e-4:
        lr_str = '1e-4'
    elif args.lr == 1e-5:
        lr_str = '1e-5'
    else:
        lr_str = f'{args.lr:g}'
    norm_str = 'norm' if not args.no_norm else 'nonorm'
    return f'{args.arch}_b{args.batch_size}_lr{lr_str}_{norm_str}'


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default=None,
                        help='Model directory (e.g., $ATTN_ROOT/internvl3_5_8b). '
                             'Auto-detects h5 files and sets output dir.')
    parser.add_argument('--h5_paths', nargs='+', default=None,
                        help='Explicit h5 paths (overrides --model-dir)')
    parser.add_argument('--output_dir', default=None,
                        help='Explicit output dir (overrides --model-dir)')
    parser.add_argument('--phases',    nargs='+', default=PHASES,    choices=PHASES)
    parser.add_argument('--emb_types', nargs='+', default=EMB_TYPES, choices=EMB_TYPES)
    parser.add_argument('--start_layer', type=int, default=0)
    parser.add_argument('--end_layer',   type=int, default=None)
    parser.add_argument('--gpu_ids',           type=int, nargs='+', default=[0, 1, 2, 3])
    parser.add_argument('--processes_per_gpu', type=int, default=2)
    parser.add_argument('--arch', default='standard', choices=['standard', 'direct'],
                        help='standard: both projectors; direct: image→head_dim only, no head projector')
    parser.add_argument('--no_norm', action='store_true',
                        help='Disable L2 normalization of projections')
    parser.add_argument('--hidden_dim',     type=int,   default=512)
    parser.add_argument('--num_mlp_layers', type=int,   default=3)
    parser.add_argument('--projection_dim', type=int,   default=256)
    parser.add_argument('--batch_size',   type=int,   default=128)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--temperature',  type=float, default=0.07)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--seed',        type=int,   default=42)
    parser.add_argument('--lr_schedule', default='cosine', choices=['cosine', 'flat'],
                        help='cosine: CosineAnnealingLR (default); flat: constant lr')
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Linear warmup epochs before cosine decay (0=no warmup)')
    parser.add_argument('--verbose',     action='store_true')

    args = parser.parse_args()

    # Resolve paths from --model-dir when explicit paths not given
    if args.model_dir and not args.h5_paths:
        args.h5_paths = _find_h5_files(args.model_dir)
        print(f'Auto-detected {len(args.h5_paths)} h5 files from {args.model_dir}/coco_mcq/')
    if args.h5_paths is None:
        parser.error('Either --model-dir or --h5_paths is required')
    if args.model_dir and not args.output_dir:
        config_name = _make_config_name(args)
        args.output_dir = os.path.join(args.model_dir, 'retriever', config_name)
        print(f'Output dir: {args.output_dir}')
    if args.output_dir is None:
        parser.error('Either --model-dir or --output_dir is required')

    main(args)
