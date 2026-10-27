"""
VHS head output + attention score extraction — family-agnostic via ModelBackend.

Saves per sample:
    sample_{id}/
        attrs: sample_id, hs, pos_idx, needle, response, correct
        image_embeddings/
            patch_mean: (n_images, D_vit)
            proj_mean:  (n_images, D_lm)
        cattok/layer_X:     (1, num_heads, head_dim)
        generation/layer_X: (1, num_heads, head_dim)
        cattok_attn_scores/layer_X:     (num_heads, n_images) float32
        generation_attn_scores/layer_X: (num_heads, n_images) float32

Usage:
    CUDA_VISIBLE_DEVICES=1 python data/save_vhs_head_outputs.py \
        --model internvl-8b --hs 10 20
    CUDA_VISIBLE_DEVICES=3 python data/save_vhs_head_outputs.py \
        --model qwen3vl-8b --hs 10
    CUDA_VISIBLE_DEVICES=2 python data/save_vhs_head_outputs.py \
        --model llava-ov-7b --hs 5 10
"""
import os, sys, json, re, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

from utils.model_backend import ModelBackend
from utils.model_utils import COCO_ROOT, DATA_ROOT, ATTN_ROOT, MODEL_CONFIGS, parse_yes_no

VHS_DATA_ROOT = os.path.join(DATA_ROOT, 'single_needle', 'VHs_large')


def uniquify_ids(samples):
    """In the multi_needle json each id appears twice (yes/no variants, e.g. 1,000 samples = 500 ids x 2)
    with different pos_image sets; a '#k' suffix based on json order makes them unique.
    When ids are already unique (single_needle) nothing changes (no-op).
    As long as extraction and evaluation apply this to the same json order, the uids match deterministically."""
    seen = {}
    for s in samples:
        k = seen.get(s['id'], 0)
        seen[s['id']] = k + 1
        if k:
            s['id'] = f"{s['id']}#{k}"
    return samples


def build_haystack_map(samples):
    """Sequential seed-42 shuffle, identical to the one used for the released hs2-20 H5 files and by
    every reader of them. A single rng is consumed over the whole json order, so the map must be
    built for the full json before any sample is skipped on resume; otherwise the
    alignment with the stored pos_idx attributes is lost."""
    import random
    rng = random.Random(42)
    result = {}
    for s in samples:
        hs_list = list(s['pos_image']) + list(s['neg_image'])
        rng.shuffle(hs_list)
        result[s['id']] = hs_list
    return result


# ── Attention score computation (shared) ───────────────────────────────────────

def compute_img_attn_scores(q_dict, k_dict, num_layers, num_heads, kv_groups,
                            scale, img_ranges, n_imgs):
    result = {}
    for li in range(num_layers):
        if li not in q_dict or li not in k_dict:
            continue
        q = q_dict[li].cpu()
        k = k_dict[li].cpu()
        k_exp = k.repeat_interleave(kv_groups, dim=0)
        scores = torch.einsum('hd,hsd->hs', q, k_exp) * scale
        attn_w = torch.softmax(scores, dim=-1)
        img_scores = torch.zeros(num_heads, n_imgs)
        for img_i, (start, end) in enumerate(img_ranges):
            e = min(end + 1, attn_w.shape[1])
            if start < attn_w.shape[1]:
                img_scores[:, img_i] = attn_w[:, start:e].sum(dim=1)
        result[li] = img_scores
    return result


# ── HDF5 save (shared) ─────────────────────────────────────────────────────────

def save_sample(hdf5, sid, hs, pos_idx, needle, response, correct,
                head_outputs, attn_scores, image_embeddings, pos_idxs=None):
    grp_key = f"sample_{sid}"
    if grp_key in hdf5:
        del hdf5[grp_key]
    grp = hdf5.create_group(grp_key)
    grp.attrs['sample_id'] = sid
    grp.attrs['hs'] = hs
    grp.attrs['pos_idx'] = pos_idx
    if pos_idxs is not None:
        grp.attrs['pos_idxs'] = pos_idxs
    grp.attrs['needle'] = needle
    grp.attrs['response'] = response
    grp.attrs['correct'] = int(correct)

    eg = grp.create_group('image_embeddings')
    for key in ('patch_mean', 'proj_mean'):
        arr = image_embeddings.get(key)
        if arr is not None:
            eg.create_dataset(key, data=arr.numpy(), compression='gzip', compression_opts=4)

    for phase in ('cattok', 'generation'):
        if phase in head_outputs:
            pg = grp.create_group(phase)
            for li, tensor in head_outputs[phase].items():
                pg.create_dataset(f'layer_{li}', data=tensor.float().numpy(),
                                  compression='gzip', compression_opts=4)

    for phase in ('cattok', 'generation'):
        if phase in attn_scores:
            ag = grp.create_group(f'{phase}_attn_scores')
            for li, s in attn_scores[phase].items():
                ag.create_dataset(f'layer_{li}', data=s.numpy(),
                                  compression='gzip', compression_opts=4)


# ── Unified extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_all(backend: ModelBackend, haystack_rels, question, needle_text):
    """Run one forward+decode, capture head outputs and attention scores.

    Uses backend.generate so the generation path matches the family's natural
    API (model.chat for InternVL, model.generate for Qwen3-VL / LLaVA-OV).
    Hook outputs are captured on backend.layers during that forward pass.
    """
    num_layers = backend.num_layers
    num_heads = backend.num_heads
    num_kv_heads = backend.num_kv_heads
    head_dim = backend.head_dim
    kv_groups = backend.kv_groups
    scale = backend.scale

    pil_images = [Image.open(os.path.join(COCO_ROOT, r)).convert('RGB')
                  for r in haystack_rels]
    n_imgs = len(haystack_rels)

    inputs = backend.build_inputs(pil_images, question)
    ids_list = inputs['ids_list']
    img_ranges = inputs['image_ranges']
    needle_pos = backend.find_token_pos(ids_list, needle_text,
                                        strip_article=True, fallback='middle')

    # Image embeddings (vision tower pass — separate from the generate path).
    embs = backend.extract_image_embeddings(inputs)
    image_embeddings = {
        'patch_mean': embs.get('patch_mean'),
        'proj_mean': embs.get('proj_mean'),
    }

    # Hooks on backend.layers[li].self_attn.{o_proj.input, q_proj.output, k_proj.output}.
    head_cattok, head_gen = {}, {}
    q_cat, k_cat, q_gen = {}, {}, {}
    hooks = []

    for li in range(num_layers):
        attn = backend.layers[li].self_attn

        def make_opre(layer_idx):
            def fn(_m, input):
                pre = input[0]
                if pre is None or pre.ndim != 3:
                    return
                bsz, seq, h = pre.shape
                if h != num_heads * head_dim:
                    return
                pre = pre.reshape(bsz, seq, num_heads, head_dim)
                if seq > 1:
                    if layer_idx not in head_cattok and needle_pos < seq:
                        head_cattok[layer_idx] = pre[0, needle_pos:needle_pos + 1, :, :].detach().cpu()
                elif seq == 1:
                    if layer_idx not in head_gen:
                        head_gen[layer_idx] = pre[0].detach().cpu()
            return fn
        hooks.append(attn.o_proj.register_forward_pre_hook(make_opre(li)))

        def make_q(layer_idx):
            def fn(_m, _a, output):
                x = output[0] if isinstance(output, tuple) else output
                x = x.detach().float()
                if x.dim() == 3:
                    bsz, seq, _ = x.shape
                    if seq > 1:
                        x = x.reshape(bsz, seq, num_heads, head_dim).permute(0, 2, 1, 3)
                        if needle_pos < seq:
                            q_cat[layer_idx] = x[0, :, needle_pos, :]
                    elif seq == 1:
                        if layer_idx not in q_gen:
                            x = x.reshape(bsz, 1, num_heads, head_dim).permute(0, 2, 1, 3)
                            q_gen[layer_idx] = x[0, :, 0, :]
            return fn
        hooks.append(attn.q_proj.register_forward_hook(make_q(li)))

        def make_k(layer_idx):
            def fn(_m, _a, output):
                x = output[0] if isinstance(output, tuple) else output
                x = x.detach().float()
                if x.dim() == 3 and x.shape[1] > 1:
                    x = x.reshape(1, x.shape[1], num_kv_heads, head_dim).permute(0, 2, 1, 3)
                    k_cat[layer_idx] = x[0]
            return fn
        hooks.append(attn.k_proj.register_forward_hook(make_k(li)))

    try:
        gen = backend.generate(inputs, max_new_tokens=20,
                               capture_first_logits=False)
    finally:
        for h in hooks:
            h.remove()
    response = gen['response']

    torch.cuda.empty_cache()

    cattok_attn = compute_img_attn_scores(q_cat, k_cat, num_layers, num_heads,
                                          kv_groups, scale, img_ranges, n_imgs)
    gen_attn = compute_img_attn_scores(q_gen, k_cat, num_layers, num_heads,
                                       kv_groups, scale, img_ranges, n_imgs)

    return response, \
           {'cattok': head_cattok, 'generation': head_gen}, \
           {'cattok': cattok_attn, 'generation': gen_attn}, \
           image_embeddings


# ── VHS needle phrase extraction ──────────────────────────────────────────────

def get_needle_phrase(sample, family):
    question = sample['conversations'][0]['value']
    needle = sample.get('needle', '')
    if family == 'qwen3vl':
        m = re.search(r'image with (?:a |an )?(.+?),', question)
        return m.group(1).strip() if m else needle
    return needle


# ── Run per haystack size ─────────────────────────────────────────────────────

def run_hs(backend, cfg, hs, out_root, limit=None, data_root=VHS_DATA_ROOT):
    json_path = os.path.join(data_root, f'visual_haystack_{hs}.json')
    if not os.path.exists(json_path):
        print(f"[SKIP] hs={hs} not found"); return

    samples = uniquify_ids(json.load(open(json_path)))
    # Build the map from the full json before applying --limit so the rng order matches the readers.
    haystack_map = build_haystack_map(samples)
    if limit:
        samples = samples[:limit]
    print(f"\n── hs={hs}: {len(samples)} samples ──")

    hdf5_path = os.path.join(out_root, f'vhs_hs{hs}.h5')
    done_keys = set()
    if os.path.exists(hdf5_path):
        try:
            with h5py.File(hdf5_path, 'r') as f:
                done_keys = set(f.keys())
        except OSError:
            os.remove(hdf5_path)
    print(f"  [resume] {len(done_keys)} already saved")

    h5f = h5py.File(hdf5_path, 'a')
    try:
        for sample in tqdm(samples, desc=f'hs={hs}'):
            sid = sample['id']
            grp_key = f"sample_{sid}"
            if grp_key in done_keys:
                continue
            haystack_rels = haystack_map[sid]
            # multi-needle has two or more pos_images: store all indices as an attribute and
            # keep pos_idx as the first one (single-needle compatibility)
            pos_idxs = [haystack_rels.index(p) for p in sample['pos_image']]
            pos_idx = pos_idxs[0]
            question = sample['conversations'][0]['value']
            gt = sample['conversations'][1]['value'].strip().lower()
            needle_text = get_needle_phrase(sample, backend.family)
            try:
                response, head_outputs, attn_scores, embs = extract_all(
                    backend, haystack_rels, question, needle_text)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  [OOM] sid={sid}"); continue
            except Exception as e:
                print(f"  [ERR] sid={sid}: {e}"); continue

            pred = parse_yes_no(response)
            correct = (pred == gt) if pred is not None else False
            save_sample(h5f, sid, hs, pos_idx,
                        sample.get('needle', ''), response, correct,
                        head_outputs, attn_scores, embs, pos_idxs=pos_idxs)
            h5f.flush()
    finally:
        h5f.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    cfg = MODEL_CONFIGS[args.model]
    out_root = args.outputs_dir or os.path.join(
        ATTN_ROOT, cfg['model_dir'], 'vhs')
    os.makedirs(out_root, exist_ok=True)

    backend = ModelBackend(args.model)
    print(f"Model {args.model} loaded (family={backend.family})  "
          f"{backend.num_layers}L x {backend.num_heads}H")

    for hs in args.hs:
        run_hs(backend, cfg, hs, out_root, limit=args.limit,
               data_root=args.data_root)

    print("\nDone.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--hs', nargs='+', type=int, default=[2, 3, 5, 10, 20])
    parser.add_argument('--outputs-dir', default=None,
                        help='Default: $ATTN_ROOT/{model_dir}/vhs')
    parser.add_argument('--limit', type=int, default=None,
                        help='preflight: only the first N samples (the shuffle map is still built on the full json)')
    parser.add_argument('--data-root', default=VHS_DATA_ROOT,
                        help='VHs json directory (for variants such as multi_needle)')
    args = parser.parse_args()
    main(args)
