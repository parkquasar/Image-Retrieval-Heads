"""
External retriever baselines on Visual Haystacks (single-needle, hs=10).

For each VHs sample the haystack is the sample's own pos + neg images
(10 images at hs=10; COCO fillers are drawn only if a sample has fewer).
The retriever scores every image against the anchor-object phrase
(the VHs ``needle`` field, e.g. "a mouse") and we count R@1 = argmax hits the
target image.  Haystack order is irrelevant for R@1.

Retrievers
  clip    openai/clip-vit-large-patch14
  siglip  google/siglip-so400m-patch14-384

Prompting
  ensemble  (default) OpenAI 80-template ensemble: the text embedding is the
            mean of the 80 templated prompts, L2-normalised (Radford et al. 2021).
  single    the single prompt "a photo of {needle}".

Usage
  CUDA_VISIBLE_DEVICES=0 python eval/eval_vhs_external_retrievers.py --retriever clip
  CUDA_VISIBLE_DEVICES=0 python eval/eval_vhs_external_retrievers.py --retriever siglip --prompt single

Output: $ATTN_ROOT/analysis/external_retrievers/hs{hs}_{retriever}_{prompt}.jsonl (resume-safe)
"""
import os
import sys
import json
import random
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor

from utils.model_utils import DATA_ROOT, COCO_ROOT, ATTN_ROOT

RETRIEVERS = {
    'clip':   'openai/clip-vit-large-patch14',
    'siglip': 'google/siglip-so400m-patch14-384',
}

# The 80 prompt templates of the OpenAI CLIP ImageNet zero-shot evaluation.
OPENAI_TEMPLATES = [
    'a bad photo of a {}.', 'a photo of many {}.', 'a sculpture of a {}.', 'a photo of the hard to see {}.',
    'a low resolution photo of the {}.', 'a rendering of a {}.', 'graffiti of a {}.', 'a bad photo of the {}.',
    'a cropped photo of the {}.', 'a tattoo of a {}.', 'the embroidered {}.', 'a photo of a hard to see {}.',
    'a bright photo of a {}.', 'a photo of a clean {}.', 'a photo of a dirty {}.', 'a dark photo of the {}.',
    'a drawing of a {}.', 'a photo of my {}.', 'the plastic {}.', 'a photo of the cool {}.',
    'a close-up photo of a {}.', 'a black and white photo of the {}.', 'a painting of the {}.', 'a painting of a {}.',
    'a pixelated photo of the {}.', 'a sculpture of the {}.', 'a bright photo of the {}.', 'a cropped photo of a {}.',
    'a plastic {}.', 'a photo of the dirty {}.', 'a jpeg corrupted photo of a {}.', 'a blurry photo of the {}.',
    'a photo of the {}.', 'a good photo of the {}.', 'a rendering of the {}.', 'a {} in a video game.',
    'a photo of one {}.', 'a doodle of a {}.', 'a close-up photo of the {}.', 'a photo of a {}.',
    'the origami {}.', 'the {} in a video game.', 'a sketch of a {}.', 'a doodle of the {}.',
    'a origami {}.', 'a low resolution photo of a {}.', 'the toy {}.', 'a rendition of the {}.',
    'a photo of the clean {}.', 'a photo of a large {}.', 'a rendition of a {}.', 'a photo of a nice {}.',
    'a photo of a weird {}.', 'a blurry photo of a {}.', 'a cartoon {}.', 'art of a {}.',
    'a sketch of the {}.', 'a embroidered {}.', 'a pixelated photo of a {}.', 'itap of the {}.',
    'a jpeg corrupted photo of the {}.', 'a good photo of a {}.', 'a plushie {}.', 'a photo of the nice {}.',
    'a photo of the small {}.', 'a photo of the weird {}.', 'the cartoon {}.', 'art of the {}.',
    'a drawing of the {}.', 'a photo of the large {}.', 'a black and white photo of a {}.', 'the plushie {}.',
    'a dark photo of a {}.', 'itap of a {}.', 'graffiti of the {}.', 'a toy {}.',
    'itap of my {}.', 'a photo of a cool {}.', 'a photo of a small {}.', 'a tattoo of the {}.',
]
assert len(OPENAI_TEMPLATES) == 80


def build_coco_pool(split_dir):
    """COCO images not used by any VHs sample; fillers for haystacks shorter than hs."""
    used = set()
    for fn in os.listdir(split_dir):
        if fn.endswith('.json'):
            for s in json.load(open(os.path.join(split_dir, fn))):
                used.update(s.get('pos_image', []))
                used.update(s.get('neg_image', []))
    pool = []
    for split in ('train2017', 'val2017'):
        d = os.path.join(COCO_ROOT, split)
        if os.path.isdir(d):
            pool.extend(f'{split}/{fn}' for fn in sorted(os.listdir(d)) if f'{split}/{fn}' not in used)
    return pool


def strip_article(phrase):
    """'a mouse' -> 'mouse' so that templates such as 'a photo of a {}.' read naturally."""
    for art in ('a ', 'an ', 'the '):
        if phrase.lower().startswith(art):
            return phrase[len(art):]
    return phrase


def _as_tensor(out):
    """transformers>=5 returns a ModelOutput from get_*_features for some models."""
    if torch.is_tensor(out):
        return out
    for attr in ('pooler_output', 'text_embeds', 'image_embeds', 'last_hidden_state'):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
    return out[0]


@torch.no_grad()
def text_embedding(model, processor, needle, prompt_mode, device):
    if prompt_mode == 'single':
        texts = [f'a photo of {needle}']
    else:
        texts = [t.format(strip_article(needle)) for t in OPENAI_TEMPLATES]
    inputs = processor(text=texts, return_tensors='pt', padding='max_length' if 'siglip' in model.name_or_path else True,
                       truncation=True).to(device)
    emb = _as_tensor(model.get_text_features(**inputs))
    emb = torch.nn.functional.normalize(emb, dim=-1)
    emb = torch.nn.functional.normalize(emb.mean(0, keepdim=True), dim=-1)
    return emb  # (1, d)


@torch.no_grad()
def image_embeddings(model, processor, pil_images, device):
    inputs = processor(images=pil_images, return_tensors='pt').to(device)
    emb = _as_tensor(model.get_image_features(**inputs))
    return torch.nn.functional.normalize(emb, dim=-1)  # (n, d)


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    split_dir = os.path.join(DATA_ROOT, 'single_needle', 'VHs_large')
    samples = json.load(open(os.path.join(split_dir, f'visual_haystack_{args.hs}.json')))
    coco_pool = build_coco_pool(split_dir)
    rng = random.Random(42)
    haystacks = {}
    for s in samples:
        hs = list(s['pos_image']) + list(s['neg_image'])
        if len(hs) < args.hs:
            hs += rng.sample(coco_pool, args.hs - len(hs))
        rng.shuffle(hs)
        haystacks[s['id']] = hs

    out_dir = args.output_dir or os.path.join(ATTN_ROOT, 'analysis', 'external_retrievers')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'hs{args.hs}_{args.retriever}_{args.prompt}.jsonl')
    done = set()
    if os.path.exists(out_path):
        done = {json.loads(l)['id'] for l in open(out_path)}
    print(f'{len(done)} samples already done -> {out_path}')

    model_id = RETRIEVERS[args.retriever]
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_id)

    hits = total = 0
    with open(out_path, 'a') as fout:
        for s in tqdm(samples, desc=f'{args.retriever}/{args.prompt} hs={args.hs}'):
            if s['id'] in done:
                continue
            haystack = haystacks[s['id']]
            pos_rel = s['pos_image'][0]
            needle = s['needle']
            pil_images = [Image.open(os.path.join(COCO_ROOT, rel)).convert('RGB') for rel in haystack]
            t = text_embedding(model, processor, needle, args.prompt, device)
            v = image_embeddings(model, processor, pil_images, device)
            scores = (v @ t.T).squeeze(1).float().cpu().tolist()
            pred_idx = max(range(len(scores)), key=scores.__getitem__)
            hit = haystack[pred_idx] == pos_rel
            hits += int(hit); total += 1
            fout.write(json.dumps({'id': s['id'], 'needle': needle, 'pos_image': pos_rel,
                                   'pred_idx': pred_idx, 'pred_image': haystack[pred_idx], 'hit': hit,
                                   'scores': scores, 'haystack': haystack}) + '\n')
            fout.flush()
    rows = [json.loads(l) for l in open(out_path)]
    r1 = sum(r['hit'] for r in rows) / len(rows)
    print(f'{args.retriever} ({args.prompt}) hs={args.hs}: R@1 = {r1:.3f}  (n={len(rows)})')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--retriever', choices=list(RETRIEVERS), required=True)
    ap.add_argument('--prompt', choices=['ensemble', 'single'], default='ensemble')
    ap.add_argument('--hs', type=int, default=10)
    ap.add_argument('--output-dir', default=None)
    main(ap.parse_args())
