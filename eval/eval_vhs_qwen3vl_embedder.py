"""
VHS hs=10 retrieval R@1 using Qwen3-VL-Embedding — analog to the archived
CLIP/SigLIP experiment.

For each of 1000 VHs-large hs=10 samples:
  1. Build haystack (pos + neg + COCO fillers, seed=42 — matches existing protocol)
  2. Embed query text  (default: "a photo of {needle}")
  3. Embed 10 haystack images (one pass, one image per sequence)
  4. Cosine sim = query · images.T (embeddings are L2-normalized)
  5. R@1 = (argmax == pos_image index in haystack)

Output JSONL: $ATTN_ROOT/analysis/qwen3vl_embedder_retrieval/hs10_{size}.jsonl

Usage:
  CUDA_VISIBLE_DEVICES=2 python eval/eval_vhs_qwen3vl_embedder.py \
      --model-size 8B --instruction-mode default
"""

import os
import sys
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Union

import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLPreTrainedModel, Qwen3VLModel, Qwen3VLConfig,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.modeling_outputs import ModelOutput


from utils.model_utils import DATA_ROOT, COCO_ROOT, ATTN_ROOT
OUT_DIR = os.path.join(ATTN_ROOT, 'analysis', 'qwen3vl_embedder_retrieval')

MODEL_IDS = {
    '2B': 'Qwen/Qwen3-VL-Embedding-2B',
    '8B': 'Qwen/Qwen3-VL-Embedding-8B',
}

# Embedder constants (mirrors Qwen3-VL-Embedding reference implementation).
MAX_LENGTH = 8192
IMAGE_BASE_FACTOR = 16
IMAGE_FACTOR = IMAGE_BASE_FACTOR * 2
MIN_PIXELS = 4 * IMAGE_FACTOR * IMAGE_FACTOR
MAX_PIXELS = 1800 * IMAGE_FACTOR * IMAGE_FACTOR

# Instruction strings (system-level task description).
# Official usage is asymmetric: the QUERY carries a task instruction while the
# DOCUMENT side uses the default "Represent the user's input."
DEFAULT_DOC_INSTRUCTION = "Represent the user's input."
QUERY_INSTRUCTION = "Retrieve images or text relevant to the user's query."

# Query text templates. {needle} and {target} are substituted per sample.
QUERY_MODES = {
    # Primary: natural retrieval phrasing, matches VHs question's internal phrasing.
    'natural':      'an image with {needle}',
    # Full VHs question tests compositional understanding (target is distractor).
    'vhs-question': 'For the image with {needle}, is there {target}?',
    # Apples-to-apples comparison with the archived CLIP/SigLIP experiment.
    'clip':         'a photo of {needle}',
}


# ── Model wrapper ──────────────────────────────────────────────────────────────

@dataclass
class _EmbeddingOutput(ModelOutput):
    last_hidden_state: Optional[torch.FloatTensor] = None
    attention_mask:    Optional[torch.Tensor]      = None


class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
    """Port of the reference Qwen3VLForEmbedding wrapper from the Qwen3-VL-Embedding
    repository (Alibaba Qwen team, Apache License 2.0). See THIRD_PARTY_NOTICES.md."""
    config: Qwen3VLConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = Qwen3VLModel(config)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, pixel_values=None,
                pixel_values_videos=None, image_grid_thw=None,
                video_grid_thw=None, **kwargs):
        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )
        return _EmbeddingOutput(
            last_hidden_state=outputs.last_hidden_state,
            attention_mask=attention_mask,
        )


def _format_conversation(text: Optional[str], image: Optional[Image.Image],
                          instruction: str) -> List[Dict]:
    content: List[Dict] = []
    conv = [
        {"role": "system", "content": [{"type": "text", "text": instruction}]},
        {"role": "user",   "content": content},
    ]
    if image is not None:
        content.append({
            'type': 'image', 'image': image,
            'min_pixels': MIN_PIXELS, 'max_pixels': MAX_PIXELS,
        })
    if text is not None:
        content.append({'type': 'text', 'text': text})
    if not content:
        content.append({'type': 'text', 'text': 'NULL'})
    return conv


def _smart_resize_image(img: Image.Image, factor: int = 32,
                        min_pixels: int = MIN_PIXELS, max_pixels: int = MAX_PIXELS) -> Image.Image:
    """Resize so both sides are multiples of `factor` and the pixel count lies in
    [min_pixels, max_pixels], keeping the aspect ratio (the qwen_vl_utils rule)."""
    import math
    h, w = img.height, img.width
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return img.convert('RGB').resize((w_bar, h_bar), Image.BICUBIC)


@torch.no_grad()
def _embed(model, processor, items: List[Dict[str, Any]], device) -> torch.Tensor:
    """Each item carries its own 'instruction' (falls back to DEFAULT_DOC_INSTRUCTION)."""
    conversations = [
        _format_conversation(
            it.get('text'), it.get('image'),
            it.get('instruction', DEFAULT_DOC_INSTRUCTION))
        for it in items
    ]
    text = processor.apply_chat_template(
        conversations, add_generation_prompt=True, tokenize=False)
    # Resize images ourselves (Qwen3-VL patch 16 x merge 2 = factor 32) so that the
    # result does not depend on the installed qwen_vl_utils version.
    images = [_smart_resize_image(part['image'])
              for c in conversations for m in c
              for part in (m['content'] if isinstance(m['content'], list) else [])
              if part.get('type') == 'image'] or None
    video_inputs, video_kwargs = None, {}
    n_img = len(images) if images else 0

    if video_inputs is not None:
        videos, video_metadata = zip(*video_inputs)
        videos = list(videos); video_metadata = list(video_metadata)
    else:
        videos, video_metadata = None, None

    inputs = processor(
        text=text, images=images, videos=videos, video_metadata=video_metadata,
        truncation=True, max_length=MAX_LENGTH, padding=True, do_resize=False,
        return_tensors='pt', **video_kwargs,
    )
    if n_img and 'pixel_values' not in inputs:
        raise RuntimeError('processor produced no pixel_values for an image input')
    inputs = {k: v.to(device) for k, v in inputs.items()}
    outputs = model(**inputs)
    last_h = outputs.last_hidden_state
    attn_mask = inputs['attention_mask']

    # Pool last valid position.
    flipped = attn_mask.flip(dims=[1])
    last_one_pos = flipped.argmax(dim=1)
    col = attn_mask.shape[1] - last_one_pos - 1
    row = torch.arange(last_h.shape[0], device=last_h.device)
    emb = last_h[row, col]

    return F.normalize(emb.float(), p=2, dim=-1)


# ── Haystack construction (matches archived CLIP retrieval protocol) ───────────

def build_coco_pool(split_dir):
    all_used = set()
    for fn in os.listdir(split_dir):
        if fn.endswith('.json'):
            data = json.load(open(os.path.join(split_dir, fn)))
            for s in data:
                all_used.update(s.get('pos_image', []))
                all_used.update(s.get('neg_image', []))
    pool = []
    for split in ['train2017', 'val2017']:
        d = os.path.join(COCO_ROOT, split)
        if os.path.isdir(d):
            for fn in os.listdir(d):
                rel = f'{split}/{fn}'
                if rel not in all_used:
                    pool.append(rel)
    return pool


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-size', default='8B', choices=list(MODEL_IDS.keys()))
    parser.add_argument('--hs', type=int, default=10)
    parser.add_argument('--query-mode', default='natural',
                        choices=list(QUERY_MODES.keys()),
                        help='natural = "an image with {needle}" (default); '
                             'vhs-question = full VHs question; '
                             'clip = "a photo of {needle}" (CLIP apples-to-apples).')
    parser.add_argument('--attn', default='flash_attention_2',
                        choices=['flash_attention_2', 'sdpa', 'eager'])
    parser.add_argument('--output-dir', default=OUT_DIR)
    args = parser.parse_args()

    model_id = MODEL_IDS[args.model_size]
    query_template = QUERY_MODES[args.query_mode]

    # Deterministic haystack (same seed as archived CLIP script).
    split_dir = os.path.join(DATA_ROOT, 'single_needle', 'VHs_large')
    json_path = os.path.join(split_dir, f'visual_haystack_{args.hs}.json')
    samples = json.load(open(json_path))
    coco_pool = build_coco_pool(split_dir)

    haystacks: Dict[str, List[str]] = {}
    _rng = random.Random(42)
    for s in samples:
        hs = list(s['pos_image']) + list(s['neg_image'])
        fillers = _rng.sample(coco_pool, max(0, args.hs - len(hs)))
        hs += fillers
        _rng.shuffle(hs)
        haystacks[s['id']] = hs

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(
        args.output_dir,
        f'hs{args.hs}_{args.model_size.lower()}_q-{args.query_mode}.jsonl')

    done_ids = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            done_ids.add(json.loads(line)['id'])
    print(f'[{args.model_size} q-{args.query_mode}]  {len(done_ids)}/{len(samples)} done, '
          f'→ {out_path}')

    print(f'Loading {model_id} ({args.attn})...')
    model = Qwen3VLForEmbedding.from_pretrained(
        model_id, dtype=torch.bfloat16,
        attn_implementation=args.attn, trust_remote_code=True,
    ).eval().cuda()
    processor = Qwen3VLProcessor.from_pretrained(model_id, padding_side='right')
    device = next(model.parameters()).device
    print(f'  query instruction = "{QUERY_INSTRUCTION}"')
    print(f'  doc   instruction = "{DEFAULT_DOC_INSTRUCTION}"  (images)')
    print(f'  query template    = "{query_template}"  (mode={args.query_mode})')

    r1_total = r1_hit = 0
    skipped = 0

    with open(out_path, 'a') as fout:
        for s in tqdm(samples, desc=f'VHs hs={args.hs} Qwen3VL-Emb-{args.model_size}'):
            sid = s['id']
            if sid in done_ids:
                continue
            haystack = haystacks[sid]
            pos_rel = s['pos_image'][0]
            needle = s['needle']
            try:
                pos_idx = haystack.index(pos_rel)
            except ValueError:
                skipped += 1; continue

            # Query: text + task instruction
            target = s.get('target', '')
            try:
                q_text = query_template.format(needle=needle, target=target)
                q_emb = _embed(model, processor,
                               [{'text': q_text, 'instruction': QUERY_INSTRUCTION}],
                               device)  # (1, D)
            except Exception as e:
                print(f'  [ERR:query] {sid}: {e}')
                skipped += 1; continue

            # Documents: 10 images, no instruction override (→ default doc instruction)
            try:
                pil_images = [Image.open(os.path.join(COCO_ROOT, rel)).convert('RGB')
                              for rel in haystack]
                img_emb = _embed(model, processor,
                                 [{'image': img} for img in pil_images],
                                 device)  # (N, D)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                skipped += 1; continue
            except Exception as e:
                print(f'  [ERR:images] {sid}: {e}')
                skipped += 1; continue

            sims = (q_emb @ img_emb.T)[0].float().cpu().numpy()  # (N,)
            pred_idx = int(sims.argmax())
            hit = (pred_idx == pos_idx)
            r1_total += 1
            r1_hit += int(hit)

            fout.write(json.dumps({
                'id':            sid,
                'needle':        needle,
                'gt_image':      pos_rel,
                'pos_idx':       pos_idx,
                'pred_idx':      pred_idx,
                'pred_image':    haystack[pred_idx],
                'hit':           hit,
                'sims':          [float(x) for x in sims],
                'haystack':      haystack,
            }) + '\n')
            fout.flush()
            torch.cuda.empty_cache()

            if r1_total and r1_total % 100 == 0:
                print(f'  [{r1_total}] R@1 = {r1_hit}/{r1_total} = {r1_hit/r1_total:.3f}')

    # Include past completions for final R@1
    all_recs = [json.loads(l) for l in open(out_path)]
    n = len(all_recs)
    if n:
        hit = sum(1 for r in all_recs if r.get('hit'))
        print(f'\n=== Final (incl. resumed) ===')
        print(f'Model: {model_id}')
        print(f'Query instruction: "{QUERY_INSTRUCTION}"')
        print(f'Doc   instruction: "{DEFAULT_DOC_INSTRUCTION}"')
        print(f'Query mode: {args.query_mode}  template="{query_template}"')
        print(f'hs={args.hs}, VHs_large, n={n}')
        print(f'R@1: {hit}/{n} = {hit/n:.3f}')
        print(f'(skipped this run: {skipped})')


if __name__ == '__main__':
    main()
