"""
Unified COCO MCQ head output + attention score + logit-stats extraction.
Family-agnostic via ModelBackend (InternVL, Qwen3-VL, LLaVA-OneVision).

HDF5 schema (same across all families):
    sample_{id}/
        attrs: sample_id, original_id, n_images, target_index,
               target_category, context_length, verified
        image_embeddings/
            patch_mean: (n_images, D_vit)
            proj_mean:  (n_images, D_lm)
        attn_score/
            scores_sum: (n_layers, n_heads, n_images) float16
            scores_max: (n_layers, n_heads, n_images) float16
        cattok/layer_X: (1, num_heads, head_dim)
        ansA/ ansB/ ansC/ ansD/
            attrs: answer_label, response, correct,
                   prob_A, prob_B, prob_C, prob_D, top1_prob, entropy
            generation/layer_X: (1, num_heads, head_dim)

Usage:
    CUDA_VISIBLE_DEVICES=1 python data/save_coco_head_outputs_mcq.py --model internvl-8b
    CUDA_VISIBLE_DEVICES=2 python data/save_coco_head_outputs_mcq.py --model llava-ov-7b
    CUDA_VISIBLE_DEVICES=3 python data/save_coco_head_outputs_mcq.py --model qwen3vl-8b
"""
import os, sys, json, re, argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import h5py
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_backend import ModelBackend
from utils.model_utils import COCO_ROOT, ATTN_ROOT, MODEL_CONFIGS

MCQ_LABELS = ['A', 'B', 'C', 'D']


# ── MCQ helpers ────────────────────────────────────────────────────────────────

def rotate_options(mcq_options, original_answer, new_answer):
    if original_answer == new_answer:
        return dict(mcq_options)
    new_opts = dict(mcq_options)
    new_opts[new_answer] = mcq_options[original_answer]
    new_opts[original_answer] = mcq_options[new_answer]
    return new_opts


def build_question(category, options):
    opts_str = '\n'.join(f'{label}) {options[label]}' for label in MCQ_LABELS)
    return (
        f"Which caption best describes the image above that contains {category}? "
        f"Answer with a single letter (A, B, C, or D).\n"
        f"{opts_str}"
    )


def parse_mcq_response(response):
    m = re.search(r'\b([A-D])\b', response.upper())
    return m.group(1) if m else None


# ── Logit stats ────────────────────────────────────────────────────────────────

def get_logit_stats(logits_last, mcq_token_ids):
    probs = torch.softmax(logits_last.float(), dim=-1)
    entropy = float(-torch.sum(probs * torch.log(probs + 1e-9)))
    top1_prob = float(probs.max())
    stats = {'entropy': entropy, 'top1_prob': top1_prob}
    for label, tid in mcq_token_ids.items():
        stats[f'prob_{label}'] = float(probs[tid])
    return stats


# ── HeadOutputCollector ────────────────────────────────────────────────────────

class HeadOutputCollector:
    """Captures o_proj inputs at `category_pos` (prefill) and at seq=1 (gen first step)."""

    def __init__(self, layers, num_layers, num_heads, head_dim):
        self.layers = layers
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.expected_hidden = num_heads * head_dim
        self.head_outputs = defaultdict(dict)
        self.category_pos = None
        self.hooks = []

    def register_hooks(self):
        for li in range(self.num_layers):
            o_proj = self.layers[li].self_attn.o_proj
            self.hooks.append(o_proj.register_forward_pre_hook(self._make_hook(li)))

    def _make_hook(self, layer_idx):
        def hook_fn(_m, input):
            pre = input[0]
            if pre is None or pre.ndim != 3:
                return
            B, S, H = pre.shape
            if H != self.expected_hidden:
                return
            pre = pre.reshape(B, S, self.num_heads, self.head_dim)
            if S > 1:
                if layer_idx not in self.head_outputs['cattok']:
                    pos = self.category_pos
                    if pos is not None and pos < S:
                        self.head_outputs['cattok'][layer_idx] = \
                            pre[0, pos:pos + 1, :, :].detach().cpu()
            elif S == 1:
                if layer_idx not in self.head_outputs['generation']:
                    self.head_outputs['generation'][layer_idx] = \
                        pre[0].detach().cpu()
        return hook_fn

    def reset(self):
        self.head_outputs = defaultdict(dict)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def get_outputs(self):
        return {k: dict(v) for k, v in self.head_outputs.items()}


# ── AttnScoreCollector ─────────────────────────────────────────────────────────

class AttnScoreCollector:
    """Hooks q_proj + k_proj during prefill to compute per-head per-image attention."""

    def __init__(self, layers, num_layers, num_heads, num_kv_heads, head_dim,
                 model=None):
        self.model = model
        self.layers = layers
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.kv_groups = num_heads // num_kv_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.q_outs = {}
        self.k_outs = {}
        self.category_pos = None
        self.img_ranges = None
        self.hooks = []
        self.collected = False

    def register_hooks(self):
        self._assert_full_attention()
        for li in range(self.num_layers):
            attn = self.layers[li].self_attn

            def make_q(layer_idx):
                def fn(_m, _a, output):
                    if self.collected: return
                    x = output[0] if isinstance(output, tuple) else output
                    x = x.detach().float()
                    if x.dim() == 3 and x.shape[1] > 1:
                        pos = self.category_pos
                        if pos is not None and pos < x.shape[1]:
                            x = x.reshape(1, x.shape[1], self.num_heads, self.head_dim) \
                                 .permute(0, 2, 1, 3)
                            self.q_outs[layer_idx] = x[0, :, pos, :]
                return fn

            def make_k(layer_idx):
                def fn(_m, _a, output):
                    if self.collected: return
                    x = output[0] if isinstance(output, tuple) else output
                    x = x.detach().float()
                    if x.dim() == 3 and x.shape[1] > 1:
                        x = x.reshape(1, x.shape[1], self.num_kv_heads, self.head_dim) \
                             .permute(0, 2, 1, 3)
                        self.k_outs[layer_idx] = x[0]
                return fn

            self.hooks.append(attn.q_proj.register_forward_hook(make_q(li)))
            self.hooks.append(attn.k_proj.register_forward_hook(make_k(li)))

    def _assert_full_attention(self):
        """attention mass 재구성은 full attention을 전제한다.

        compute_scores는 q(cattok 행)와 전체 시퀀스 k의 내적에 softmax를 걸어
        per-image attention mass를 복원한다. sliding-window attention을 쓰는
        모델에서는 window 밖 위치가 실제로는 마스킹되어 0인데, 마스크 없이
        softmax를 걸면 그 위치들에 **실재하지 않는 mass**가 배분된다. 로그에는
        아무 이상이 없어 그대로 통과한다.

        예: Gemma-3는 48층 중 40층이 sliding(window 1024)이고 full은 8층뿐이다
        (layer_types가 5:1 패턴). ICIR-MCQ 프롬프트가 약 9,000 토큰이라 sliding
        층에서 cattok은 후보 이미지의 약 90%를 물리적으로 볼 수 없다. 이 경우
        640개 head의 pointer score가 전부 허수가 된다.

        현재 지원 모델(InternVL 3.5, Qwen3-VL, LLaVA-OneVision, Pixtral)은 전부
        full attention이므로 기존 결과에는 영향이 없다. sliding-window 모델을
        추가하려면 compute_scores에 window 마스크를 먼저 구현해야 한다.
        """
        if self.model is None:
            print('  [WARN] full-attention 가드: model 미전달로 검사 생략')
            return
        cfg = getattr(self.model, 'config', None)
        tcfg = getattr(cfg, 'text_config', cfg) if cfg is not None else None
        if tcfg is None:
            print('  [WARN] full-attention 가드: config 없음으로 검사 생략')
            return
        types = getattr(tcfg, 'layer_types', None)
        sw = getattr(tcfg, 'sliding_window', None)
        bad = []
        if types:
            bad = [i for i, t in enumerate(types) if 'sliding' in str(t)]
        elif sw:
            bad = list(range(getattr(tcfg, 'num_hidden_layers', 0)))
        if bad:
            raise RuntimeError(
                f'[FATAL] sliding-window attention 감지: {len(bad)}/{len(types) if types else "?"} 층 '
                f'(window={sw}). compute_scores가 마스크를 적용하지 않으므로 '
                f'이 층들의 per-image attention mass는 실재하지 않는 값이 됩니다. '
                f'window 마스크를 구현하기 전까지 이 모델로 추출하지 마세요. '
                f'(full-attention 층 인덱스: '
                f'{[i for i, t in enumerate(types) if "sliding" not in str(t)] if types else "없음"})')

    def compute_scores(self, n_images):
        scores_sum = np.zeros((self.num_layers, self.num_heads, n_images), dtype=np.float32)
        scores_max = np.zeros((self.num_layers, self.num_heads, n_images), dtype=np.float32)
        for li in range(self.num_layers):
            if li not in self.q_outs or li not in self.k_outs:
                continue
            q = self.q_outs[li]
            k_exp = self.k_outs[li].repeat_interleave(self.kv_groups, dim=0)
            attn_w = torch.softmax(
                torch.einsum('hd,hsd->hs', q, k_exp) * self.scale, dim=-1)
            seq_len = attn_w.shape[1]
            for img_i, (start, end) in enumerate(self.img_ranges):
                e = min(end + 1, seq_len)
                if start < seq_len:
                    patch_w = attn_w[:, start:e]
                    scores_sum[li, :, img_i] = patch_w.sum(dim=-1).cpu().numpy()
                    scores_max[li, :, img_i] = patch_w.max(dim=-1).values.cpu().numpy()
        self.collected = True
        return scores_sum, scores_max

    def reset_for_variant(self):
        self.q_outs.clear()
        self.k_outs.clear()
        self.collected = False

    def reset(self):
        self.q_outs.clear()
        self.k_outs.clear()
        self.collected = False
        self.category_pos = None
        self.img_ranges = None

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()


# ── Run one MCQ variant ────────────────────────────────────────────────────────

class QAnchorNotFound(Exception):
    """--anchor question 인데 ' {category}' 토큰열을 못 찾은 경우 (샘플 skip)."""


@torch.no_grad()
def run_variant(backend: ModelBackend, pil_images, category, context, options,
                head_collector, attn_collector, mcq_token_ids,
                cached_img_ranges=None, cached_embs=None, anchor='last'):
    """Execute one MCQ variant: build prompt, run forward, collect head outputs."""
    head_collector.reset()
    attn_collector.reset_for_variant()

    question = build_question(category, options)
    # Prompt format: "{context}\n{question}" with embedded <image> markers
    # resolved by ModelBackend.build_inputs (interleave_marker='<image>').
    full_prompt = context + '\n' + question
    inputs = backend.build_inputs(pil_images, full_prompt,
                                  interleave_marker='<image>')

    ids_list = inputs['ids_list']
    image_ranges = cached_img_ranges if cached_img_ranges is not None \
        else inputs['image_ranges']

    # Category token position (0-indexed last token of " {category}").
    cat_pos = backend.find_token_pos(
        ids_list, category,
        prefix_variants=[' ' + category], fallback='tail-15')

    anchor_ctx = ''
    if anchor == 'question':
        # 질문 템플릿 "... that contains {category}?" 을 직접 겨냥한다:
        # ' contains' 토큰열 출현 뒤에 category 변형이 바로 이어지는 첫 위치.
        #
        # ※ 단순 '첫 {category} 출현' 검색(구 cattok_anchor_check.py 방식)은 쓰면
        #   안 된다. filler 텍스트가 'person', 'car', 'orange' 같은 흔한 category
        #   단어를 포함하는 샘플이 23.5%(1,411/6,000)로 실측됐다 (2026-08-09).
        #   구 리부탈 question-anchor 수치(offset -1)도 이 오염을 포함한다.
        cont = backend.tokenizer(' contains', add_special_tokens=False)['input_ids']
        clean = re.sub(r'^(a|an)\s+', '', category.strip())
        var_ids = [backend.tokenizer(v, add_special_tokens=False)['input_ids']
                   for v in (' ' + clean, ' ' + category.strip(), clean)]
        q_pos = None
        for i in range(len(ids_list) - len(cont) + 1):
            if ids_list[i:i + len(cont)] != cont:
                continue
            j = i + len(cont)
            for vids in var_ids:
                if vids and ids_list[j:j + len(vids)] == vids:
                    q_pos = j + len(vids) - 1
                    break
            if q_pos is not None:
                break
        if q_pos is None:
            raise QAnchorNotFound('contains_seq_not_found')
        # 실행 시점 이중 검증: 디코드된 직전 문맥에 'contains' 가 실제로 있는지.
        anchor_ctx = backend.tokenizer.decode(ids_list[max(0, q_pos - 8):q_pos + 2])
        if 'contains' not in anchor_ctx:
            raise QAnchorNotFound('ctx_mismatch')
        cat_pos = q_pos

    head_collector.category_pos = cat_pos
    attn_collector.category_pos = cat_pos
    attn_collector.img_ranges = image_ranges

    # Generate one step to capture first-step logits + head outputs.
    result = backend.generate(inputs, max_new_tokens=10,
                              capture_first_logits=True)

    logit_stats = (get_logit_stats(result['first_logits'], mcq_token_ids)
                   if result.get('first_logits') is not None else {})
    scores_sum, scores_max = attn_collector.compute_scores(len(pil_images))

    # Embeddings computed once per sample (variant 0). Reuse afterward.
    embs = cached_embs if cached_embs is not None \
        else backend.extract_image_embeddings(inputs)

    parsed = parse_mcq_response(result['response'])
    torch.cuda.empty_cache()

    return {
        'parsed': parsed,
        'response': result['response'],
        'logit_stats': logit_stats,
        'head_out': head_collector.get_outputs(),
        'scores_sum': scores_sum,
        'scores_max': scores_max,
        'image_ranges': image_ranges,
        'embs': embs,
        'cat_pos': int(cat_pos),
        'anchor_ctx': anchor_ctx,
    }


# ── HDF5 save ──────────────────────────────────────────────────────────────────

def _save_head_group(parent_grp, phase_outputs, phases):
    for phase in phases:
        if phase in phase_outputs:
            pg = parent_grp.create_group(phase)
            for layer_idx, tensor in phase_outputs[phase].items():
                pg.create_dataset(f'layer_{layer_idx}',
                                  data=tensor.float().numpy(),
                                  compression='gzip', compression_opts=4)


def save_sample_group(hdf5_file, sample, target_0idx, run_results, verified,
                      scores_sum, scores_max):
    grp_key = f"sample_{sample['id']}"
    if grp_key in hdf5_file:
        del hdf5_file[grp_key]
    grp = hdf5_file.create_group(grp_key)

    grp.attrs['sample_id'] = sample['id']
    grp.attrs['original_id'] = sample['original_id']
    grp.attrs['n_images'] = sample['n_images']
    grp.attrs['target_index'] = target_0idx
    grp.attrs['target_category'] = sample['target_category']
    grp.attrs['context_length'] = sample['context_length']
    grp.attrs['verified'] = int(verified)
    # anchor 출처 자기기술 (question 모드 검증·사후 감사용)
    grp.attrs['anchor_pos'] = int(run_results[0].get('cat_pos', -1))
    grp.attrs['anchor_ctx'] = str(run_results[0].get('anchor_ctx', ''))

    first_img_embs = next((r['embs'] for r in run_results if r.get('embs')), None)
    if first_img_embs is not None:
        eg = grp.create_group('image_embeddings')
        for key in ('patch_mean', 'proj_mean'):
            arr = first_img_embs.get(key)
            if arr is not None and isinstance(arr, torch.Tensor):
                eg.create_dataset(key, data=arr.numpy(),
                                  compression='gzip', compression_opts=4)

    if scores_sum is not None:
        ag = grp.create_group('attn_score')
        ag.create_dataset('scores_sum', data=scores_sum.astype(np.float16),
                          compression='gzip', compression_opts=4)
        ag.create_dataset('scores_max', data=scores_max.astype(np.float16),
                          compression='gzip', compression_opts=4)

    first_head_out = run_results[0]['head_out']
    if 'cattok' in first_head_out:
        _save_head_group(grp, first_head_out, ['cattok'])

    for r in run_results:
        ans_grp = grp.create_group(f"ans{r['ans_label']}")
        ans_grp.attrs['answer_label'] = r['ans_label']
        ans_grp.attrs['response'] = r['response']
        ans_grp.attrs['correct'] = int(r['correct'])
        for k, v in r['logit_stats'].items():
            ans_grp.attrs[k] = float(v)
        if 'generation' in r['head_out']:
            _save_head_group(ans_grp, r['head_out'], ['generation'])


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    cfg = MODEL_CONFIGS[args.model]

    print("Loading datasets...")
    all_samples = []
    for dataset_path in args.datasets:
        file_samples = [json.loads(l) for l in open(dataset_path)]
        print(f"  {len(file_samples)} samples from {os.path.basename(dataset_path)}")
        all_samples.extend(file_samples)

    samples = all_samples[args.worker_id::args.num_workers]
    print(f"  {len(all_samples)} total, worker {args.worker_id}/{args.num_workers} "
          f"→ {len(samples)} samples")

    outputs_dir = args.outputs_dir or os.path.join(
        ATTN_ROOT, cfg['model_dir'], 'coco_mcq')
    os.makedirs(outputs_dir, exist_ok=True)

    suffix = f'_w{args.worker_id}' if args.num_workers > 1 else ''
    hdf5_path = os.path.join(outputs_dir, f'head_outputs{suffix}.h5')
    result_path = os.path.join(outputs_dir, f'results{suffix}.jsonl')

    done_keys = set()
    if os.path.exists(hdf5_path):
        with h5py.File(hdf5_path, 'r') as f:
            done_keys = set(f.keys())
        print(f"  Resuming: {len(done_keys)} sample groups already saved")

    evaluated_ids = set()
    if os.path.exists(result_path):
        for line in open(result_path):
            r = json.loads(line)
            if 'id' in r:
                evaluated_ids.add(r['id'])
        print(f"  Skipping: {len(evaluated_ids)} already evaluated IDs")

    print(f"Loading {args.model} (FA2)...")
    backend = ModelBackend(args.model)
    print(f"  {backend.num_layers} layers, {backend.num_heads} heads "
          f"({backend.num_kv_heads} kv), head_dim={backend.head_dim}")

    head_collector = HeadOutputCollector(
        backend.layers, backend.num_layers, backend.num_heads, backend.head_dim)
    head_collector.register_hooks()

    attn_collector = AttnScoreCollector(
        backend.layers, backend.num_layers, backend.num_heads,
        backend.num_kv_heads, backend.head_dim,
        model=getattr(backend, 'model', None))
    attn_collector.register_hooks()

    mcq_token_ids = backend.get_letter_ids(MCQ_LABELS)
    print(f"  MCQ token IDs: {mcq_token_ids}")

    if args.preflight:
        # knob 검증 (launch 전 필수). 세 단계:
        #  (0) 모듈 의미 검증 — 예외 클래스가 진짜 예외인지 (2026-08-09 데코레이터
        #      삽입 사고: py_compile 은 통과하는데 런타임에 죽는 유형을 여기서 잡는다)
        #  (1) 위치 검증 — 일반 3개 + filler-충돌 3개 샘플에서 anchor 가 질문 문장에
        #      착지하고('contains' 문맥), canonical 과 다른지
        #  (2) 경로 검증 — 충돌 샘플 1개를 run_variant 로 끝까지 실행 (수정된 코드
        #      경로 전체가 실제로 돈다는 증명)
        ok = True
        if not (isinstance(QAnchorNotFound, type)
                and issubclass(QAnchorNotFound, Exception)):
            print('  [preflight] FAIL: QAnchorNotFound 가 예외 클래스가 아님')
            ok = False
        normal, collide = [], []
        for s in samples:
            kw = ' ' + s['target_category'].split()[-1].lower()
            (collide if kw in s['context'].lower() else normal).append(s)
            if len(normal) >= 3 and len(collide) >= 3:
                break
        for s in (normal[:3] + collide[:3]):
            cat = s['target_category']
            prompt = s['context'] + '\n' + build_question(cat, s['mcq_options'])
            pils = [Image.open(os.path.join(COCO_ROOT, p)).convert('RGB')
                    for p in s['images_list']]
            inp = backend.build_inputs(pils, prompt, interleave_marker='<image>')
            ids = inp['ids_list']
            last = backend.find_token_pos(ids, cat, prefix_variants=[' ' + cat],
                                          fallback='tail-15')
            try:
                r = run_variant(backend, pils, cat, s['context'], s['mcq_options'],
                                head_collector, attn_collector, mcq_token_ids,
                                anchor='question') if s is collide[0] else None
            except QAnchorNotFound as e:
                print(f"  [preflight] id={s['id']} run_variant skip: {e}"); ok = False; r = None
            # 위치만 별도 재계산 (run_variant 와 동일 로직)
            cont = backend.tokenizer(' contains', add_special_tokens=False)['input_ids']
            _clean = re.sub(r'^(a|an)\s+', '', cat.strip())
            vidss = [backend.tokenizer(v, add_special_tokens=False)['input_ids']
                     for v in (' ' + _clean, ' ' + cat.strip(), _clean)]
            q = None
            for i in range(len(ids) - len(cont) + 1):
                if ids[i:i + len(cont)] != cont: continue
                j = i + len(cont)
                for vids in vidss:
                    if vids and ids[j:j + len(vids)] == vids: q = j + len(vids) - 1; break
                if q is not None: break
            ctx_q = backend.tokenizer.decode(ids[max(0, (q or 0) - 8):(q or 0) + 2])
            tag = '충돌' if s in collide else '일반'
            good = q is not None and q != last and 'contains' in ctx_q
            ok = ok and good
            print(f"  [preflight] [{tag}] id={s['id']} cat={cat!r} last={last} q={q} "
                  f"ctx={ctx_q!r} {'✓' if good else '✗'}")
            if r is not None:
                e2e = r.get('cat_pos') == q and 'contains' in r.get('anchor_ctx', '')
                ok = ok and e2e
                print(f"  [preflight] [경로] run_variant cat_pos={r.get('cat_pos')} "
                      f"(기대 {q}) {'✓' if e2e else '✗'}")
        print(f"  [preflight] {'OK' if ok else 'FAIL'}")
        return

    hdf5_file = h5py.File(hdf5_path, 'a')
    hdf5_file.attrs['anchor'] = args.anchor
    result_file = open(result_path, 'a')
    stats = {'total': 0, 'verified': 0, 'failed': 0, 'oom': 0}

    try:
        for sample in tqdm(samples, desc=f"Worker{args.worker_id}"):
            stats['total'] += 1
            if sample['id'] in evaluated_ids:
                continue
            grp_key = f"sample_{sample['id']}"
            if grp_key in done_keys:
                stats['verified'] += 1
                continue

            cat = sample['target_category']
            target_0idx = sample['target_index']
            orig_answer = sample['mcq_answer']
            orig_options = sample['mcq_options']
            context = sample['context']
            n = sample['n_images']

            # Load images once.
            def _load_pil(rel):
                return Image.open(os.path.join(COCO_ROOT, rel)).convert('RGB')
            with ThreadPoolExecutor(max_workers=min(8, n)) as ex:
                pil_images = list(ex.map(_load_pil, sample['images_list']))

            cached_ranges = None
            cached_embs = None
            run_results = []
            scores_sum_final = scores_max_final = None

            try:
                for vi, ans_label in enumerate(MCQ_LABELS):
                    rotated_opts = rotate_options(orig_options, orig_answer, ans_label)
                    r = run_variant(
                        backend, pil_images, cat, context, rotated_opts,
                        head_collector, attn_collector, mcq_token_ids,
                        cached_img_ranges=cached_ranges,
                        cached_embs=cached_embs,
                        anchor=args.anchor,
                    )
                    if vi == 0:
                        cached_ranges = r['image_ranges']
                        cached_embs = r['embs']
                        scores_sum_final = r['scores_sum']
                        scores_max_final = r['scores_max']
                    run_results.append({
                        'ans_label': ans_label,
                        'parsed': r['parsed'],
                        'response': r['response'],
                        'head_out': r['head_out'],
                        'embs': r['embs'] if vi == 0 else None,
                        'logit_stats': r['logit_stats'],
                        'correct': (r['parsed'] == ans_label),
                        'cat_pos': r['cat_pos'],
                        'anchor_ctx': r['anchor_ctx'],
                    })
                if args.anchor == 'question':
                    # 질문 anchor 는 선택지 앞이므로 4개 rotation 에서 위치가
                    # 동일해야 한다. 다르면 anchor 가 선택지에 걸렸다는 뜻.
                    _ps = {rr['cat_pos'] for rr in run_results}
                    if len(_ps) != 1:
                        raise QAnchorNotFound(f'rotation_inconsistent:{sorted(_ps)}')
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                stats['oom'] += 1
                result_file.write(json.dumps({'id': sample['id'], 'status': 'oom'}) + '\n')
                result_file.flush()
                continue
            except QAnchorNotFound as e:
                stats['q_pos_not_found'] = stats.get('q_pos_not_found', 0) + 1
                result_file.write(json.dumps({'id': sample['id'],
                                              'status': f'q_anchor_skip:{e}'}) + '\n')
                result_file.flush()
                continue
            finally:
                torch.cuda.empty_cache()

            verified = all(r['correct'] for r in run_results)
            if verified:
                stats['verified'] += 1
            else:
                stats['failed'] += 1

            result_file.write(json.dumps({
                'id':        sample['id'],
                'cat':       cat,
                'n_images':  n,
                'responses': [r['response']  for r in run_results],
                'parsed':    [r['parsed']    for r in run_results],
                'correct':   [r['correct']   for r in run_results],
                'verified':  verified,
            }) + '\n')
            result_file.flush()

            if grp_key not in done_keys:
                save_sample_group(hdf5_file, sample, target_0idx, run_results,
                                  verified, scores_sum_final, scores_max_final)
                done_keys.add(grp_key)
            hdf5_file.flush()

            if stats['total'] % 50 == 0:
                vrate = 100 * stats['verified'] / stats['total']
                print(f"  [{stats['total']}/{len(samples)}] verified={stats['verified']} "
                      f"({vrate:.1f}%) failed={stats['failed']} oom={stats['oom']}")
    finally:
        result_file.close()
        hdf5_file.close()
        head_collector.remove_hooks()
        attn_collector.remove_hooks()

    print(f"\nDone.")
    print(f"  Total    : {stats['total']}")
    print(f"  Verified : {stats['verified']} "
          f"({100 * stats['verified'] / max(stats['total'], 1):.1f}%)")
    print(f"  Failed   : {stats['failed']}")
    print(f"  OOM      : {stats['oom']}")
    print(f"  HDF5     : {hdf5_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Unified COCO MCQ head output + attn score extractor.')
    parser.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()),
                        help='Model alias (e.g. internvl-8b, qwen3vl-8b, llava-ov-7b)')
    parser.add_argument('--datasets', nargs='+', default=[
        os.path.join(ATTN_ROOT, 'datasets', 'coco_retrieval_n5_mcq.jsonl'),
        os.path.join(ATTN_ROOT, 'datasets', 'coco_retrieval_n6_mcq.jsonl'),
    ])
    parser.add_argument('--outputs-dir', default=None,
                        help='Default: $ATTN_ROOT/{model_dir}/coco_mcq')
    parser.add_argument('--worker-id', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--anchor', choices=['last', 'question'], default='last',
                        help="cattok anchor: 'last'=canonical(논문, 마지막 언급), "
                             "'question'=질문 줄 첫 언급 (robustness 확장 전용. "
                             "산출물을 canonical 트리에 섞지 말 것)")
    parser.add_argument('--preflight', type=int, default=0,
                        help='N개 샘플로 anchor 위치만 출력하고 종료 (knob 검증)')
    args = parser.parse_args()
    main(args)
