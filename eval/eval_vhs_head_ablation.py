"""
VHs head ablation — family-agnostic via ModelBackend.
Sweep over hs × ranking sources × top-k with single model load.

╔══════════════════════════════════════════════════════════════════════════╗
║  PAPER-ALIGNED DEFAULT (use this for any ablation reported in the paper) ║
║                                                                          ║
║    --ablation-mode catpos --ablation-pos cattok_onwards                  ║
║                                                                          ║
║  Output-zero on the head's o_proj input over the image-retrieval span    ║
║  (anchor-object phrase first token → target-object phrase last token).   ║
║  Removes the head's residual-stream write entirely on this window.       ║
║  Decode positions are not modified: the first answer token's logit is    ║
║  read from the last prefill position, which already reflects the         ║
║  prefill-window ablation.                                                ║
║                                                                          ║
║  TERMINOLOGY (code ↔ paper):                                             ║
║    catpos        ↔ output-zero (PAPER ablation, eq:writer-zero on        ║
║                                  o_proj input)                           ║
║    qzero_catpos  ↔ Q-zero  (LEGACY; removed from the paper)              ║
║    ranking-source attn_score  ↔ pointer ranking                          ║
║    ranking-source retriever   ↔ writer ranking                           ║
║                                                                          ║
║  All --ablation-mode values other than `catpos` are LEGACY (kept for     ║
║  reproducing prior experiments only — do NOT use for new paper experiments). ║
╚══════════════════════════════════════════════════════════════════════════╝

Usage (paper protocol):
  CUDA_VISIBLE_DEVICES=1 python eval/eval_vhs_head_ablation.py \
      --model internvl-8b --hs 5,10,20 --top-k 30 \
      --ranking-source baseline,random,attn_score,retriever \
      --phase cattok --ablation-mode catpos \
      --ablation-pos cattok_onwards --subset VHs_large

NOTE: jsonl filenames continue to embed the legacy mode tag (e.g.
``..._catpos_cattok_onwards.jsonl``) for backward compatibility with
existing data on disk; do not rename the suffix or the result loaders
(build_setpartition_view, build_topk_sweep_table) will fail to find
prior outputs.
"""
import os, sys, json, random, argparse, csv
from contextlib import contextmanager

import numpy as np
import torch
import h5py
from tqdm import tqdm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.model_backend import ModelBackend
from utils.model_utils import DATA_ROOT, COCO_ROOT, ATTN_ROOT, MODEL_CONFIGS, parse_yes_no
from utils.ranking_manifest import get_resolver
from utils.generation_utils import YesNoStop  # early-stop on first 'yes'/'no'

try:
    from transformers import StoppingCriteriaList
except ImportError:
    StoppingCriteriaList = list


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY ablation context managers — kept for reproducing prior experiments
# only. NOT used in the paper. Do NOT use for new paper experiments.
#   - ablate_heads_oproj          (`--ablation-mode zero`)
#   - ablate_heads_qzero_decode   (`--ablation-mode qzero_decode`)
#   - ablate_heads_qzero_catpos   (`--ablation-mode qzero_catpos`,  Q-zero)
#   - ablate_heads_mean_catpos    (`--ablation-mode mean_catpos`)
#   - ablate_heads_image_value    (`--ablation-mode image_value`)
# The PAPER ablation (`--ablation-mode catpos`, output-zero) is
# `ablate_heads_catpos` further down.
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def ablate_heads_oproj(layers, head_dim, mask_dict):
    """[LEGACY — `--ablation-mode zero`] Zero the o_proj columns corresponding
    to each ablated head (full forward). Removes the head from every position
    of every step. Superseded by ablate_heads_catpos (output-zero) for the paper."""
    if not mask_dict:
        yield
        return
    backups = []
    try:
        for lidx, heads in mask_dict.items():
            o_proj = layers[lidx].self_attn.o_proj
            for h in heads:
                sl = slice(h * head_dim, (h + 1) * head_dim)
                backups.append((o_proj.weight, sl,
                                o_proj.weight[:, sl].detach().clone().cpu()))
                o_proj.weight.data[:, sl] = 0.0
        yield
    finally:
        for param, sl, orig in backups:
            param.data[:, sl] = orig.to(param.device)


@contextmanager
def ablate_heads_qzero_decode(layers, head_dim, mask_dict):
    """[LEGACY — `--ablation-mode qzero_decode`] Zero last-position Q for
    ablated heads (answer-token-only ablation). Closest in spirit to
    Wu et al. 2024 retrieval-head decode-only ablation; superseded for
    the paper by ablate_heads_catpos (output-zero, prefill-window)."""
    if not mask_dict:
        yield
        return
    hooks = []
    def make_hook(heads_list):
        def hook(_m, _inputs, output):
            if output.dim() != 3:
                return output
            out = output.clone()
            for h in heads_list:
                out[:, -1:, h * head_dim:(h + 1) * head_dim] = 0
            return out
        return hook
    try:
        for lidx, heads in mask_dict.items():
            q_proj = layers[lidx].self_attn.q_proj
            hooks.append(q_proj.register_forward_hook(make_hook(list(heads))))
        yield
    finally:
        for h in hooks:
            h.remove()


# ══════════════════════════════════════════════════════════════════════════════
# LEGACY: Q-zero variant (qzero_catpos). Removed from the paper but kept
# here for reproducing prior experiments only. PAPER ablation is the
# output-zero `ablate_heads_catpos` defined below.
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def ablate_heads_qzero_catpos(layers, head_dim, mask_dict, pos_holder, onwards=False):
    """[LEGACY — `--ablation-mode qzero_catpos`, "Q-zero"]. NOT used in the paper.

    Zero q_proj output for ablated heads over the prefill image-retrieval
    span. Decode positions are not modified, matching the paper-aligned
    output-zero scope, so the only difference vs. `ablate_heads_catpos`
    is *what* is zeroed (q_proj output vs. o_proj input).

    Why output-zero (catpos) is the paper choice instead: under Q-zero the
    head still writes mean(V) into the residual stream, which can retain
    the writer-criterion signal and biases the comparison between pointer
    and writer head sets toward the pointer side. Output-zero removes the
    head's residual write in full and is equally fair across criteria.

    Computation:
      - Prefill: zero q_proj output at [start, seq_end] (onwards=True) or
        [start, end] (onwards=False).
      - Decode (q_len == 1): no-op.
      - Effect at zeroed prefill positions: softmax(0·K^T) = uniform →
        attn_output = mean(V) → head still writes mean(V)·W_o^h to the
        residual stream.
    """
    if not mask_dict:
        yield
        return
    hooks = []
    def make_hook(heads_list):
        def hook(_m, _inputs, output):
            if output.dim() != 3 or output.shape[1] == 1:
                return output
            pos = pos_holder[0]
            if pos is None:
                return output
            start, end = pos
            if start >= output.shape[1]:
                return output
            out = output.clone()
            sl = slice(start, None) if onwards else slice(start, end + 1)
            for h in heads_list:
                out[:, sl, h * head_dim:(h + 1) * head_dim] = 0
            return out
        return hook
    try:
        for lidx, heads in mask_dict.items():
            q_proj = layers[lidx].self_attn.q_proj
            hooks.append(q_proj.register_forward_hook(make_hook(list(heads))))
        yield
    finally:
        for h in hooks:
            h.remove()

# ══════════════════════════════════════════════════════════════════════════════
# PAPER ablation — output-zero (`--ablation-mode catpos`).
# This is the ablation reported in all paper figures and tables.
# ══════════════════════════════════════════════════════════════════════════════


@contextmanager
def ablate_heads_catpos(layers, head_dim, mask_dict, pos_holder, onwards=False,
                        ablate_gen=False):
    """**PAPER ablation** [`--ablation-mode catpos`, "output-zero"].

    Zero ablated heads' o_proj input over the image-retrieval span: from
    the first token of the anchor-object phrase to the final token of the
    target-object phrase (with `onwards=True` / `--ablation-pos cattok_onwards`,
    the paper default — the slice extends to the end of the prefill but
    decode positions are still no-ops, see below).

    Computation:
      - Prefill: zero o_proj input at [start, seq_end] (onwards=True) or
        [start, end] (onwards=False). This sets the head's contribution to
        the residual stream to zero on this window, leaving every other
        component (image encoding, KV cache, other heads) intact.
      - Decode (q_len == 1): no-op. The first answer token's logit is read
        from the last prefill position, where the prefill ablation has
        already been applied, so the answer-relevant logit reflects the
        intervention without intervening on autoregressive generation.

    Why output-zero (this hook) rather than Q-zero (`ablate_heads_qzero_catpos`):
    Q-zero leaves the head writing mean(V) into the residual stream,
    which can retain the writer-criterion signal and biases the comparison
    between pointer and writer top sets toward the pointer side. Output-zero
    removes the head's residual write in full and is equally fair to both
    head-detection criteria.
    """
    if not mask_dict:
        yield
        return
    hooks = []
    def make_pre_hook(heads_list):
        def pre_hook(_m, inputs):
            pos = pos_holder[0]
            if pos is None:
                return inputs
            start, end = pos
            x = inputs[0]
            if x.dim() != 3:
                return inputs
            if x.shape[1] == 1:
                if not ablate_gen:
                    return inputs
                x = x.clone()
                for h in heads_list:
                    x[:, :, h * head_dim:(h + 1) * head_dim] = 0
                return (x,) + inputs[1:]
            if start >= x.shape[1]:
                return inputs
            x = x.clone()
            sl = slice(start, None) if onwards else slice(start, end + 1)
            for h in heads_list:
                x[:, sl, h * head_dim:(h + 1) * head_dim] = 0
            return (x,) + inputs[1:]
        return pre_hook
    try:
        for lidx, heads in mask_dict.items():
            o_proj = layers[lidx].self_attn.o_proj
            hooks.append(o_proj.register_forward_pre_hook(make_pre_hook(list(heads))))
        yield
    finally:
        for h in hooks:
            h.remove()


# Paper-terminology alias: the paper calls this "output-zero". The legacy
# code name (`catpos`) is kept as the canonical implementation because all
# on-disk JSONL filenames embed it in their suffix; new callers may use
# either name.
ablate_heads_output_zero = ablate_heads_catpos
ablate_heads_query_zero  = ablate_heads_qzero_catpos  # paper: Q-zero (legacy)


@contextmanager
def ablate_heads_mean_catpos(layers, head_dim, mask_dict, pos_holder, mean_tensor, onwards=False):
    """[LEGACY — `--ablation-mode mean_catpos`] Replace ablated heads'
    o_proj input at needle position(s) with precomputed mean. Mean-substitution
    variant of catpos; kept for prior-experiment reproduction. Superseded by
    qzero_catpos for the paper."""
    if not mask_dict:
        yield
        return
    hooks = []
    def make_pre_hook(lidx, heads_list):
        def pre_hook(_m, inputs):
            pos = pos_holder[0]
            if pos is None:
                return inputs
            start, end = pos
            x = inputs[0]
            if x.dim() != 3 or start >= x.shape[1]:
                return inputs
            x = x.clone()
            sl = slice(start, None) if onwards else slice(start, end + 1)
            n_pos = x[:, sl, :].shape[1]
            for h in heads_list:
                m = mean_tensor[lidx, h].to(x.dtype)
                x[:, sl, h * head_dim:(h + 1) * head_dim] = m.unsqueeze(0).unsqueeze(0).expand(1, n_pos, -1)
            return (x,) + inputs[1:]
        return pre_hook
    try:
        for lidx, heads in mask_dict.items():
            o_proj = layers[lidx].self_attn.o_proj
            hooks.append(o_proj.register_forward_pre_hook(make_pre_hook(lidx, list(heads))))
        yield
    finally:
        for h in hooks:
            h.remove()


@contextmanager
def ablate_heads_image_value(layers, head_dim, mask_dict, img_ranges_holder):
    """[LEGACY — `--ablation-mode image_value`] Zero ablated heads' v_proj
    input at image token positions. Different intervention site (V at image
    tokens) — kept for prior-experiment reproduction; not used in the paper."""
    if not mask_dict:
        yield
        return
    hooks = []
    def make_pre_hook(heads_list):
        def pre_hook(_m, inputs):
            img_ranges = img_ranges_holder[0]
            if not img_ranges:
                return inputs
            x = inputs[0]
            if x.dim() != 3:
                return inputs
            x = x.clone()
            for st, en in img_ranges:
                e = min(en + 1, x.shape[1])
                if st >= x.shape[1]:
                    continue
                for h in heads_list:
                    x[:, st:e, h * head_dim:(h + 1) * head_dim] = 0
            return (x,) + inputs[1:]
        return pre_hook
    try:
        for lidx, heads in mask_dict.items():
            v_proj = layers[lidx].self_attn.v_proj
            hooks.append(v_proj.register_forward_pre_hook(make_pre_hook(list(heads))))
        yield
    finally:
        for h in hooks:
            h.remove()


# ── Position helpers ───────────────────────────────────────────────────────────

def compute_lastq_pos(backend, ids_list):
    """Position of the last <|im_end|> token (end-of-user-turn)."""
    im_end_id = backend.tokenizer.convert_tokens_to_ids('<|im_end|>')
    if im_end_id is None or im_end_id == backend.tokenizer.unk_token_id:
        return len(ids_list) - 1
    for i in range(len(ids_list) - 1, -1, -1):
        if ids_list[i] == im_end_id:
            return i
    return len(ids_list) - 1


def compute_needle_range(backend, ids_list, needle_text):
    """Token range (start, end) of the needle phrase."""
    import re as _re
    clean = _re.sub(r'^(a|an)\s+', '', needle_text.strip())
    for variant in [' ' + clean, ' ' + needle_text.strip(), clean]:
        ids = backend.tokenizer(variant, add_special_tokens=False)['input_ids']
        if not ids:
            continue
        for i in range(len(ids_list) - len(ids), -1, -1):
            if ids_list[i:i + len(ids)] == ids:
                return (i, i + len(ids) - 1)
    mid = len(ids_list) // 2
    return (mid, mid)


# ── Top-K head selection ───────────────────────────────────────────────────────

# Sources whose head set depends on the set-partition threshold N.
# Only these get an `_N{n}` filename suffix when n != 100; every other source
# (random / attn_score / retriever / ...) is N-independent, so it keeps the
# canonical filename and is shared across partition thresholds.
PARTITION_SOURCES = frozenset({
    'overlap_set_attn', 'attn_only_set', 'retr_only_set',
    # paper-terminology aliases (resolved inside get_top_k_heads)
    'pointer_only', 'writer_only',
})

# 제거된 이름들 (2026-08-08). max-rank 정렬 overlap 은 논문 arm 이 아닌데
# alias `pointer_writer_overlap` 이 조용히 그쪽으로 매핑돼 있어 E3(리부탈)가
# 의도치 않게 다른 정렬로 실행됐다. 논문 overlap arm 은 `overlap_set_attn`
# (pointer-rank 정렬) 하나뿐이다. 옛 이름을 넘기면 명시적으로 실패한다.
REMOVED_SOURCES = frozenset({'overlap_set', 'pointer_writer_overlap'})


def get_top_k_heads(cfg, hs, k, phase='cattok', source='attn_score', random_seed=42,
                    partition_n=100):
    """Return (heads, head_info) for the requested ranking source.

    TERMINOLOGY (code source-name ↔ paper):
      attn_score      ↔ pointer ranking          (per-image attention mass R@1)
      retriever       ↔ writer ranking           (trained per-head probe R@1)
      attn_only_set   ↔ pointer-only tail        (in pointer top-100, not writer top-100)
      retr_only_set   ↔ writer-only tail         (in writer top-100, not pointer top-100)
      overlap_set     ↔ pointer & writer overlap (in both top-100)
      overlap_set_attn ↔ same overlap, ordered by pointer-rank within the set

    `pointer` / `writer` are accepted as aliases for `attn_score` / `retriever`
    so callers can use paper terminology directly.

    `partition_n` is the set-partition threshold N (paper default 100). It only
    affects the four set-partition sources listed above; all other sources
    ignore it.
    """
    # accept paper-terminology aliases
    if source == 'pointer':
        source = 'attn_score'
    elif source == 'writer':
        source = 'retriever'
    elif source == 'pointer_only':
        source = 'attn_only_set'
    elif source == 'writer_only':
        source = 'retr_only_set'
    if source in REMOVED_SOURCES:
        raise ValueError(
            f"ranking-source '{source}' 는 제거됐다 (2026-08-08). max-rank 정렬 overlap 은 "
            f"논문 arm 이 아니며, alias 'pointer_writer_overlap' 이 그쪽으로 매핑돼 있던 것이 "
            f"E3(리부탈) 오실행의 원인이었다. 논문 overlap arm 은 'overlap_set_attn' "
            f"(pointer-rank 정렬) 하나다. build_setpartition_view.py SOURCES 참조.")
    if k == 0 or source == 'baseline':
        return [], []
    if source == 'retriever':
        return _top_k_retriever(cfg, k, phase)
    elif source == 'retriever_gen':
        return _top_k_retriever(cfg, k, 'generation')
    elif source == 'random':
        return _random_k(k, cfg['num_layers'], cfg['num_heads'], seed=random_seed)
    elif source == 'overlap':
        return _top_k_overlap(cfg, k, phase, agg='max')
    elif source == 'overlap_gen':
        return _top_k_overlap(cfg, k, phase, agg='max', retriever_phase='generation')
    elif source == 'overlap_avg':
        return _top_k_overlap(cfg, k, phase, agg='avg')
    elif source == 'overlap_avg_gen':
        return _top_k_overlap(cfg, k, phase, agg='avg', retriever_phase='generation')
    elif source == 'score_sum':
        return _top_k_score_sum(cfg, k, phase)
    elif source == 'overlap_set_attn':
        return _top_k_set_partition(cfg, k, phase, mode='overlap', N=partition_n,
                                    sort_by='attn')
    elif source == 'attn_only_set':
        return _top_k_set_partition(cfg, k, phase, mode='attn_only', N=partition_n)
    elif source == 'retr_only_set':
        return _top_k_set_partition(cfg, k, phase, mode='retr_only', N=partition_n)
    elif source == 'interleave':
        return _top_k_interleave(cfg, k, phase)
    elif source == 'interleave_gen':
        return _top_k_interleave(cfg, k, phase, retriever_phase='generation')
    else:
        return _top_k_attn_score(cfg, hs, k, phase)


def _load_rank_dict(rows_with_score):
    """Given a list of (layer, head, score) rows, return {(l,h): rank} where
    rank=0 is the highest-scored head."""
    rows = sorted(rows_with_score, key=lambda x: -x[2])
    return {(l, h): i for i, (l, h, _) in enumerate(rows)}


def _top_k_overlap(cfg, k, phase, agg='max', retriever_phase=None):
    """Top-k by *agreement* between the attention-score and trained-retriever
    rankings.

    Two aggregation modes:
      - agg='max':  overlap_score(h) = max(rank_attn(h), rank_retr(h))
                    (heads in the intersection of two top-N rankings as N grows)
      - agg='avg':  overlap_score(h) = (rank_attn(h) + rank_retr(h)) / 2
                    (heads with the smallest mean rank across the two criteria)

    `retriever_phase`, if given, overrides which retriever ranking CSV is used
    (e.g. 'generation' to use the retriever trained on generation-phase head
    outputs, while still using the attn-score (ICIR-MCQ) ranking at `phase`).
    """
    if retriever_phase is None:
        retriever_phase = phase
    assert agg in ('max', 'avg')
    # ── attention-score (ICIR-MCQ) ───────────────────────────────────────────
    coco_csv = get_resolver().pointer_csv(cfg['model_dir'], phase, require=False)
    if not os.path.exists(coco_csv):
        raise FileNotFoundError(
            f'ICIR-MCQ attn_score ranking missing: {coco_csv}; overlap '
            f'requires both rankings.')
    attn_rows = []
    with open(coco_csv) as f:
        for row in csv.DictReader(f):
            attn_rows.append((int(row['layer']), int(row['head']),
                              float(row['r1'])))
    attn_rank = _load_rank_dict(attn_rows)

    # ── trained retriever ────────────────────────────────────────────────────
    ranking_name = get_resolver().writer_ranking(cfg['model_dir'], cfg.get('retriever_ranking'))
    if ranking_name is None:
        raise FileNotFoundError(
            f"No retriever ranking for {cfg['model_dir']}; overlap requires both rankings.")
    model_abs = os.path.join(ATTN_ROOT, cfg['model_dir'])
    ranking_path = (ranking_name if os.path.isabs(ranking_name)
                    else os.path.join(model_abs, 'head_rankings', ranking_name))
    if not os.path.exists(ranking_path):
        ranking_path = os.path.join(model_abs, ranking_name)
    if os.path.isdir(ranking_path):
        csv_path = os.path.join(ranking_path, f'{retriever_phase}_proj_mean_fullpool.csv')
        r1_col = 'r1_dedup'
    else:
        csv_path = ranking_path
        r1_col = 'recall@1'
    retr_rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            retr_rows.append((int(row['layer']), int(row['head']),
                              float(row[r1_col])))
    retr_rank = _load_rank_dict(retr_rows)

    # ── score by chosen aggregation; tie-break by the orthogonal stat ─────────
    all_heads = set(attn_rank.keys()) | set(retr_rank.keys())
    big = max(len(attn_rank), len(retr_rank)) + 1
    scored = []
    for (l, h) in all_heads:
        ra = attn_rank.get((l, h), big)
        rr = retr_rank.get((l, h), big)
        if agg == 'max':
            primary, tiebreak = max(ra, rr), min(ra, rr)
        else:  # agg == 'avg'
            primary, tiebreak = (ra + rr), abs(ra - rr)
        scored.append((l, h, primary, tiebreak))
    scored.sort(key=lambda x: (x[2], x[3]))
    top = scored[:k]
    return [(l, h) for l, h, _, _ in top], [(l, h, float(s)) for l, h, s, _ in top]


def _top_k_set_partition(cfg, k, phase, mode='overlap', N=100, sort_by='default'):
    """Within the partition (defined by top-N) of attn∪retr, take the top-k
    heads from one disjoint subset:
      overlap     : heads in BOTH attn-top-N and retr-top-N.
                    논문 arm 은 sort_by='attn' (rank_attn ascending) 하나뿐이다.
                    max-rank 정렬(구 'default')은 2026-08-08 에 제거됐다 — E3 오실행 원인.
      attn_only   : in attn-top-N but NOT retr-top-N, ordered by rank_attn
      retr_only   : in retr-top-N but NOT attn-top-N, ordered by rank_retr
    `N` is the partition threshold (default 100). `k` is # heads to ablate; if
    `k` exceeds the subset size the full subset is returned.
    """
    if mode == 'overlap' and sort_by != 'attn':
        raise ValueError(
            f"overlap 정렬은 'attn'(논문 arm)만 지원한다. sort_by={sort_by!r} 는 제거됨.")
    coco_csv = get_resolver().pointer_csv(cfg['model_dir'], phase, require=False)
    if not os.path.exists(coco_csv):
        raise FileNotFoundError(f'attn_score CSV missing: {coco_csv}')
    attn_rows = []
    with open(coco_csv) as f:
        for row in csv.DictReader(f):
            attn_rows.append((int(row['layer']), int(row['head']), float(row['r1'])))
    attn_rank = _load_rank_dict(attn_rows)

    # writer 랭킹은 반드시 resolver를 통해 해석한다.
    # 2026-07-26 사고: 이 함수가 cfg['retriever_ranking']만 읽어서, matched-writer
    # ablation이 override를 설정했는데도 20개 arm 전부가 canonical partition으로
    # 조용히 실행됐다. resolver가 manifest/env/cfg 우선순위를 한 곳에서 처리하고
    # 해석 결과를 로그에 남기므로 같은 침묵 실패가 재발하지 않는다.
    retr_dir_name = get_resolver().writer_ranking(cfg['model_dir'], cfg.get('retriever_ranking'))
    if os.path.isabs(retr_dir_name):
        retr_dir = retr_dir_name
    else:
        retr_dir = os.path.join(ATTN_ROOT, cfg['model_dir'], 'head_rankings', retr_dir_name)
        if not os.path.exists(retr_dir):
            retr_dir = os.path.join(ATTN_ROOT, cfg['model_dir'], retr_dir_name)
    retr_csv = (os.path.join(retr_dir, f'{phase}_proj_mean_fullpool.csv')
                if os.path.isdir(retr_dir) else retr_dir)
    r1_col = 'r1_dedup' if os.path.isdir(retr_dir) else 'recall@1'
    retr_rows = []
    with open(retr_csv) as f:
        for row in csv.DictReader(f):
            retr_rows.append((int(row['layer']), int(row['head']), float(row[r1_col])))
    retr_rank = _load_rank_dict(retr_rows)

    a_set = {h for h, r in attn_rank.items() if r < N}
    r_set = {h for h, r in retr_rank.items() if r < N}
    if mode == 'overlap':
        cand = a_set & r_set
        big = N + 1
        if sort_by == 'attn':
            scored = sorted(cand, key=lambda h: attn_rank.get(h, big))
        elif sort_by == 'retr':
            scored = sorted(cand, key=lambda h: retr_rank.get(h, big))
        else:  # default = max-rank, tie-break by min-rank
            scored = sorted(cand, key=lambda h: (max(attn_rank.get(h, big), retr_rank.get(h, big)),
                                                  min(attn_rank.get(h, big), retr_rank.get(h, big))))
    elif mode == 'attn_only':
        cand = a_set - r_set
        scored = sorted(cand, key=lambda h: attn_rank[h])
    elif mode == 'retr_only':
        cand = r_set - a_set
        scored = sorted(cand, key=lambda h: retr_rank[h])
    else:
        raise ValueError(mode)
    top = scored[:k]
    return top, [(l, h, 0.0) for l, h in top]


def _top_k_score_sum(cfg, k, phase):
    """Top-k by *score-sum* of attn_score R@1 (ICIR-MCQ in-sample, 5-6 way) +
    retriever R@1 (in-sample, same 5-6 way pool from `*_insample.csv`).

    Both R@1 are evaluated within each MCQ sample's image set (5-6 candidates),
    so they share the same chance baseline (~0.18) and can be summed directly
    without rank/scale normalization.
    """
    coco_csv = get_resolver().pointer_csv(cfg['model_dir'], phase, require=False)
    if not os.path.exists(coco_csv):
        raise FileNotFoundError(
            f'ICIR-MCQ attn_score ranking missing: {coco_csv}')
    attn = {}
    with open(coco_csv) as f:
        for row in csv.DictReader(f):
            attn[(int(row['layer']), int(row['head']))] = float(row['r1'])

    retr_dir = os.path.join(ATTN_ROOT, cfg['model_dir'], 'head_rankings',
                            'retriever_best_insample')
    if not os.path.exists(retr_dir):
        raise FileNotFoundError(
            f'In-sample retriever ranking missing: {retr_dir}; create symlink '
            f'or set up the *_insample directory before using score_sum.')
    retr_csv = os.path.join(retr_dir, f'{phase}_proj_mean_insample.csv')
    retr = {}
    with open(retr_csv) as f:
        for row in csv.DictReader(f):
            retr[(int(row['layer']), int(row['head']))] = float(row['r1_insample'])

    all_heads = set(attn) | set(retr)
    scored = []
    for (l, h) in all_heads:
        score = attn.get((l, h), 0.0) + retr.get((l, h), 0.0)
        scored.append((l, h, score))
    scored.sort(key=lambda x: -x[2])
    top = scored[:k]
    return [(l, h) for l, h, _ in top], [(l, h, float(s)) for l, h, s in top]


def _top_k_interleave(cfg, k, phase, retriever_phase=None):
    """Top-k by *round-robin merge* of the attention-score and retriever
    rankings: pick attn #1, then retr #1, then attn #2, then retr #2, ...
    skipping any head already picked. The pointer that reached the duplicate
    advances; the alternation continues from the other side. Ties in K parity
    (e.g. K=5) yield ceil(K/2) attn + floor(K/2) retr unless duplicates push
    the count one way.

    `retriever_phase`, if given, overrides which retriever ranking CSV is used
    (e.g. 'generation' for the generation-phase-trained retriever, while
    keeping attn-score from the ICIR-MCQ `phase` ranking).
    """
    if retriever_phase is None:
        retriever_phase = phase
    coco_csv = get_resolver().pointer_csv(cfg['model_dir'], phase, require=False)
    if not os.path.exists(coco_csv):
        raise FileNotFoundError(
            f'ICIR-MCQ attn_score ranking missing: {coco_csv}; interleave '
            f'requires both rankings.')
    attn_rows = []
    with open(coco_csv) as f:
        for row in csv.DictReader(f):
            attn_rows.append((int(row['layer']), int(row['head']),
                              float(row['r1'])))
    attn_rows.sort(key=lambda x: -x[2])
    attn_list = [(l, h) for l, h, _ in attn_rows]

    ranking_name = cfg.get('retriever_ranking')
    if ranking_name is None:
        raise FileNotFoundError(
            f"No retriever ranking for {cfg['model_dir']}; interleave "
            f"requires both rankings.")
    model_abs = os.path.join(ATTN_ROOT, cfg['model_dir'])
    ranking_path = os.path.join(model_abs, 'head_rankings', ranking_name)
    if not os.path.exists(ranking_path):
        ranking_path = os.path.join(model_abs, ranking_name)
    if os.path.isdir(ranking_path):
        csv_path = os.path.join(ranking_path, f'{retriever_phase}_proj_mean_fullpool.csv')
        r1_col = 'r1_dedup'
    else:
        csv_path = ranking_path
        r1_col = 'recall@1'
    retr_rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            retr_rows.append((int(row['layer']), int(row['head']),
                              float(row[r1_col])))
    retr_rows.sort(key=lambda x: -x[2])
    retr_list = [(l, h) for l, h, _ in retr_rows]

    picked, info = [], []
    seen = set()
    i = j = 0
    turn = 'attn'
    while len(picked) < k and (i < len(attn_list) or j < len(retr_list)):
        if turn == 'attn':
            while i < len(attn_list) and attn_list[i] in seen:
                i += 1
            if i < len(attn_list):
                head = attn_list[i]
                picked.append(head); seen.add(head); info.append((*head, float(i + 1)))
                i += 1
            turn = 'retr'
        else:
            while j < len(retr_list) and retr_list[j] in seen:
                j += 1
            if j < len(retr_list):
                head = retr_list[j]
                picked.append(head); seen.add(head); info.append((*head, -float(j + 1)))
                j += 1
            turn = 'attn'
    return picked, info


def _random_k(k, num_layers, num_heads, seed=42):
    rng = random.Random(seed)
    all_heads = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    picked = rng.sample(all_heads, k)
    picked.sort()
    return picked, [(l, h, 0.0) for l, h in picked]


def _top_k_retriever(cfg, k, phase):
    ranking_name = cfg['retriever_ranking']
    if ranking_name is None:
        raise FileNotFoundError(f"No retriever ranking for {cfg['model_dir']}")
    model_abs = os.path.join(ATTN_ROOT, cfg['model_dir'])
    ranking_path = os.path.join(model_abs, 'head_rankings', ranking_name)
    if not os.path.exists(ranking_path):
        ranking_path = os.path.join(model_abs, ranking_name)
    if os.path.isdir(ranking_path):
        csv_path = os.path.join(ranking_path, f'{phase}_proj_mean_fullpool.csv')
        r1_col = 'r1_dedup'
    else:
        csv_path = ranking_path
        r1_col = 'recall@1'
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append((int(row['layer']), int(row['head']), float(row[r1_col])))
    rows.sort(key=lambda x: x[2], reverse=True)
    top = rows[:k]
    return [(l, h) for l, h, _ in top], top


def _top_k_attn_score(cfg, hs, k, phase):
    """Top-k by attention-score R@1, computed on ICIR-MCQ (COCO-MCQ).

    Per the project rule (memory: feedback_icir_mcq_general_head_set.md),
    ALL top-K head selections must use ICIR-MCQ-derived rankings — no
    per-benchmark tuning. If the ICIR-MCQ ranking CSV is missing for the
    target model, this is a hard error (do NOT silently fall back to a
    VHs-h5-derived per-haystack ranking, which would invalidate the
    "general retrieval head" claim).
    """
    coco_csv = get_resolver().pointer_csv(cfg['model_dir'], phase, require=False)
    if not os.path.exists(coco_csv):
        raise FileNotFoundError(
            f'ICIR-MCQ attn_score ranking missing: {coco_csv}\n'
            f'Run data/save_coco_head_outputs_mcq.py + the rank-extraction '
            f'pipeline for {cfg["model_dir"]} before any ablation. Per '
            f'feedback_icir_mcq_general_head_set.md, falling back to a '
            f'VHs-h5-derived ranking is forbidden.')
    rows = []
    with open(coco_csv) as f:
        for row in csv.DictReader(f):
            rows.append((int(row['layer']), int(row['head']), float(row['r1'])))
    rows.sort(key=lambda x: -x[2])
    return [(l, h) for l, h, _ in rows[:k]], rows[:k]


# ── Run one condition ──────────────────────────────────────────────────────────

def run_condition(backend: ModelBackend, cfg, hs, top_k, phase, ranking_source,
                  ablation_mode, samples, ablation_pos='cattok',
                  subset_tag='', random_seed=42, ablate_gen=False, out_root=None,
                  partition_n=100):
    num_layers = backend.num_layers
    num_heads = backend.num_heads
    head_dim = backend.head_dim
    layers = backend.layers

    top_heads, top_heads_info = get_top_k_heads(cfg, hs, top_k, phase=phase,
                                                 source=ranking_source,
                                                 random_seed=random_seed,
                                                 partition_n=partition_n)
    mask_dict = {}
    for l, h in top_heads:
        mask_dict.setdefault(l, []).append(h)

    tag_source = ranking_source if ranking_source != 'baseline' else 'baseline'
    tag_k = top_k if ranking_source != 'baseline' else 0
    # Encode random seed in the source tag so different seeds get separate
    # output files; the canonical seed (42) keeps the legacy filename.
    if ranking_source == 'random' and random_seed != 42:
        tag_source = f'random_seed{random_seed}'

    # Set-partition threshold suffix. Empty for the canonical N=100 (legacy
    # filenames preserved bit-for-bit) and for every N-independent source, so a
    # random/pointer/writer arm is never recomputed per threshold.
    part_suffix = ('' if partition_n == 100 or ranking_source not in PARTITION_SOURCES
                   else f'_N{partition_n}')

    print(f'\n{"="*60}')
    print(f'Condition: hs={hs}, top-{tag_k} {phase} ({tag_source})')
    if ranking_source in PARTITION_SOURCES:
        print(f'  set-partition N={partition_n}')
    elif partition_n != 100:
        print(f'  (N={partition_n} ignored: {ranking_source} is N-independent, '
              f'reusing canonical filename)')
    if top_heads_info:
        for l, h, r1 in top_heads_info[:10]:
            print(f'  L{l:02d}H{h:02d}  R@1={r1:.3f}')
        if len(top_heads_info) > 10:
            print(f'  ... ({len(top_heads_info)} total)')
        print(f'Layers affected: {sorted(mask_dict.keys())}')
        print(f'Total heads: {sum(len(v) for v in mask_dict.values())}')
    else:
        print('  No ablation (baseline)')

    out_dir = os.path.join(out_root or ATTN_ROOT, cfg['model_dir'], 'ablation')
    os.makedirs(out_dir, exist_ok=True)
    pos_suffix = (f'_{ablation_pos}'
                  if ablation_mode in ('catpos', 'mean_catpos', 'qzero_catpos')
                     and ablation_pos != 'cattok'
                  else '')
    gen_suffix = '_ablgen' if ablate_gen else ''
    tag = (f'hs{hs}_top{tag_k}_{phase}_{tag_source}_{ablation_mode}'
           f'{pos_suffix}{gen_suffix}{subset_tag}{part_suffix}')
    out_path = os.path.join(out_dir, f'{tag}.jsonl')

    done_ids = set()
    if os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                done_ids.add(json.loads(line)['id'])
        print(f'Resuming: {len(done_ids)} already done')

    remaining = sum(1 for s in samples if s['id'] not in done_ids)
    if remaining == 0:
        print(f'All {len(done_ids)} samples already done, skipping.')
        all_correct, all_total = 0, 0
        with open(out_path) as f:
            for line in f:
                r = json.loads(line)
                all_total += 1
                if r.get('correct'):
                    all_correct += 1
        acc = all_correct / max(all_total, 1)
        print(f'Accuracy: {all_correct}/{all_total} = {acc:.1%}')
        return acc, all_total, 0

    rng = random.Random(42)
    out_f = open(out_path, 'a')
    correct, total, nc, oom = 0, 0, 0, 0

    pos_holder = [None]
    img_ranges_holder = [None]
    # ── PAPER ablation: output-zero on o_proj input ──────────────────────────
    if ablation_mode == 'catpos':
        onwards = (ablation_pos == 'cattok_onwards')
        ablation_ctx = ablate_heads_catpos(
            layers, head_dim, mask_dict, pos_holder, onwards=onwards,
            ablate_gen=ablate_gen)
    # ── LEGACY ablations (kept for prior-experiment repro only) ───────────────
    elif ablation_mode == 'qzero_catpos':
        onwards = (ablation_pos == 'cattok_onwards')
        ablation_ctx = ablate_heads_qzero_catpos(
            layers, head_dim, mask_dict, pos_holder, onwards=onwards)
    elif ablation_mode == 'qzero_decode':
        ablation_ctx = ablate_heads_qzero_decode(layers, head_dim, mask_dict)
    elif ablation_mode == 'mean_catpos':
        onwards = (ablation_pos == 'cattok_onwards')
        mean_path = os.path.join(ATTN_ROOT, cfg['model_dir'], 'ablation',
                                 f'mean_head_output_cattok_hs{hs}_all.npz')
        mean_np = np.load(mean_path)['mean']
        mean_tensor = torch.from_numpy(mean_np).cuda()
        ablation_ctx = ablate_heads_mean_catpos(
            layers, head_dim, mask_dict, pos_holder, mean_tensor, onwards=onwards)
    elif ablation_mode == 'image_value':
        ablation_ctx = ablate_heads_image_value(layers, head_dim, mask_dict,
                                                img_ranges_holder)
    else:  # 'zero' — full-forward o_proj weight zero (LEGACY)
        ablation_ctx = ablate_heads_oproj(layers, head_dim, mask_dict)

    with ablation_ctx:
        for sample in tqdm(samples, desc=f'hs={hs} {tag_source} top-{tag_k}'):
            sid = sample['id']
            haystack_rels = list(sample['pos_image']) + list(sample['neg_image'])
            rng.shuffle(haystack_rels)
            needle_pos_idx = haystack_rels.index(sample['pos_image'][0])

            if sid in done_ids:
                continue

            question = sample['conversations'][0]['value']
            answer = sample['conversations'][1]['value'].strip().lower()
            needle = sample.get('needle', '')

            try:
                pil_images = [Image.open(os.path.join(COCO_ROOT, r)).convert('RGB')
                              for r in haystack_rels]
                inputs = backend.build_inputs(pil_images, question)
            except Exception as e:
                print(f'  Input build error {sid}: {e}')
                continue

            # Position hooks (per-sample) for catpos / image_value
            if ablation_mode in ('catpos', 'mean_catpos', 'qzero_catpos'):
                if ablation_pos == 'lastq':
                    p = compute_lastq_pos(backend, inputs['ids_list'])
                    pos_holder[0] = (p, p)
                elif needle:
                    pos_holder[0] = compute_needle_range(
                        backend, inputs['ids_list'], needle)
            if ablation_mode == 'image_value':
                img_ranges_holder[0] = inputs['image_ranges']

            try:
                stopping = StoppingCriteriaList([YesNoStop(backend.tokenizer)])
                result = backend.generate(inputs, max_new_tokens=128,
                                          capture_first_logits=False,
                                          stopping_criteria=stopping)
                response = result['response']
                pred = parse_yes_no(response)
                is_correct = (pred == answer) if pred else False
                total += 1
                if is_correct:
                    correct += 1
                if pred is None:
                    nc += 1
                out_f.write(json.dumps({
                    'id': sid, 'response': response,
                    'pred': pred, 'gt': answer,
                    'correct': is_correct, 'needle_pos': needle_pos_idx,
                }) + '\n')
                out_f.flush()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                oom += 1

            if total > 0 and total % 50 == 0:
                print(f'  [{total}] acc={correct/total:.1%} oom={oom} nc={nc}')

    out_f.close()

    all_correct, all_total = 0, 0
    with open(out_path) as f:
        for line in f:
            r = json.loads(line)
            all_total += 1
            if r.get('correct'):
                all_correct += 1
    acc = all_correct / max(all_total, 1)
    print(f'\nResult: hs={hs} {tag_source} top-{tag_k}')
    print(f'Accuracy: {all_correct}/{all_total} = {acc:.1%}  OOM: {oom}')
    print(f'Output: {out_path}')
    return acc, all_total, oom


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    cfg = MODEL_CONFIGS[args.model]
    hs_list = [int(x) for x in args.hs.split(',')]
    sources = [s.strip() for s in args.ranking_source.split(',')]
    top_k_list = [int(x) for x in str(args.top_k).split(',')]
    random_seeds = [int(x) for x in str(args.random_seeds).split(',')]

    retriever_dependent = {'retriever', 'retriever_gen', 'overlap',
                           'overlap_gen', 'interleave', 'interleave_gen'}
    if cfg.get('retriever_ranking') is None:
        skip = [s for s in sources if s in retriever_dependent]
        if skip:
            print(f'WARNING: No retriever ranking for {args.model}, skipping {skip}')
            sources = [s for s in sources if s not in skip]

    paper_mode = 'qzero_catpos'
    paper_pos = 'cattok_onwards'
    if args.ablation_mode != paper_mode or args.ablation_pos != paper_pos:
        print(f'\n[LEGACY MODE] --ablation-mode={args.ablation_mode} '
              f'--ablation-pos={args.ablation_pos}\n'
              f'              Paper-aligned default is '
              f'--ablation-mode={paper_mode} --ablation-pos={paper_pos}.\n'
              f'              Use this only for prior-experiment reproduction.')

    print(f'Model: {args.model} ({cfg["family"]})')
    print(f'Sweep: hs={hs_list} × sources={sources} × top_k={top_k_list}')
    if args.partition_n != 100:
        part_srcs = [s for s in sources if s in PARTITION_SOURCES]
        print(f'Set-partition N={args.partition_n} (paper default 100) '
              f'-> applies to {part_srcs or "no requested source"}; '
              f'outputs get an "_N{args.partition_n}" filename suffix.')

    print('\nLoading model...')
    backend = ModelBackend(args.model)
    print('Model loaded.\n')

    vhs_cache = {}
    subset_tag = '' if args.subset == 'VHs_large' else f'_{args.subset.lower()}'
    results = []

    for hs in hs_list:
        if hs not in vhs_cache:
            json_path = os.path.join(DATA_ROOT, 'single_needle', args.subset,
                                     f'visual_haystack_{hs}.json')
            samples = json.load(open(json_path))
            vhs_cache[hs] = samples
            print(f'Loaded hs={hs} ({args.subset}): {len(samples)} samples')

        samples = vhs_cache[hs]
        if args.n_samples is not None:
            if args.balanced:
                yes_s = [s for s in samples if s['conversations'][1]['value'].strip().lower() == 'yes']
                no_s = [s for s in samples if s['conversations'][1]['value'].strip().lower() == 'no']
                rng_bal = random.Random(42)
                rng_bal.shuffle(yes_s); rng_bal.shuffle(no_s)
                half = args.n_samples // 2
                samples = yes_s[:half] + no_s[:half]
                rng_bal.shuffle(samples)
                print(f'  Balanced sample: {half} yes + {half} no = {len(samples)} total')
            else:
                samples = samples[:args.n_samples]

        for top_k in top_k_list:
            for source in sources:
                # For random, run each requested seed; for other sources,
                # the seed is irrelevant.
                seed_iter = (random_seeds if source == 'random' else [42])
                for seed in seed_iter:
                    try:
                        acc, n_total, n_oom = run_condition(
                            backend, cfg, hs, top_k, args.phase, source,
                            args.ablation_mode, samples,
                            ablation_pos=args.ablation_pos,
                            subset_tag=subset_tag,
                            random_seed=seed,
                            ablate_gen=args.ablate_generation,
                            out_root=args.out_root,
                            partition_n=args.partition_n)
                        results.append({
                            'hs': hs, 'source': source, 'top_k': top_k,
                            'random_seed': seed if source == 'random' else None,
                            'partition_n': (args.partition_n
                                            if source in PARTITION_SOURCES else None),
                            'acc': acc, 'n': n_total, 'oom': n_oom,
                        })
                    except Exception as e:
                        print(f'ERROR: hs={hs} top_k={top_k} {source} seed={seed}: {e}')
                        import traceback; traceback.print_exc()

    print(f'\n\n{"="*70}')
    print(f'ABLATION SWEEP: {args.model} (top_k={args.top_k}, phase={args.phase})')
    print(f'{"="*70}')
    print(f'{"hs":>4}  {"source":>12}  {"top_k":>5}  {"acc":>8}  {"n":>5}  {"oom":>4}')
    print(f'{"-"*4}  {"-"*12}  {"-"*5}  {"-"*8}  {"-"*5}  {"-"*4}')
    for r in results:
        print(f'{r["hs"]:>4}  {r["source"]:>12}  {r["top_k"]:>5}  '
              f'{r["acc"]:>7.1%}  {r["n"]:>5}  {r["oom"]:>4}')

    gen_sfx = '_ablgen' if args.ablate_generation else ''
    part_sfx = '' if args.partition_n == 100 else f'_N{args.partition_n}'
    summary_path = os.path.join(args.out_root or ATTN_ROOT, cfg['model_dir'], 'ablation',
                                f'sweep_top{args.top_k}_{args.phase}_{args.ablation_mode}_{args.ablation_pos}{gen_sfx}{subset_tag}{part_sfx}_summary.json')
    json.dump(results, open(summary_path, 'w'), indent=2)
    print(f'\nSummary: {summary_path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--hs', type=str, default='10,5,3,2,20')
    parser.add_argument('--top-k', type=str, default='50',
                        help='Comma-separated list of K values (e.g., "5,10,20,30,50") '
                             'or single int.')
    parser.add_argument('--phase', default='cattok', choices=['cattok', 'generation'])
    parser.add_argument('--ranking-source', default='attn_score,retriever,random',
                        help='Comma-separated: attn_score,retriever,random,baseline')
    parser.add_argument('--ablation-mode', default='catpos',
                        choices=['catpos',  # PAPER default (output-zero)
                                 # LEGACY (do not use for new paper experiments):
                                 'zero', 'qzero_decode', 'qzero_catpos',
                                 'mean_catpos', 'image_value'],
                        help='catpos = paper-aligned output-zero on o_proj '
                             'input (use with --ablation-pos cattok_onwards). '
                             'All others are LEGACY, kept for prior-experiment '
                             'reproduction.')
    parser.add_argument('--n-samples', type=int, default=None)
    parser.add_argument('--balanced', action='store_true')
    parser.add_argument('--random-seeds', type=str, default='42',
                        help='Comma-separated seeds for `--ranking-source random` '
                             '(e.g., "42,1,7"). Each seed is run as a separate '
                             'condition with output filename `..._random_seedN_...` '
                             '(seed=42 keeps the legacy filename `..._random_...`).'
                             ' Ignored for non-random sources.')
    parser.add_argument('--ablation-pos', default='cattok_onwards',
                        choices=['cattok_onwards',  # PAPER default
                                 'cattok', 'lastq'],  # LEGACY
                        help='cattok_onwards = paper-aligned (needle pos to '
                             'sequence end). cattok and lastq are LEGACY.')
    parser.add_argument('--ablate-generation', action='store_true',
                        help='Extend output-zero ablation through decode steps '
                             '(reviewer-requested; default off = paper setting)')
    parser.add_argument('--partition-n', type=int, default=100,
                        help='Set-partition threshold N for the set-partition '
                             'sources (overlap_set, overlap_set_attn, '
                             'attn_only_set, retr_only_set and their paper '
                             'aliases). Paper default 100. N!=100 appends an '
                             '"_N{n}" suffix to the output filename so alt-'
                             'threshold runs never collide with the canonical '
                             'top-100 results. N-independent sources (random, '
                             'attn_score, retriever, ...) ignore this flag and '
                             'keep their canonical filename.')
    parser.add_argument('--out-root', default=None,
                        help='Override output root (default: ATTN_ROOT, may be read-only)')
    parser.add_argument('--subset', default='VHs_large',
                        choices=['VHs_large', 'VHs_small'],
                        help='Which VHs subset to evaluate. VHs_small uses 100 '
                             'samples per hs with haystacks filtered from the '
                             'VHs_large shuffle (matches baseline behavior).')
    args = parser.parse_args()
    main(args)
