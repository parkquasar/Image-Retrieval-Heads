"""
Build COCO retrieval dataset — MCQ variant.

Same image-selection logic as v2 (no image reuse), but each sample includes
a multiple-choice caption question:

  Q: "Which caption best describes the image containing [category]?"
  A/B/C/D: one is the GT caption of the needle image,
            three are captions from *other images of the same category*
            (so the distractors are semantically related but wrong).

The answer position (which of A/B/C/D is correct) is rotated across samples.

New fields added to each record (on top of v2 fields):
  mcq_options : {"A": str, "B": str, "C": str, "D": str}
  mcq_answer  : "A" | "B" | "C" | "D"   (correct option)
  needle_caption : str                   (GT caption, also stored in mcq_options[mcq_answer])

Usage:
  python data/build_coco_retrieval_dataset_mcq.py \
      --output-n5 outputs/datasets/coco_retrieval_n5_mcq.jsonl \
      --output-n6 outputs/datasets/coco_retrieval_n6_mcq.jsonl \
      --n-samples-n5 3000 \
      --n-samples-n6 3000
"""

import os
import sys
import json
import random
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

COCO_ANN_ROOT = os.environ.get('COCO_ANN_ROOT', './data/COCO/annotations')   # captions_*.json, instances_*.json (2017)
COCO_IMG_ROOT = os.environ.get('COCO_IMG_ROOT', './data/COCO/images')
MM_NIAH_ANN   = os.environ.get('MM_NIAH_ANN', './data/MM-NIAH/mm_niah_val/annotations/retrieval-image.jsonl')   # MM-NIAH val annotation

MCQ_LABELS = ['A', 'B', 'C', 'D']


# ── Caption index ──────────────────────────────────────────────────────────────

def build_caption_index():
    """Build img_id → list[caption] and rel_path → img_id mappings."""
    img_captions = defaultdict(list)   # img_id → [caption, ...]
    path_to_id   = {}                  # "train2017/filename.jpg" → img_id

    for split in ['train2017', 'val2017']:
        with open(os.path.join(COCO_ANN_ROOT, f'captions_{split}.json')) as f:
            cap = json.load(f)
        for ann in cap['annotations']:
            img_captions[ann['image_id']].append(ann['caption'])

        with open(os.path.join(COCO_ANN_ROOT, f'instances_{split}.json')) as f:
            inst = json.load(f)
        for img in inst['images']:
            path_to_id[f"{split}/{img['file_name']}"] = img['id']

    return dict(img_captions), path_to_id


# ── Category index (identical to v2) ──────────────────────────────────────────

def build_category_index(min_images_per_cat: int):
    cat_names    = {}
    cat_to_imgs  = defaultdict(list)
    img_all_cats = {}

    for split in ['train2017', 'val2017']:
        ann_file = os.path.join(COCO_ANN_ROOT, f'instances_{split}.json')
        print(f'  Loading {ann_file} ...')
        with open(ann_file) as f:
            coco = json.load(f)

        for c in coco['categories']:
            cat_names[c['id']] = c['name']

        img_id_to_file = {img['id']: img['file_name'] for img in coco['images']}

        img_cat_set   = defaultdict(set)
        img_cat_count = defaultdict(Counter)
        for ann in coco['annotations']:
            img_cat_set[ann['image_id']].add(ann['category_id'])
            img_cat_count[ann['image_id']][ann['category_id']] += 1

        for img_id, cat_counter in img_cat_count.items():
            dom_cat  = cat_counter.most_common(1)[0][0]
            rel_path = f'{split}/{img_id_to_file[img_id]}'
            cat_to_imgs[dom_cat].append(rel_path)
            img_all_cats[rel_path] = frozenset(img_cat_set[img_id])

    valid_cats = sorted(
        cat_id for cat_id, imgs in cat_to_imgs.items()
        if len(imgs) >= min_images_per_cat
    )
    print(f'  {len(valid_cats)} categories with >= {min_images_per_cat} images')

    print('  Pre-computing exclusion-filtered image lists ...')
    cat_to_imgs_excl = {}
    for dom_cat in valid_cats:
        imgs = cat_to_imgs[dom_cat]
        cat_to_imgs_excl[dom_cat] = {}
        for excl_cat in valid_cats:
            if excl_cat == dom_cat:
                continue
            filtered = [img for img in imgs
                        if excl_cat not in img_all_cats.get(img, frozenset())]
            cat_to_imgs_excl[dom_cat][excl_cat] = filtered

    return cat_names, dict(cat_to_imgs), img_all_cats, cat_to_imgs_excl, valid_cats


# ── MM-NIAH context loader (identical to v2) ──────────────────────────────────

def load_mmniah_samples(n_images_filter=None):
    samples = []
    with open(MM_NIAH_ANN) as f:
        for line in f:
            s = json.loads(line)
            n_images = len(s['images_list'])
            if n_images < 1:
                continue
            if n_images_filter is not None and n_images != n_images_filter:
                continue
            samples.append({
                'original_id':    s['id'],
                'context':        s['context'],
                'n_images':       n_images,
                'placed_depth':   s['meta']['placed_depth'][0],
                'context_length': s['meta']['context_length'],
            })
    return samples


# ── MCQ option builder ─────────────────────────────────────────────────────────

def _keyword_captions(img_id, captions_list, keyword):
    """Return captions from captions_list that contain keyword (case-insensitive)."""
    return [c for c in captions_list if keyword in c.lower()]


def build_mcq_options(needle_img_path, needle_cat_id, needle_cat_name, cat_to_imgs,
                      path_to_id, img_captions, rng, answer_label,
                      cat_kw_caption_pool):
    """
    Build 4-option MCQ where ALL captions contain the category keyword.

    needle_img_path   : rel path of the needle image
    needle_cat_id     : dominant category id of the needle
    needle_cat_name   : human-readable category name (e.g. 'sports ball')
    cat_to_imgs       : cat_id → [rel_path, ...]
    cat_kw_caption_pool : cat_id → list of (rel_path, caption) with keyword match
                          (precomputed; distractor captions sampled from here)
    answer_label      : which label (A/B/C/D) gets the correct caption

    Returns: (mcq_options dict, needle_caption str)
             or None if not enough keyword-matching captions available.
    """
    # Use last word of category name as keyword: 'sports ball' → 'ball'
    keyword = needle_cat_name.split()[-1].lower()

    needle_id = path_to_id.get(needle_img_path)
    if needle_id is None or needle_id not in img_captions:
        return None

    needle_kw_caps = _keyword_captions(needle_id, img_captions[needle_id], keyword)
    if not needle_kw_caps:
        return None
    needle_caption = rng.choice(needle_kw_caps)

    # Pick 3 distractor captions: from same category, different image from needle
    pool = [
        (p, c) for p, c in cat_kw_caption_pool.get(needle_cat_id, [])
        if p != needle_img_path
    ]
    if len(pool) < 3:
        return None

    chosen = rng.sample(pool, 3)
    distractor_captions = [c for _, c in chosen]

    # Assemble options
    others = iter(distractor_captions)
    options = {}
    for label in MCQ_LABELS:
        if label == answer_label:
            options[label] = needle_caption
        else:
            options[label] = next(others)

    return options, needle_caption


# ── Needle schedule (identical to v2) ─────────────────────────────────────────

def make_needle_schedule(valid_cats, n_total, rng):
    repeats = (n_total + len(valid_cats) - 1) // len(valid_cats)
    schedule = valid_cats * repeats
    rng.shuffle(schedule)
    return schedule[:n_total]


# ── Main dataset builder ───────────────────────────────────────────────────────

def build_dataset(
    n_images: int,
    n_samples: int,
    cat_names, cat_to_imgs, img_all_cats, cat_to_imgs_excl, valid_cats,
    used_images: set,
    mmniah_samples,
    path_to_id, img_captions,
    cat_kw_caption_pool,
    rng: random.Random,
    start_id: int = 0,
):
    cat_pools = {}
    for cat in valid_cats:
        keyword = cat_names[cat].split()[-1].lower()
        imgs = [
            img for img in cat_to_imgs[cat]
            if img not in used_images
            and path_to_id.get(img) in img_captions
            and _keyword_captions(path_to_id[img], img_captions[path_to_id[img]], keyword)
        ]
        rng.shuffle(imgs)
        cat_pools[cat] = imgs

    cat_excl_pools = {}
    for dom_cat in valid_cats:
        cat_excl_pools[dom_cat] = {}
        for excl_cat in valid_cats:
            if excl_cat == dom_cat:
                continue
            imgs = [img for img in cat_to_imgs_excl[dom_cat].get(excl_cat, [])
                    if img not in used_images]
            rng.shuffle(imgs)
            cat_excl_pools[dom_cat][excl_cat] = imgs

    mmniah_cycle = list(mmniah_samples)
    rng.shuffle(mmniah_cycle)
    mmniah_idx = 0

    needle_schedule = make_needle_schedule(valid_cats, n_samples * 10, rng)
    schedule_idx = 0

    # Perfectly uniform answer label distribution
    base_ans, rem_ans = divmod(n_samples, len(MCQ_LABELS))
    answer_cycle = MCQ_LABELS * base_ans + MCQ_LABELS[:rem_ans]
    rng.shuffle(answer_cycle)

    # Perfectly uniform target_index distribution
    base_pos, rem_pos = divmod(n_samples, n_images)
    position_cycle = list(range(n_images)) * base_pos + list(range(rem_pos))
    rng.shuffle(position_cycle)

    records = []
    failures = 0

    while len(records) < n_samples:
        if schedule_idx >= len(needle_schedule):
            print(f'  [warn] needle schedule exhausted after {len(records)} samples')
            break

        needle_cat = needle_schedule[schedule_idx]
        schedule_idx += 1

        # Pick needle image
        needle_img = None
        pool = cat_pools.get(needle_cat, [])
        while pool:
            candidate = pool.pop(0)
            if candidate not in used_images:
                needle_img = candidate
                break
        if needle_img is None:
            failures += 1
            continue

        # Pick distractor images
        other_cats_pool = [c for c in valid_cats if c != needle_cat]
        if len(other_cats_pool) < n_images - 1:
            failures += 1
            continue
        other_cats = rng.sample(other_cats_pool, n_images - 1)

        other_imgs = []
        ok = True
        for oc in other_cats:
            img = None
            excl_pool = cat_excl_pools.get(oc, {}).get(needle_cat, [])
            while excl_pool:
                candidate = excl_pool.pop(0)
                if candidate not in used_images:
                    img = candidate
                    break
            if img is None:
                ok = False
                break
            other_imgs.append(img)

        if not ok:
            used_images.discard(needle_img)
            cat_pools[needle_cat].insert(0, needle_img)
            failures += 1
            continue

        # Build MCQ options
        answer_label = answer_cycle[len(records)]
        mcq_result = build_mcq_options(
            needle_img, needle_cat, cat_names[needle_cat], cat_to_imgs,
            path_to_id, img_captions, rng, answer_label, cat_kw_caption_pool
        )
        if mcq_result is None:
            # Return images to pools
            used_images.discard(needle_img)
            cat_pools[needle_cat].insert(0, needle_img)
            failures += 1
            continue

        mcq_options, needle_caption = mcq_result

        # Commit images
        used_images.add(needle_img)
        for img in other_imgs:
            used_images.add(img)

        orig = mmniah_cycle[mmniah_idx % len(mmniah_cycle)]
        mmniah_idx += 1

        target_index = position_cycle[len(records)]

        images_list = list(other_imgs)
        images_list.insert(target_index, needle_img)
        assert len(images_list) == n_images

        records.append({
            'id':               start_id + len(records),
            'original_id':      orig['original_id'],
            'context':          orig['context'],
            'n_images':         n_images,
            'placed_depth':     orig['placed_depth'],
            'target_index':     target_index,
            'target_category':  cat_names[needle_cat],
            'images_list':      images_list,
            'context_length':   orig['context_length'],
            'needle_caption':   needle_caption,
            'mcq_options':      mcq_options,
            'mcq_answer':       answer_label,
        })

    print(f'  Generated {len(records)} samples '
          f'({failures} failures, {len(used_images)} total images used)')
    return records


# ── Main ──────────────────────────────────────────────────────────────────────

def main(args):
    print('Building COCO category index...')
    cat_names, cat_to_imgs, img_all_cats, cat_to_imgs_excl, valid_cats = \
        build_category_index(args.min_images_per_cat)

    print('Building caption index...')
    img_captions, path_to_id = build_caption_index()
    print(f'  {len(img_captions)} images with captions')

    print('Loading MM-NIAH samples...')
    mmniah_n5 = load_mmniah_samples(n_images_filter=5)
    mmniah_n6 = load_mmniah_samples(n_images_filter=6)
    print(f'  n5 contexts: {len(mmniah_n5)}, n6 contexts: {len(mmniah_n6)}')

    # Precompute per-category keyword caption pool (all images, not deduped)
    print('Building keyword caption pools...')
    cat_kw_caption_pool = {}
    for cat_id in valid_cats:
        keyword = cat_names[cat_id].split()[-1].lower()
        pool = []
        for img_path in cat_to_imgs.get(cat_id, []):
            img_id = path_to_id.get(img_path)
            if img_id is None:
                continue
            kw_caps = _keyword_captions(img_id, img_captions.get(img_id, []), keyword)
            for cap in kw_caps:
                pool.append((img_path, cap))
        cat_kw_caption_pool[cat_id] = pool
    print(f'  Done. Total (cat, caption) pairs: {sum(len(v) for v in cat_kw_caption_pool.values())}')

    rng = random.Random(args.seed)
    used_images = set()  # shared: scene images only

    print(f'\nBuilding n5 MCQ dataset ({args.n_samples_n5} samples)...')
    records_n5 = build_dataset(
        n_images=5,
        n_samples=args.n_samples_n5,
        cat_names=cat_names, cat_to_imgs=cat_to_imgs,
        img_all_cats=img_all_cats, cat_to_imgs_excl=cat_to_imgs_excl,
        valid_cats=valid_cats,
        used_images=used_images,
        mmniah_samples=mmniah_n5,
        path_to_id=path_to_id, img_captions=img_captions,
        cat_kw_caption_pool=cat_kw_caption_pool,
        rng=rng,
        start_id=0,
    )

    print(f'\nBuilding n6 MCQ dataset ({args.n_samples_n6} samples)...')
    records_n6 = build_dataset(
        n_images=6,
        n_samples=args.n_samples_n6,
        cat_names=cat_names, cat_to_imgs=cat_to_imgs,
        img_all_cats=img_all_cats, cat_to_imgs_excl=cat_to_imgs_excl,
        valid_cats=valid_cats,
        used_images=used_images,
        mmniah_samples=mmniah_n6,
        path_to_id=path_to_id, img_captions=img_captions,
        cat_kw_caption_pool=cat_kw_caption_pool,
        rng=rng,
        start_id=len(records_n5),
    )

    # Verify no image overlap
    all_imgs_n5 = set(img for r in records_n5 for img in r['images_list'])
    all_imgs_n6 = set(img for r in records_n6 for img in r['images_list'])
    overlap = all_imgs_n5 & all_imgs_n6
    print(f'\nVerification:')
    print(f'  n5 unique images: {len(all_imgs_n5)}')
    print(f'  n6 unique images: {len(all_imgs_n6)}')
    print(f'  cross-dataset overlap: {len(overlap)}')
    assert len(overlap) == 0, f'BUG: {len(overlap)} images shared between n5 and n6!'

    for tag, records in [('n5', records_n5), ('n6', records_n6)]:
        img_counter = Counter(img for r in records for img in r['images_list'])
        dupes = {img: cnt for img, cnt in img_counter.items() if cnt > 1}
        print(f'  {tag} within-dataset duplicates: {len(dupes)}')
        assert len(dupes) == 0

    # MCQ answer distribution
    for tag, records in [('n5', records_n5), ('n6', records_n6)]:
        dist = Counter(r['mcq_answer'] for r in records)
        print(f'  {tag} answer distribution: {dict(sorted(dist.items()))}')

    # target_index distribution
    for tag, records in [('n5', records_n5), ('n6', records_n6)]:
        dist = Counter(r['target_index'] for r in records)
        n = records[0]['n_images']
        print(f'  {tag} target_index distribution:')
        for i in range(n):
            cnt = dist.get(i, 0)
            print(f'    pos {i}: {cnt} ({100*cnt/len(records):.1f}%)')

    print(f'  All checks passed.')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_n5)), exist_ok=True)
    with open(args.output_n5, 'w') as f:
        for r in records_n5:
            f.write(json.dumps(r) + '\n')
    print(f'\nSaved {len(records_n5)} n5 samples → {args.output_n5}')

    os.makedirs(os.path.dirname(os.path.abspath(args.output_n6)), exist_ok=True)
    with open(args.output_n6, 'w') as f:
        for r in records_n6:
            f.write(json.dumps(r) + '\n')
    print(f'Saved {len(records_n6)} n6 samples → {args.output_n6}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-n5',
                        default='outputs/datasets/coco_retrieval_n5_mcq.jsonl')
    parser.add_argument('--output-n6',
                        default='outputs/datasets/coco_retrieval_n6_mcq.jsonl')
    parser.add_argument('--n-samples-n5', type=int, default=3000)
    parser.add_argument('--n-samples-n6', type=int, default=3000)
    parser.add_argument('--min-images-per-cat', type=int, default=50)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
