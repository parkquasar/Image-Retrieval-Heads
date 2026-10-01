"""
Head Output DataLoader for Retriever Training (MCQ format)

HDF5 structure (save_coco_head_outputs_mcq.py):
    sample_{id}/
        attrs: sample_id, original_id, target_index, target_category, context_length
        image_embeddings/
            patch_mean: (n_images, D_vit)   ← ViT patch mean pool (before projector)
            proj_mean:  (n_images, D_lm)    ← projected visual token mean pool (after projector)
        cattok/layer_X: (1, num_heads, head_dim)   ← stored once per sample
        ansA/ ansB/ ansC/ ansD/
            attrs: answer_label, response, correct
            generation/layer_X: (1, num_heads, head_dim)

phase:    'cattok'    → 1 item per sample  (shared across all MCQ variants)
          'generation'→ 4 items per sample (one per variant ansA/B/C/D, data augmentation)
emb_type: 'patch_mean' or 'proj_mean'
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from collections import defaultdict
from typing import List, Optional, Dict

MCQ_ANSWERS = ['ansA', 'ansB', 'ansC', 'ansD']


def get_h5_dimensions(h5_path: str, phase: str = 'generation',
                      emb_type: str = 'patch_mean') -> Dict[str, int]:
    """
    Read dimension metadata from an MCQ-format HDF5 file.

    Returns:
        num_layers, num_heads, head_dim, image_dim, num_samples
    """
    with h5py.File(h5_path, 'r') as f:
        sample_names = sorted(f.keys())
        if not sample_names:
            raise ValueError(f"No samples in {h5_path}")
        grp = f[sample_names[0]]

        if phase == 'cattok':
            phase_grp = grp['cattok']
        else:  # generation
            ans_keys = sorted(k for k in grp.keys() if k.startswith('ans'))
            if not ans_keys:
                raise ValueError(f"No ansX groups found in {sample_names[0]}")
            phase_grp = grp[ans_keys[0]]['generation']

        layer_keys = [k for k in phase_grp.keys() if k.startswith('layer_')]
        num_layers = len(layer_keys)
        first_layer = phase_grp['layer_0'][:]   # (1, num_heads, head_dim)
        _, num_heads, head_dim = first_layer.shape

        emb_arr = grp['image_embeddings'][emb_type][:]  # (n_imgs, image_dim)
        image_dim = emb_arr.shape[1]

        return {
            'num_layers': num_layers,
            'num_heads': num_heads,
            'head_dim': head_dim,
            'image_dim': image_dim,
            'num_samples': len(sample_names),
        }


class CombinedHeadOutputDataset(Dataset):
    """
    Dataset over one or more MCQ-format HDF5 files for a given (phase, emb_type) pair.

    For phase='cattok':    1 item per sample
    For phase='generation': 4 items per sample (ansA / ansB / ansC / ansD)

    Train/test split is performed by sample_id, so all variants of a sample
    always land in the same split.

    All data is pre-loaded into numpy arrays during __init__ so that __getitem__
    is pure array indexing with no HDF5 overhead during training.
    """
    def __init__(
        self,
        h5_paths: List[str],
        layer_idx: int = 0,
        head_idx: Optional[int] = None,
        phase: str = 'generation',
        emb_type: str = 'patch_mean',
        split: Optional[str] = None,
        train_ratio: float = 0.8,
        seed: int = 42,
        verbose: bool = True,
        # kept for API compatibility, not used in MCQ format
        exclude_single_image: bool = False,
    ):
        self.layer_idx = layer_idx
        self.head_idx = head_idx
        self.phase = phase
        self.emb_type = emb_type

        h5_files = [h5py.File(p, 'r') for p in h5_paths]

        # ── Build item list ────────────────────────────────────────────────────
        # (file_idx, sample_name, ans_key_or_None)
        all_items = []
        for file_idx, h5f in enumerate(h5_files):
            for sample_name in sorted(h5f.keys()):
                grp = h5f[sample_name]
                if 'image_embeddings' not in grp:
                    continue
                if emb_type not in grp['image_embeddings']:
                    continue
                if phase == 'cattok':
                    if 'cattok' not in grp:
                        continue
                    all_items.append((file_idx, sample_name, None))
                else:  # generation
                    for ans in MCQ_ANSWERS:
                        if ans in grp and 'generation' in grp[ans]:
                            all_items.append((file_idx, sample_name, ans))

        # ── Train/test split by sample_id ──────────────────────────────────────
        if split is not None:
            unique_ids = sorted(set(int(name.split('_')[1]) for _, name, _ in all_items))
            rng = np.random.default_rng(seed)
            perm = rng.permutation(len(unique_ids))
            n_train = int(len(unique_ids) * train_ratio)
            if split == 'train':
                selected_ids = set(unique_ids[i] for i in perm[:n_train])
            elif split == 'test':
                selected_ids = set(unique_ids[i] for i in perm[n_train:])
            else:
                raise ValueError(f"split must be 'train', 'test', or None, got {split!r}")
            items = [(fi, name, ans) for fi, name, ans in all_items
                     if int(name.split('_')[1]) in selected_ids]
        else:
            items = all_items

        # Keep items for UniqueGtBatchSampler (needs sample_name to extract sample_id)
        self.items = items

        if verbose:
            n_samples = len(set(name for _, name, _ in items))
            print(f"Dataset [{phase}/{emb_type}]: "
                  f"{len(items)} items ({n_samples} unique samples, split={split})")

        # ── Pre-load all data into numpy arrays ────────────────────────────────
        # After this, h5 files are closed and __getitem__ does pure array indexing.
        n = len(items)
        self._sample_ids = np.empty(n, dtype=np.int64)
        self._gt_embeddings  = None   # allocated after first sample
        self._head_outputs   = None
        self._neg_embeddings: List[np.ndarray] = [None] * n  # variable-length list

        for i, (file_idx, sample_name, ans_key) in enumerate(items):
            grp = h5_files[file_idx][sample_name]

            # sample_id
            self._sample_ids[i] = int(grp.attrs['sample_id'])

            # head output: (1, num_heads, head_dim) → slice to (head_dim,) or (num_heads, head_dim)
            if phase == 'cattok':
                layer_data = grp['cattok'][f'layer_{layer_idx}'][:]
            else:
                layer_data = grp[ans_key]['generation'][f'layer_{layer_idx}'][:]

            if head_idx is not None:
                ho = layer_data[0, head_idx, :]   # (head_dim,)
            else:
                ho = layer_data[0]                 # (num_heads, head_dim)

            if self._head_outputs is None:
                self._head_outputs = np.empty((n,) + ho.shape, dtype=np.float32)
            self._head_outputs[i] = ho

            # image embeddings → gt + negatives
            emb = grp['image_embeddings'][emb_type][:]  # (n_images, image_dim)
            target_idx = int(grp.attrs['target_index'])

            if self._gt_embeddings is None:
                self._gt_embeddings = np.empty((n, emb.shape[1]), dtype=np.float32)
            self._gt_embeddings[i] = emb[target_idx]

            neg_mask = np.ones(len(emb), dtype=bool)
            neg_mask[target_idx] = False
            self._neg_embeddings[i] = emb[neg_mask].astype(np.float32)

        for h5f in h5_files:
            h5f.close()

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'sample_id':          int(self._sample_ids[idx]),
            'head_output':        torch.from_numpy(self._head_outputs[idx]),
            'gt_embedding':       torch.from_numpy(self._gt_embeddings[idx]),
            'negative_embeddings': torch.from_numpy(self._neg_embeddings[idx]),
        }


class UniqueGtBatchSampler(Sampler):
    """
    For generation phase: ensures each batch contains at most one variant
    per sample_id, preventing false negatives in InfoNCE loss.

    Strategy: round-robin across sample groups → consecutive batch_size
    items always come from batch_size distinct sample_ids (assuming
    num_unique_samples >= batch_size, which is always true here).
    """
    def __init__(self, dataset: 'CombinedHeadOutputDataset', batch_size: int,
                 drop_last: bool = False, seed: int = 42):
        groups: dict = defaultdict(list)
        for idx, (_, sample_name, _) in enumerate(dataset.items):
            sid = int(sample_name.split('_')[1])
            groups[sid].append(idx)
        self.groups    = list(groups.values())
        self.batch_size = batch_size
        self.drop_last  = drop_last
        self.seed       = seed
        self._epoch     = 0

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self._epoch)
        # Shuffle within each group, then shuffle group order
        groups = [rng.permutation(g).tolist() for g in self.groups]
        order  = rng.permutation(len(groups)).tolist()
        groups = [groups[i] for i in order]

        # Round-robin: item i from group g → position g + i * num_groups
        # → within any window of batch_size consecutive items, all sample_ids differ
        sequence: List[int] = []
        max_len = max(len(g) for g in groups)
        for i in range(max_len):
            for g in groups:
                if i < len(g):
                    sequence.append(g[i])

        for start in range(0, len(sequence), self.batch_size):
            batch = sequence[start:start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __len__(self) -> int:
        n = sum(len(g) for g in self.groups)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size


def collate_fn_variable_negatives(batch: List[Dict]) -> Dict:
    """Collate with variable-length negatives kept as a list of tensors."""
    return {
        'sample_ids': [item['sample_id'] for item in batch],
        'head_outputs': torch.stack([item['head_output'] for item in batch]),
        'gt_embeddings': torch.stack([item['gt_embedding'] for item in batch]),
        'negative_embeddings': [item['negative_embeddings'] for item in batch],
    }


def get_combined_dataloader(
    h5_paths: List[str],
    layer_idx: int = 0,
    head_idx: Optional[int] = None,
    phase: str = 'generation',
    emb_type: str = 'patch_mean',
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    split: Optional[str] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    verbose: bool = True,
    drop_last: bool = False,
    exclude_single_image: bool = False,   # API compat
) -> DataLoader:
    dataset = CombinedHeadOutputDataset(
        h5_paths=h5_paths,
        layer_idx=layer_idx,
        head_idx=head_idx,
        phase=phase,
        emb_type=emb_type,
        split=split,
        train_ratio=train_ratio,
        seed=seed,
        verbose=verbose,
    )

    # For generation phase training, use UniqueGtBatchSampler to prevent
    # multiple variants of the same sample (identical gt) from appearing
    # in the same batch, which would cause false negatives in InfoNCE loss.
    if phase == 'generation' and shuffle:
        sampler = UniqueGtBatchSampler(dataset, batch_size=batch_size,
                                       drop_last=drop_last, seed=seed)
        return DataLoader(
            dataset,
            batch_sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn_variable_negatives,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn_variable_negatives,
        drop_last=drop_last,
    )
