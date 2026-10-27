# Heads That Write, Not Just Point: Image Retrieval Heads in Vision-Language Models

Code and data for the NeurIPS 2026 paper.

- Dataset (ICIR-MCQ, 6,000 samples, CC BY 4.0): https://huggingface.co/datasets/jerryray/ICIR-MCQ
- Code: this repository (MIT, see `LICENSE`; third-party notices in `THIRD_PARTY_NOTICES.md`) The pipeline finds *pointer heads* (attention mass
on the target image) and *writer heads* (a lightweight retriever trained on the head output can
recover the target image) in InternVL3.5 and Qwen3-VL, then checks causal relevance with
output-zero ablations on Visual Haystacks.

## What is here

```
data/      build_coco_retrieval_dataset_mcq.py   ICIR-MCQ builder (COCO + MM-NIAH contexts)
           save_coco_head_outputs_mcq.py         per-head outputs + attention scores on ICIR-MCQ
           save_vhs_head_outputs.py              same on Visual Haystacks
training/  train_retrieval_heads.py              writer probe (per-head dual encoder, InfoNCE)
           rank_heads_attn_score.py              pointer ranking
           rank_heads_retriever.py               writer ranking
eval/      eval_vhs_head_ablation.py             output-zero ablation on Visual Haystacks
           eval_vhs_external_retrievers.py       CLIP / SigLIP baselines (80-template ensemble)
           eval_vhs_qwen3vl_embedder.py          Qwen3-VL-Embedding baseline
           build_topk_sweep_table.py             aggregate top-K ablations into LaTeX
           build_setpartition_view.py            aggregate set-partition ablations into LaTeX + figures
utils/     model_backend.py, model_utils.py      InternVL3.5 / Qwen3-VL loading, chat, needle positions
assets/    head_rankings/<model>/                the pointer and writer rankings reported in the paper
```

`assets/head_rankings/SUMMARY.md` lists the top heads and the pointer-writer overlap per model, so
you can run the ablations without re-extracting or re-training anything.

## Setup

```bash
conda create -n irh python=3.11 -y && conda activate irh
pip install -r requirements.txt
pip install flash-attn --no-build-isolation      # optional; set ATTN_IMPL=sdpa without it
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab')"
```

Paths are read from environment variables (defaults in parentheses):

| variable | meaning |
|---|---|
| `ATTN_ROOT` | where all outputs go: H5 extractions, probes, rankings, ablation JSONL (`./outputs`) |
| `DATA_ROOT` | Visual Haystacks root, from https://github.com/visual-haystacks/vhs_benchmark (`./data/visual_haystacks`) |
| `COCO_ROOT` | COCO 2017 images, `train2017/` and `val2017/` (`./data/COCO/images`) |
| `COCO_ANN_ROOT`, `MM_NIAH_ANN` | only for rebuilding ICIR-MCQ: COCO 2017 annotations and the MM-NIAH `retrieval-image.jsonl` |
| `HF_CACHE` | HuggingFace hub cache (`~/.cache/huggingface/hub`) |

Model snapshots are pinned to specific commits in `utils/model_utils.py`. Model aliases:
`internvl-8b`, `internvl-4b`, `qwen3vl-8b`, `qwen3vl-4b` (output directories `internvl3_5_8b` etc.).

## ICIR-MCQ

The 6,000 samples used in the paper are on Hugging Face: https://huggingface.co/datasets/jerryray/ICIR-MCQ
(`coco_retrieval_n5n6_mcq.jsonl`, 3,000 with five images, then 3,000 with six; data card and
checksums included). Images are referenced by COCO-relative path, so `COCO_ROOT` is still needed.

```bash
huggingface-cli download jerryray/ICIR-MCQ coco_retrieval_n5n6_mcq.jsonl --repo-type dataset \
    --local-dir $ATTN_ROOT/datasets
```

The builder reproduces the file bit for bit (seed 42); it writes the two halves separately and
their concatenation equals the shipped file:

```bash
python data/build_coco_retrieval_dataset_mcq.py \
    --output-n5 $ATTN_ROOT/datasets/coco_retrieval_n5_mcq.jsonl \
    --output-n6 $ATTN_ROOT/datasets/coco_retrieval_n6_mcq.jsonl --seed 42
```

## Pipeline

All commands below use `internvl-8b`; repeat for the other aliases.

**1. Extract head outputs and attention scores on ICIR-MCQ** (about 8 s per sample on an L40;
shard with `--num-workers N --worker-id k`, one GPU each, then keep the shards in
`$ATTN_ROOT/<model_dir>/coco_mcq/`).

```bash
CUDA_VISIBLE_DEVICES=0 python data/save_coco_head_outputs_mcq.py --model internvl-8b \
    --datasets $ATTN_ROOT/datasets/coco_retrieval_n5n6_mcq.jsonl
```

**2. Pointer ranking** (no training):

```bash
python training/rank_heads_attn_score.py --model-dir $ATTN_ROOT/internvl3_5_8b --source coco_mcq --phases cattok
# -> $ATTN_ROOT/internvl3_5_8b/head_rankings/attn_score_coco_mcq/cattok.csv
```

**3. Writer probe and writer ranking.** The paper configuration is a 1-layer projector,
hidden 512, projection 256, L2-normalised, batch 256, lr 1e-3, 50 epochs
(Qwen3-VL-8B used batch 128).

```bash
python training/train_retrieval_heads.py --model-dir $ATTN_ROOT/internvl3_5_8b \
    --phases cattok --emb_types proj_mean --arch standard --num_mlp_layers 1 \
    --batch_size 256 --lr 1e-3 --epochs 50 --gpu_ids 0 1 --processes_per_gpu 2
python training/rank_heads_retriever.py --model-dir $ATTN_ROOT/internvl3_5_8b \
    --retriever-name standard_b256_lr1e-3_norm --phases cattok --emb_types proj_mean --gpu_id 0
ln -s standard_b256_lr1e-3_norm $ATTN_ROOT/internvl3_5_8b/head_rankings/retriever_best
```

To skip steps 1 to 3, copy `assets/head_rankings/<model_dir>/pointer_cattok.csv` to
`$ATTN_ROOT/<model_dir>/head_rankings/attn_score_coco_mcq/cattok.csv` and
`writer_cattok_proj_mean_fullpool.csv` to `$ATTN_ROOT/<model_dir>/head_rankings/retriever_best/cattok_proj_mean_fullpool.csv`.

**4. Single-head retrieval on Visual Haystacks** (Table 2, left):

```bash
CUDA_VISIBLE_DEVICES=0 python data/save_vhs_head_outputs.py --model internvl-8b --hs 10
python training/rank_heads_attn_score.py --model-dir $ATTN_ROOT/internvl3_5_8b --source vhs --hs 10
# L21H10 row of head_rankings/attn_score_vhs/cattok_hs10.csv
```

**5. External retrievers** (Table 2, right):

```bash
python eval/eval_vhs_external_retrievers.py --retriever clip      # 80-template ensemble
python eval/eval_vhs_external_retrievers.py --retriever siglip
python eval/eval_vhs_qwen3vl_embedder.py --model-size 8B --query-mode natural
```

**6. Output-zero ablations on Visual Haystacks** (Sections 6 and Appendix). The paper grid is
haystack sizes 2, 3, 5, 10, 20, 50 and K = 5, 10, 15, 20, 25, 30, with matched-budget random
controls over three seeds.

```bash
CUDA_VISIBLE_DEVICES=0 python eval/eval_vhs_head_ablation.py --model internvl-8b \
    --hs 2,3,5,10,20,50 --top-k 5,10,15,20,25,30 \
    --ranking-source baseline,random,overlap_set_attn,attn_only_set,retr_only_set \
    --random-seeds 42,1,7 --phase cattok --ablation-mode catpos --ablation-pos cattok_onwards \
    --subset VHs_large
```

`overlap_set_attn`, `attn_only_set` and `retr_only_set` are the pointer-and-writer, pointer-only
and writer-only partitions of the top-100 union; `attn_score` and `retriever` give the plain
top-K sweeps. Each cell is one JSONL under `$ATTN_ROOT/<model_dir>/ablation/`; runs resume.

**7. Tables and figures**

```bash
python eval/build_topk_sweep_table.py --out outputs/appendix_topk.tex
python eval/build_setpartition_view.py --out outputs/setpartition.tex --figs-dir outputs/figs
```

## Reproducibility notes

- Everything that does not involve probe training reproduces exactly on our hardware: the ICIR-MCQ
  builder (byte-identical files), head extraction (bit-identical H5), pointer rankings, single-head
  R@1 on Visual Haystacks, baselines and random controls of the ablations.
- Probe training is not bit-deterministic (multi-process GPU training), so a retrained writer ranking
  agrees with the shipped one at Spearman about 0.995 with 93 of 100 top heads shared. Set-partition
  cells that depend on which heads sit at the top-100 boundary (pointer-only at large K in particular)
  can move by a few points between retrainings. To reproduce the paper's ablation cells exactly, use
  the shipped rankings in `assets/head_rankings/` as described in step 3.
- CLIP and SigLIP numbers require the prompt handling in `eval/eval_vhs_external_retrievers.py`
  (80-template mean of normalised embeddings; SigLIP text padded to its 64-token maximum).

## Notes

- Use FlashAttention-2 for anything with long contexts; `ATTN_IMPL=sdpa` works but is slower and
  needs more memory.
- The VHs evaluation parses Yes/No with NLTK tokenisation, not substring matching.
- Everything is resumable: extraction skips saved sample groups, ablations skip finished cells.

## Citation

```bibtex
@inproceedings{park2026heads,
  title     = {Heads That Write, Not Just Point: Image Retrieval Heads in Vision-Language Models},
  author    = {Park, Junsung and Yu, Sangwon and Kang, Donghun and Oh, Yeongtak and Jeong, Jaewon and Yoo, Gyeongtae and Moon, Han Cheol and Lee, Jungbeom and Yoon, Sungroh},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```
