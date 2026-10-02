# Third-party code and data

This repository contains small amounts of code adapted from other projects. Their
licenses apply to those parts.

| where | origin | license |
|---|---|---|
| `utils/conversation.py` | InternVL (OpenGVLab), itself derived from FastChat (lm-sys) | MIT (InternVL), Apache-2.0 (FastChat) |
| `utils/model_utils.py`, InternVL image loading and chat helpers | InternVL model card and repository (OpenGVLab) | MIT |
| `eval/eval_vhs_qwen3vl_embedder.py`, `Qwen3VLForEmbedding` wrapper and embedding pooling | Qwen3-VL-Embedding reference implementation (Alibaba Qwen team) | Apache-2.0 |
| `eval/eval_vhs_external_retrievers.py`, the 80 prompt templates | CLIP (OpenAI) | MIT |

Models are downloaded from the HuggingFace Hub under their own licenses:
InternVL3.5 (Apache-2.0), Qwen3-VL and Qwen3-VL-Embedding (Apache-2.0), CLIP (MIT),
SigLIP (Apache-2.0).

Data: ICIR-MCQ is derived from MS-COCO 2017 annotations (CC BY 4.0; images are not
redistributed) and from the interleaved documents of MM-NIAH (MIT), which originate from
OBELICS (CC BY 4.0). Visual Haystacks is used as distributed by its authors and is not
redistributed here.
