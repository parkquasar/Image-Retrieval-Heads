"""
Shared model utilities for InternVL and Qwen3-VL experiments.

Provides: constants, MODEL_CONFIGS, image transforms, model loaders,
          get_layers, parse_yes_no, needle position helpers.
"""
import os, sys, re
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# ── Paths ──
# All roots are overridable through environment variables (see README).
HF_CACHE  = os.environ.get('HF_CACHE',  os.path.expanduser('~/.cache/huggingface/hub'))
DATA_ROOT = os.environ.get('DATA_ROOT', os.path.abspath('./data/visual_haystacks'))   # Visual Haystacks root
COCO_ROOT = os.environ.get('COCO_ROOT', os.path.abspath('./data/COCO/images'))       # COCO 2017 images
ATTN_ROOT = os.environ.get('ATTN_ROOT', os.path.abspath('./outputs'))                # extracted H5 / rankings / ablation outputs

# ── Attention kernel ──
# Set ATTN_IMPL=sdpa (or eager) on servers without flash-attn; default FA2.
ATTN_IMPL = os.environ.get('ATTN_IMPL', 'flash_attention_2')

# ── Image constants ──
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# ── Model configs ──
MODEL_CONFIGS = {
    'internvl-8b': {
        'snapshot': 'models--OpenGVLab--InternVL3_5-8B/snapshots/9bb6a56ad9cc69db95e2d4eeb15a52bbcac4ef79',
        'family': 'internvl',
        'model_dir': 'internvl3_5_8b',
        'retriever_ranking': 'retriever_best',
        'num_layers': 36, 'num_heads': 32, 'head_dim': 128,
    },
    'internvl-14b': {
        'snapshot': 'models--OpenGVLab--InternVL3_5-14B/snapshots/a1e37197b393ce9eec9df700fef65c11f4a6ffbd',
        'family': 'internvl',
        'model_dir': 'internvl3_5_14b',
        'retriever_ranking': 'retriever_best',
        'num_layers': 40, 'num_heads': 40, 'head_dim': 128,
    },
    'internvl-4b': {
        'snapshot': 'models--OpenGVLab--InternVL3_5-4B/snapshots/481f6e32467eab4e922ccd7fd6cf420441a62331',
        'family': 'internvl',
        'model_dir': 'internvl3_5_4b',
        'retriever_ranking': 'retriever_best',
        'num_layers': 36, 'num_heads': 32, 'head_dim': 128,
    },
    'qwen3vl-8b': {
        'snapshot': 'models--Qwen--Qwen3-VL-8B-Instruct/snapshots/0c351dd01ed87e9c1b53cbc748cba10e6187ff3b',
        'family': 'qwen3vl',
        'model_dir': 'qwen3_vl_8b',
        'retriever_ranking': 'retriever_best',
        'num_layers': 36, 'num_heads': 32, 'head_dim': 128,
    },
    'qwen3vl-4b': {
        'snapshot': 'models--Qwen--Qwen3-VL-4B-Instruct/snapshots/ebb281ec70b05090aa6165b016eac8ec08e71b17',
        'family': 'qwen3vl',
        'model_dir': 'qwen3_vl_4b',
        'retriever_ranking': 'retriever_best',
        'num_layers': 36, 'num_heads': 32, 'head_dim': 128,
    },
    # 384 정사각 강제 버전 (논문 convention: 모델 native 단일 타일 해상도로 이미지당 토큰 균일화).
    # 기존 llava_ov_7b 트리는 read-only(555)라 새 model_dir 사용.
}


# ── Image loading (InternVL) ──
_internvl_transform = T.Compose([
    T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
    T.Resize((448, 448), interpolation=InterpolationMode.BICUBIC),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# InternVL image preprocessing (ImageNet normalisation, 448px tiles) follows the
# InternVL model card / repository (OpenGVLab, MIT License). See THIRD_PARTY_NOTICES.md.
def load_image_internvl(path):
    return _internvl_transform(Image.open(path)).unsqueeze(0)


# ── Model loading ──
# Multi-GPU model parallel: set env var MULTI_GPU=1 (e.g.,
# `CUDA_VISIBLE_DEVICES=1,3 MULTI_GPU=1 python ...`) to load with
# device_map='auto', which splits layers across all visible GPUs (used for
# hs=50 + 8B where a single 48GB GPU is tight). Default (.cuda()) puts the
# whole model on cuda:0 — keep this for normal single-GPU runs.
def _multi_gpu():
    return os.environ.get('MULTI_GPU', '').lower() in ('1', 'true', 'yes')


def load_internvl(model_path):
    sys.path.insert(0, model_path)

    _orig_linspace = torch.linspace
    def _safe_linspace(*args, **kwargs):
        try:
            result = _orig_linspace(*args, **kwargs)
            if result.is_meta:
                with torch.device('cpu'):
                    return _orig_linspace(*args, **kwargs)
            return result
        except Exception:
            with torch.device('cpu'):
                return _orig_linspace(*args, **kwargs)
    torch.linspace = _safe_linspace

    from transformers import AutoModel, AutoTokenizer
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, '_orig_mark_tied'):
        _orig_mark = getattr(PreTrainedModel, 'mark_tied_weights_as_initialized', None)
        if _orig_mark is not None:
            def _safe_mark(self, *a, **kw):
                if not hasattr(self, 'all_tied_weights_keys'):
                    self.all_tied_weights_keys = {}
                return _orig_mark(self, *a, **kw)
            PreTrainedModel.mark_tied_weights_as_initialized = _safe_mark
            PreTrainedModel._orig_mark_tied = True
    # device_map='auto' (multi-GPU) accesses model.all_tied_weights_keys
    # during infer_auto_device_map BEFORE mark_tied_weights_as_initialized
    # is called, so the _safe_mark patch above is too late. Add a class-level
    # default of {} so the attribute access succeeds for any model that
    # hasn't set it yet (used by InternVL multi-GPU).
    if not hasattr(PreTrainedModel, 'all_tied_weights_keys'):
        PreTrainedModel.all_tied_weights_keys = {}

    if _multi_gpu():
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True, device_map='auto').eval()
    else:
        model = AutoModel.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            trust_remote_code=True).eval().cuda()
    torch.linspace = _orig_linspace
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    return model, tokenizer


def load_qwen3vl(model_path):
    from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
    if _multi_gpu():
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPL,
            device_map='auto',
        ).eval()
    else:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPL,
        ).eval().cuda()
    # MM_NATIVE_RES=1 → official lmms-eval qwen defaults (dynamic smart_resize
    # between 200,704 and 1,605,632 pixels). Default: fixed 448x448 (=256
    # visual tokens per image), the paper convention for multi-image runs.
    if os.environ.get('MM_NATIVE_RES', '0') == '1':
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=256*28*28, max_pixels=1605632)
    else:
        processor = AutoProcessor.from_pretrained(
            model_path, min_pixels=448*448, max_pixels=448*448)
    processor.tokenizer.model_max_length = 256000
    return model, processor


def load_gemma3(model_path):
    """Gemma-3 multimodal (Gemma3ForConditionalGeneration).

    Gemma-3 encodes every image to a FIXED 256 soft tokens (SigLIP 896x896 ->
    4096 patches -> avg-pooled to 256), so per-image visual-token counts are
    already uniform across candidates.  No resize hack is needed here, unlike
    LLaVA-OneVision's anyres.
    """
    from transformers import Gemma3ForConditionalGeneration, AutoProcessor
    if _multi_gpu():
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPL,
            device_map='auto',
        ).eval()
    else:
        model = Gemma3ForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPL,
        ).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor


def load_llava_ov(model_path):
    from transformers import LlavaOnevisionForConditionalGeneration, AutoProcessor
    if _multi_gpu():
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPL,
            device_map='auto',
        ).eval()
    else:
        model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16,
            attn_implementation=ATTN_IMPL,
        ).eval().cuda()
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.model_max_length = 32768
    return model, processor


def get_layers(model, family):
    if family == 'internvl':
        return model.language_model.model.layers
    else:
        return model.model.language_model.layers


# ── Generation configs ──
VHS_GENERATION_CONFIG = dict(max_new_tokens=128, do_sample=False)

# ── Evaluation helpers ──
def parse_yes_no(response):
    import nltk
    tokens = nltk.word_tokenize(response.lower())
    if 'yes' in tokens: return 'yes'
    if 'no' in tokens: return 'no'
    return None


def _find_needle_in_ids(tokenizer_or_proc, ids_list, needle, is_qwen=False):
    """Return (start, end) inclusive token index range of needle in ids_list."""
    tok = tokenizer_or_proc.tokenizer if is_qwen else tokenizer_or_proc
    clean = re.sub(r'^(a|an)\s+', '', needle.strip())
    for variant in [' ' + clean, ' ' + needle.strip(), clean]:
        cat_ids = tok(variant, add_special_tokens=False)['input_ids']
        for i in range(len(ids_list) - len(cat_ids), -1, -1):
            if ids_list[i:i + len(cat_ids)] == cat_ids:
                return (i, i + len(cat_ids) - 1)
    mid = len(ids_list) // 2
    return (mid, mid)


def compute_needle_pos_internvl(model, tokenizer, haystack_rels, question, needle):
    """Return (start, end) inclusive token range of needle in InternVL prompt."""
    from utils.conversation import get_conv_template
    IMG_CTX = '<IMG_CONTEXT>'
    n_tok = model.num_image_token
    n_imgs = len(haystack_rels)
    img_prefix = '\n'.join(['<image>'] * n_imgs) + '\n'
    qs = img_prefix + question
    for _ in haystack_rels:
        qs = qs.replace('<image>', f'<img>{IMG_CTX * n_tok}</img>', 1)
    tmpl = get_conv_template(model.template)
    tmpl.append_message(tmpl.roles[0], qs)
    tmpl.append_message(tmpl.roles[1], None)
    ids_list = tokenizer(tmpl.get_prompt(), return_tensors='pt',
                         max_length=60000, truncation=True)['input_ids'][0].tolist()
    return _find_needle_in_ids(tokenizer, ids_list, needle)


def load_pixel_values_internvl(haystack_rels):
    """Load and stack InternVL pixel values from COCO-relative paths."""
    pixel_values = [load_image_internvl(os.path.join(COCO_ROOT, r)) for r in haystack_rels]
    return torch.cat(pixel_values).to(dtype=torch.bfloat16, device='cuda'), [1] * len(haystack_rels)


@torch.no_grad()
def chat_internvl(model, tokenizer, haystack_rels, question, max_new_tokens=128):
    """Run InternVL inference on a haystack, return response string."""
    pixel_values, num_patches_list = load_pixel_values_internvl(haystack_rels)
    n_imgs = len(haystack_rels)
    return model.chat(
        tokenizer=tokenizer,
        pixel_values=pixel_values,
        question='\n'.join(['<image>'] * n_imgs) + '\n' + question,
        generation_config=dict(do_sample=False, num_beams=1, max_new_tokens=max_new_tokens),
        num_patches_list=num_patches_list,
    )


@torch.no_grad()
def chat_qwen3vl(model, processor, haystack_rels, question, max_new_tokens=128, do_sample=False):
    """Run Qwen3-VL inference on a haystack, return response string."""
    pil_images = [Image.open(os.path.join(COCO_ROOT, r)).convert('RGB') for r in haystack_rels]
    content = [{"type": "image", "image": pil} for pil in pil_images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_images, padding=True,
                       return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample)
    input_len = inputs.input_ids.shape[1]
    return processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


@torch.no_grad()
def chat_llava_ov(model, processor, haystack_rels, question, max_new_tokens=128, do_sample=False):
    """Run LLaVA-OneVision inference on a haystack, return response string."""
    pil_images = [Image.open(os.path.join(COCO_ROOT, r)).convert('RGB') for r in haystack_rels]
    content = [{"type": "image"} for _ in pil_images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_images, return_tensors='pt').to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=do_sample,
                             pad_token_id=processor.tokenizer.eos_token_id)
    input_len = inputs.input_ids.shape[1]
    return processor.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)


def chat(model, tok_or_proc, family, haystack_rels, question, max_new_tokens=128, do_sample=False):
    """Unified inference wrapper. Returns response string."""
    if family == 'internvl':
        return chat_internvl(model, tok_or_proc, haystack_rels, question, max_new_tokens)
    elif family == 'qwen3vl':
        return chat_qwen3vl(model, tok_or_proc, haystack_rels, question, max_new_tokens, do_sample)
    else:
        return chat_llava_ov(model, tok_or_proc, haystack_rels, question, max_new_tokens, do_sample)


def compute_needle_pos_qwen3vl(processor, haystack_rels, question, needle):
    """Return (start, end) inclusive token range of needle in Qwen3-VL prompt."""
    pil_images = [Image.open(os.path.join(COCO_ROOT, r)).convert('RGB')
                  for r in haystack_rels]
    content = [{"type": "image", "image": pil} for pil in pil_images]
    content.append({"type": "text", "text": question})
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False,
                                          add_generation_prompt=True)
    inputs = processor(text=[text], images=pil_images, return_tensors='pt')
    ids_list = inputs['input_ids'][0].tolist()
    return _find_needle_in_ids(processor, ids_list, needle, is_qwen=True)
