"""
Family-agnostic interface for InternVL, Qwen3-VL, and LLaVA-OneVision.

Wraps family-specific model/processor APIs behind a unified surface so that
consumer scripts (eval_vhs_baseline.py, save_vhs_head_outputs.py,
save_coco_head_outputs_mcq.py, ModelRunner, etc.) stay family-free.

Usage:
    backend = ModelBackend('llava-ov-7b')
    inputs  = backend.build_inputs(pil_images, prompt)
    # inputs['image_ranges'] → list of (start, end) per image
    # inputs['ids_list']     → flat List[int] for token-level lookups
    # hooks attach to backend.layers[li].self_attn.{q,k,v,o}_proj …
    result  = backend.generate(inputs, max_new_tokens=128,
                               capture_first_logits=True)
    # result['response'], result['first_logits']
    embs    = backend.extract_image_embeddings(inputs)
"""
import os
import re
import sys
import types
from typing import List, Optional, Tuple

import torch
from PIL import Image

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_THIS_DIR)
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)

from utils.model_utils import (  # noqa: E402
    HF_CACHE, COCO_ROOT, MODEL_CONFIGS,
    load_internvl, load_qwen3vl, load_llava_ov, load_gemma3,
    load_image_internvl,
)


_VALID_FAMILIES = ('internvl', 'qwen3vl', 'llava_ov', 'gemma3')


class ModelBackend:
    """Family-agnostic wrapper around one multimodal model checkpoint."""

    def __init__(self, model_key: str):
        if model_key not in MODEL_CONFIGS:
            raise ValueError(
                f"Unknown model_key {model_key!r}. Valid: {list(MODEL_CONFIGS.keys())}")
        self.model_key = model_key
        self.cfg = MODEL_CONFIGS[model_key]
        self.family = self.cfg['family']
        if self.family not in _VALID_FAMILIES:
            raise ValueError(f"Unsupported family: {self.family}")

        # os.path.join drops HF_CACHE when snapshot is absolute.
        model_path = os.path.join(HF_CACHE, self.cfg['snapshot'])

        if self.family == 'internvl':
            self.model, self.tokenizer = load_internvl(model_path)
            self.processor = None
        elif self.family == 'qwen3vl':
            self.model, self.processor = load_qwen3vl(model_path)
            self.tokenizer = self.processor.tokenizer
        else:
            self.model, self.processor = load_llava_ov(model_path)
            self.tokenizer = self.processor.tokenizer

        # Fix InternVL extract_feature device drift (same trick as save_vhs_head_outputs.py).
        if self.family == 'internvl':
            _orig_ef = self.model.extract_feature
            _lm = self.model.language_model
            def _fixed_ef(self_, pv):
                return _orig_ef(pv).to(next(_lm.parameters()).device)
            self.model.extract_feature = types.MethodType(_fixed_ef, self.model)

        # Dims and layer access.
        if self.family == 'internvl':
            lm_cfg = self.model.language_model.config
        else:
            lm_cfg = self.model.config.text_config
        self.num_layers = lm_cfg.num_hidden_layers
        self.num_heads = lm_cfg.num_attention_heads
        self.num_kv_heads = getattr(lm_cfg, 'num_key_value_heads', self.num_heads)
        self.head_dim = getattr(lm_cfg, 'head_dim',
                                lm_cfg.hidden_size // self.num_heads)
        self.hidden_size = lm_cfg.hidden_size
        self.kv_groups = self.num_heads // self.num_kv_heads
        self.scale = self.head_dim ** -0.5

        self.layers = self._get_layers()
        self.device = next(self._language_module().parameters()).device

        # Special IDs used by multiple consumers.
        self._init_special_ids()

    # ── Family-specific internals ─────────────────────────────────────────

    def _get_layers(self):
        if self.family == 'internvl':
            return self.model.language_model.model.layers
        elif self.family == 'qwen3vl':
            return self.model.model.language_model.layers
        elif self.family == 'gemma3':
            # transformers 버전에 따라 두 경로 모두 존재 가능 -> 방어적으로 탐색
            for path in (lambda m: m.model.language_model.layers,
                         lambda m: m.language_model.model.layers,
                         lambda m: m.model.language_model.model.layers):
                try:
                    lay = path(self.model)
                    if hasattr(lay, '__len__') and len(lay) > 0:
                        return lay
                except AttributeError:
                    continue
            raise AttributeError('gemma3: language model layers를 찾지 못함')
        else:  # llava_ov
            return self.model.model.language_model.layers

    def _language_module(self):
        if self.family == 'internvl':
            return self.model.language_model
        return self.model.model.language_model

    def _init_special_ids(self):
        """Populate commonly needed special token IDs."""
        self.yes_id = self.tokenizer('Yes', add_special_tokens=False)['input_ids'][0]
        self.no_id = self.tokenizer('No', add_special_tokens=False)['input_ids'][0]

        if self.family == 'internvl':
            self.img_ctx_id = self.tokenizer.convert_tokens_to_ids('<IMG_CONTEXT>')
            self.model.img_context_token_id = self.img_ctx_id
            self.n_image_tokens = self.model.num_image_token
            eos_id = self.tokenizer.convert_tokens_to_ids('<|im_end|>')
            self.eos_id = eos_id or self.tokenizer.eos_token_id
        elif self.family == 'gemma3':
            # Gemma-3는 이미지당 고정 256 soft token. image_token_index가 그 placeholder.
            cfg = self.model.config
            tid = getattr(cfg, 'image_token_index', None)
            if tid is None:
                tid = getattr(cfg, 'image_token_id', None)
            if tid is None:
                tid = self.tokenizer.convert_tokens_to_ids('<image_soft_token>')
            if tid is None or tid < 0:
                raise ValueError('gemma3: image token id를 찾지 못함')
            self.image_token_id = int(tid)
            self.eos_id = self.tokenizer.eos_token_id
        elif self.family == 'qwen3vl':
            self.vs_id = self.tokenizer.convert_tokens_to_ids('<|vision_start|>')
            self.ve_id = self.tokenizer.convert_tokens_to_ids('<|vision_end|>')
            self.eos_id = self.tokenizer.eos_token_id
        else:  # llava_ov
            # Same token id used in both config.image_token_index and tokenizer.
            self.image_token_id = self.model.config.image_token_index
            self.eos_id = self.tokenizer.eos_token_id

    # ── Letter-token helper (MCQ) ─────────────────────────────────────────

    def get_letter_ids(self, letters=('A', 'B', 'C', 'D')) -> dict:
        out = {}
        for L in letters:
            ids = self.tokenizer(L, add_special_tokens=False)['input_ids']
            out[L] = ids[0]
        return out

    # ── Token-level phrase search (family-agnostic) ───────────────────────

    def find_token_pos(self, ids_list: List[int], phrase: str,
                       prefix_variants=None, strip_article: bool = True,
                       fallback='middle') -> int:
        """Find the last token index of `phrase` in `ids_list`.

        Args:
            prefix_variants: explicit list of prefix strings to try (overrides
                strip_article). Must include leading space if desired.
            strip_article: when True and prefix_variants is None, also try
                phrase with leading 'a '/'an ' stripped.
            fallback: one of {'middle', 'tail-5', 'tail-15'} or an int.

        Returns the index of the LAST token of the matched span, or a fallback
        position if not found.
        """
        if prefix_variants is None:
            if strip_article:
                clean = re.sub(r'^(a|an)\s+', '', phrase.strip())
                prefix_variants = [' ' + clean, ' ' + phrase.strip(), clean]
            else:
                prefix_variants = [' ' + phrase, phrase]

        for variant in prefix_variants:
            ids = self.tokenizer(variant, add_special_tokens=False)['input_ids']
            if not ids:
                continue
            for i in range(len(ids_list) - len(ids), -1, -1):
                if ids_list[i:i + len(ids)] == ids:
                    return i + len(ids) - 1

        if isinstance(fallback, int):
            return fallback
        n = len(ids_list)
        if fallback == 'middle':
            return n // 2
        if fallback == 'tail-5':
            return max(0, n - 5)
        if fallback == 'tail-15':
            return max(0, n - 15)
        raise ValueError(f"Unknown fallback: {fallback!r}")

    # ── Image token range detection ───────────────────────────────────────

    def find_image_ranges(self, ids_list: List[int]) -> List[Tuple[int, int]]:
        """Return list of (start, end) inclusive index ranges, one per image.

        Per-family algorithm:
          - internvl  : consecutive runs of <IMG_CONTEXT> token.
          - qwen3vl   : tokens between <|vision_start|> and <|vision_end|>.
          - llava_ov  : consecutive runs of image_token_index.
        """
        if self.family == 'internvl':
            return _find_consecutive_ranges(ids_list, self.img_ctx_id)
        elif self.family == 'qwen3vl':
            return _find_bracketed_ranges(ids_list, self.vs_id, self.ve_id)
        elif self.family == 'gemma3':
            # 이미지당 정확히 256개의 연속 image token -> consecutive run 분할로 충분
            return _find_consecutive_ranges(ids_list, self.image_token_id)
        else:  # llava_ov
            return _find_consecutive_ranges(ids_list, self.image_token_id)

    # ── Input building ────────────────────────────────────────────────────

    def build_inputs(self, pil_images: List[Image.Image], prompt: str,
                     interleave_marker: Optional[str] = None) -> dict:
        """Tokenize and prepare all model inputs from PIL images and a prompt.

        If `interleave_marker` is None, every image is prepended before
        `prompt` (single consecutive block of image placeholders).

        Returns a dict with at least:
            family, input_ids (tensor on device), ids_list (List[int]),
            image_ranges (List[(start, end)]),
            attention_mask (tensor or None),
            _generate_kwargs — dict passed to the family generate path,
            _raw_prompt, _pil_images, _num_patches_list — InternVL chat path.
        """
        pil_images = [img.convert('RGB') if img.mode != 'RGB' else img
                      for img in pil_images]
        if self.family == 'internvl':
            return self._build_inputs_internvl(pil_images, prompt, interleave_marker)
        elif self.family == 'qwen3vl':
            return self._build_inputs_qwen3vl(pil_images, prompt, interleave_marker)
        elif self.family == 'gemma3':
            return self._build_inputs_gemma3(pil_images, prompt, interleave_marker)
        else:
            return self._build_inputs_llava_ov(pil_images, prompt, interleave_marker)

    def _build_inputs_internvl(self, pil_images, prompt, interleave_marker):
        from utils.conversation import get_conv_template
        IMG_CTX = '<IMG_CONTEXT>'
        n_tok = self.n_image_tokens
        n_imgs = len(pil_images)

        # Pixel values first so the tile counts can size the IMG_CTX blocks.
        # MM_NATIVE_RES=1 → official dynamic tiling (single-image benchmarks
        # only; multi-image haystacks would blow up the context). Default off:
        # single 448x448 tile per image, byte-identical to prior runs.
        native = os.environ.get('MM_NATIVE_RES', '0') == '1'
        if native:
            pil_list = [Image.open(_resolve_image_path(img)).convert('RGB')
                        if isinstance(img, str) else img for img in pil_images]
            pv_list = [_internvl_pv_from_pil_native(im) for im in pil_list]
        else:
            pv_list = [load_image_internvl(_resolve_image_path(img)) if isinstance(img, str)
                       else _internvl_pv_from_pil(img) for img in pil_images]
        num_patches_list = [pv.shape[0] for pv in pv_list]

        if interleave_marker is not None:
            if prompt.count(interleave_marker) != n_imgs:
                raise ValueError(
                    f"interleave marker count mismatch: "
                    f"{prompt.count(interleave_marker)} markers vs {n_imgs} images")
            qs = prompt.replace(interleave_marker, '<image>')
        else:
            qs = '\n'.join(['<image>'] * n_imgs) + '\n' + prompt

        for np_i in num_patches_list:
            qs = qs.replace('<image>', f'<img>{IMG_CTX * (n_tok * np_i)}</img>', 1)

        tmpl = get_conv_template(self.model.template)
        tmpl.append_message(tmpl.roles[0], qs)
        tmpl.append_message(tmpl.roles[1], None)
        query = tmpl.get_prompt()

        enc = self.tokenizer(query, return_tensors='pt',
                             max_length=256000, truncation=True)
        input_ids = enc['input_ids'].to(self.device)
        ids_list = input_ids[0].tolist()
        image_ranges = _find_consecutive_ranges(ids_list, self.img_ctx_id)

        pixel_values = torch.cat(pv_list, dim=0).to(
            device=self.device, dtype=torch.bfloat16)

        return {
            'family': 'internvl',
            'input_ids': input_ids,
            'ids_list': ids_list,
            'image_ranges': image_ranges,
            'attention_mask': enc.get('attention_mask',
                                      torch.ones_like(input_ids)).to(self.device),
            'pixel_values': pixel_values,
            'num_patches_list': num_patches_list,
            # For native model.chat(...) fallback
            '_raw_question_with_placeholders': (
                '\n'.join(['<image>'] * n_imgs) + '\n' + prompt
                if interleave_marker is None
                else prompt.replace(interleave_marker, '<image>')),
            '_pil_images': pil_images,
            '_generate_kwargs': {},
        }

    def _build_inputs_qwen3vl(self, pil_images, prompt, interleave_marker):
        content = _build_content_with_images(prompt, pil_images, interleave_marker,
                                             image_entry=lambda pil: {'type': 'image',
                                                                      'image': pil})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        if pil_images:
            proc_inputs = self.processor(text=[text], images=pil_images,
                                         padding=True, return_tensors='pt').to(self.device)
        else:
            proc_inputs = self.processor(text=[text], padding=True,
                                         return_tensors='pt').to(self.device)

        input_ids = proc_inputs['input_ids']
        ids_list = input_ids[0].tolist()
        image_ranges = _find_bracketed_ranges(ids_list, self.vs_id, self.ve_id)
        pixel_values = proc_inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': proc_inputs.get('attention_mask'),
            'pixel_values': pixel_values,
            'image_grid_thw': proc_inputs.get('image_grid_thw'),
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        return {
            'family': 'qwen3vl',
            'input_ids': input_ids,
            'ids_list': ids_list,
            'image_ranges': image_ranges,
            'attention_mask': proc_inputs.get('attention_mask'),
            'pixel_values': pixel_values,
            'image_grid_thw': proc_inputs.get('image_grid_thw'),
            '_pil_images': pil_images,
            '_generate_kwargs': gen_kwargs,
        }

    def _build_inputs_gemma3(self, pil_images, prompt, interleave_marker):
        """Gemma-3: 이미지당 고정 256 soft token이므로 별도 resize 규약이 불필요하다.

        LLaVA-OneVision에서 문제가 됐던 "한 프롬프트 안에서 후보 이미지마다
        visual token 수가 다름" 이슈가 Gemma-3에는 구조적으로 없다.
        """
        content = _build_content_with_images(prompt, pil_images, interleave_marker,
                                             image_entry=lambda pil: {'type': 'image', 'image': pil})
        messages = [{'role': 'user', 'content': content}]
        proc_inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors='pt').to(self.device)

        pixel_values = proc_inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        input_ids = proc_inputs['input_ids']
        ids_list = input_ids[0].tolist()
        image_ranges = self.find_image_ranges(ids_list)

        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': proc_inputs.get('attention_mask'),
            'pixel_values': pixel_values,
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        return {
            'family': 'gemma3',
            'input_ids': input_ids,
            'ids_list': ids_list,
            'image_ranges': image_ranges,
            'attention_mask': proc_inputs.get('attention_mask'),
            'pixel_values': pixel_values,
            '_pil_images': pil_images,
            '_generate_kwargs': gen_kwargs,
        }

    @torch.no_grad()
    def _extract_emb_gemma3(self, inputs):
        """이미지별 pre/post-projector feature 평균.

        vision_tower -> multi_modal_projector 경로를 직접 태운다. Gemma-3는
        이미지당 토큰 수가 고정이라 단순 reshape로 이미지별 분할이 가능하다.
        """
        pv = inputs.get('pixel_values')
        if pv is None:
            return {'patch_mean': None, 'proj_mean': None}
        base = self.model.model if hasattr(self.model, 'model') else self.model
        vt = getattr(base, 'vision_tower', None) or getattr(self.model, 'vision_tower')
        mmp = getattr(base, 'multi_modal_projector', None) or getattr(self.model, 'multi_modal_projector')
        vout = vt(pixel_values=pv)
        pre = vout.last_hidden_state if hasattr(vout, 'last_hidden_state') else vout[0]
        post = mmp(pre)
        n_img = pre.shape[0]
        patch_mean = pre.reshape(n_img, -1, pre.shape[-1]).mean(1).detach().cpu().float()
        proj_mean = post.reshape(n_img, -1, post.shape[-1]).mean(1).detach().cpu().float()
        return {'patch_mean': patch_mean, 'proj_mean': proj_mean}

    def _build_inputs_llava_ov(self, pil_images, prompt, interleave_marker):
        # Paper convention: every image gets the model's NATIVE single-tile resolution so
        # that per-image visual-token counts are equal across candidates (InternVL 448->256,
        # Qwen3-VL forced 448->256).  LLaVA-OV's SigLIP tower is 384-native; leaving the
        # processor at its anyres default gives 1728-3699 tokens depending on aspect ratio,
        # which biases per-image attention mass.  Resizing to 384x384 makes every image 1485.
        if os.environ.get('LLAVA_OV_ANYRES', '0') != '1':
            pil_images = [im.resize((384, 384)) for im in pil_images]
        content = _build_content_with_images(prompt, pil_images, interleave_marker,
                                             image_entry=lambda pil: {'type': 'image'})
        messages = [{'role': 'user', 'content': content}]
        text = self.processor.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)
        if pil_images:
            proc_inputs = self.processor(text=[text], images=pil_images,
                                         return_tensors='pt').to(self.device)
        else:
            proc_inputs = self.processor(text=[text],
                                         return_tensors='pt').to(self.device)

        # LlavaOnevision processor may return float32 pixel_values; cast to bf16.
        pixel_values = proc_inputs.get('pixel_values')
        if pixel_values is not None:
            pixel_values = pixel_values.to(dtype=torch.bfloat16)

        input_ids = proc_inputs['input_ids']
        ids_list = input_ids[0].tolist()

        # Per-image ranges: LLaVA-OV concatenates all image_token positions into
        # one block (no per-image separator). Use pack_image_features to get
        # feature_lens (#tokens per image) and split the block accordingly.
        image_ranges = self._llava_ov_image_ranges(ids_list, pixel_values,
                                                    proc_inputs.get('image_sizes'),
                                                    len(pil_images))

        gen_kwargs = {
            'input_ids': input_ids,
            'attention_mask': proc_inputs.get('attention_mask'),
            'pixel_values': pixel_values,
            'image_sizes': proc_inputs.get('image_sizes'),
            'pad_token_id': self.tokenizer.eos_token_id,
        }
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
        return {
            'family': 'llava_ov',
            'input_ids': input_ids,
            'ids_list': ids_list,
            'image_ranges': image_ranges,
            'attention_mask': proc_inputs.get('attention_mask'),
            'pixel_values': pixel_values,
            'image_sizes': proc_inputs.get('image_sizes'),
            '_pil_images': pil_images,
            '_generate_kwargs': gen_kwargs,
        }

    @torch.no_grad()
    def _llava_ov_vision_forward(self, pixel_values, image_sizes) -> dict:
        """Run SigLIP + projector and return everything needed for embeddings + ranges.

        Returns dict with:
          pre_proj  : List[Tensor(n_tiles_i, seq_len, 1152)]  per image
          post_proj : List[Tensor(n_tiles_i, seq_len, 3584)]  per image
          feature_lens : List[int]  #tokens per image after pack
        """
        from transformers.models.llava_onevision.modeling_llava_onevision import (
            image_size_to_num_patches,
        )
        cfg = self.model.config
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=cfg.image_grid_pinpoints,
                patch_size=cfg.vision_config.image_size,
            )
            for imsize in image_sizes
        ]
        # Flatten pixel_values to (total_tiles, C, H, W).
        if pixel_values.dim() == 5:
            per_img = [pv[:n] for pv, n in zip(pixel_values, image_num_patches)]
            pv_flat = torch.cat(per_img, dim=0)
        else:
            pv_flat = pixel_values

        lm_model = self.model.model  # LlavaOnevisionModel
        vis_out = lm_model.vision_tower(pv_flat, output_hidden_states=True,
                                         return_dict=True)
        vision_feature_layer = cfg.vision_feature_layer
        vision_feature_select_strategy = cfg.vision_feature_select_strategy
        if isinstance(vision_feature_layer, int):
            selected = vis_out.hidden_states[vision_feature_layer]
        else:
            selected = torch.cat(
                [vis_out.hidden_states[i] for i in vision_feature_layer], dim=-1)
        if vision_feature_select_strategy == 'default':
            selected = selected[:, 1:]  # strip CLS
        # selected : (total_tiles, seq_len, 1152)
        post = lm_model.multi_modal_projector(selected)  # (total_tiles, seq_len, 3584)

        pre_list = torch.split(selected, image_num_patches, dim=0)
        post_list = torch.split(post, image_num_patches, dim=0)

        _, feature_lens = lm_model.pack_image_features(
            list(post_list),
            image_sizes,
            image_newline=lm_model.image_newline,
            vision_aspect_ratio=getattr(cfg, 'vision_aspect_ratio', 'anyres_max_9'),
        )
        return {
            'pre_proj': [t.detach().cpu().float() for t in pre_list],
            'post_proj': [t.detach().cpu().float() for t in post_list],
            'feature_lens': [int(x) for x in feature_lens.tolist()],
        }

    def _llava_ov_image_ranges(self, ids_list, pixel_values, image_sizes, n_imgs
                               ) -> List[Tuple[int, int]]:
        """Split LLaVA-OV's single image-token block into per-image ranges."""
        if pixel_values is None or image_sizes is None or n_imgs == 0:
            return _find_consecutive_ranges(ids_list, self.image_token_id)

        gross_block = _find_consecutive_ranges(ids_list, self.image_token_id)
        if not gross_block:
            return []
        block_start = gross_block[0][0]
        block_end = gross_block[-1][1]

        vf = self._llava_ov_vision_forward(pixel_values, image_sizes)
        lens = vf['feature_lens']
        # Cache for later extract_image_embeddings call.
        self._last_vision_forward = vf

        if sum(lens) != (block_end - block_start + 1):
            total = block_end - block_start + 1
            per = total // n_imgs
            return [(block_start + i * per,
                     block_start + (i + 1) * per - 1 if i < n_imgs - 1 else block_end)
                    for i in range(n_imgs)]
        ranges, off = [], block_start
        for L in lens:
            ranges.append((off, off + L - 1))
            off += L
        return ranges

    # ── Image embedding extraction ────────────────────────────────────────

    @torch.no_grad()
    def extract_image_embeddings(self, inputs: dict) -> dict:
        """Return {'patch_mean': (n_imgs, D_vit), 'proj_mean': (n_imgs, D_lm)}.

        Both on CPU, float32. Vision-tower patch features are averaged over
        spatial patches (excluding CLS for InternVL). Projector output is
        averaged over the LM-side tokens per image.
        """
        if self.family == 'internvl':
            return self._extract_emb_internvl(inputs)
        elif self.family == 'gemma3':
            return self._extract_emb_gemma3(inputs)
        elif self.family == 'qwen3vl':
            return self._extract_emb_qwen3vl(inputs)
        else:
            return self._extract_emb_llava_ov(inputs)

    def _extract_emb_internvl(self, inputs):
        pv = inputs['pixel_values']
        n_imgs = pv.shape[0]
        n_tok = self.n_image_tokens
        vis_out = self.model.vision_model(pv)
        patch_mean = vis_out.last_hidden_state[:, 1:, :].float().mean(1).detach().cpu()
        vit_embeds = self.model.extract_feature(pv)
        proj_mean = vit_embeds.reshape(n_imgs, n_tok, -1).float().mean(1).detach().cpu()
        return {'patch_mean': patch_mean, 'proj_mean': proj_mean,
                '_vit_embeds': vit_embeds}

    def _extract_emb_qwen3vl(self, inputs):
        pre_h, post_h = {}, {}
        def pre(_m, args): pre_h['h'] = args[0].detach().cpu().float()
        def post(_m, _a, out): post_h['h'] = out.detach().cpu().float()
        vis = self.model.model.visual
        h1 = vis.merger.register_forward_pre_hook(pre)
        h2 = vis.merger.register_forward_hook(post)
        pv = inputs['pixel_values']
        grid = inputs['image_grid_thw']
        with torch.no_grad():
            vis(pv, grid)
        h1.remove(); h2.remove()

        merge_size = vis.spatial_merge_size
        patch_means, proj_means = [], []
        pre_off = post_off = 0
        for thw in grid:
            t, h, w = int(thw[0]), int(thw[1]), int(thw[2])
            n_pre = t * h * w
            n_post = t * (h // merge_size) * (w // merge_size)
            patch_means.append(pre_h['h'][pre_off:pre_off + n_pre].mean(0))
            proj_means.append(post_h['h'][post_off:post_off + n_post].mean(0))
            pre_off += n_pre; post_off += n_post
        return {'patch_mean': torch.stack(patch_means),
                'proj_mean': torch.stack(proj_means)}

    def _extract_emb_llava_ov(self, inputs):
        """Per-image pre/post-projector feature means.

        Reuses the cached vision forward from build_inputs (if available) to
        avoid running SigLIP twice. Each image's mean is taken over all of its
        assigned tiles × tokens.
        """
        if inputs.get('pixel_values') is None:
            return {'patch_mean': None, 'proj_mean': None}

        vf = getattr(self, '_last_vision_forward', None)
        if vf is None:
            vf = self._llava_ov_vision_forward(
                inputs['pixel_values'], inputs['image_sizes'])
            self._last_vision_forward = vf

        patch_means = torch.stack([
            t.reshape(-1, t.shape[-1]).mean(0) for t in vf['pre_proj']
        ])
        proj_means = torch.stack([
            t.reshape(-1, t.shape[-1]).mean(0) for t in vf['post_proj']
        ])
        return {'patch_mean': patch_means, 'proj_mean': proj_means}

    # ── Generation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, inputs: dict, max_new_tokens: int = 128,
                 capture_first_logits: bool = False,
                 stopping_criteria=None) -> dict:
        """Run generation with hooks already installed on self.layers.

        Returns {'response', 'first_logits' (optional), 'generated_ids'}.

        Family routes:
          - internvl : model.chat(...) with lm_head hook for first-step logits.
          - qwen3vl  : model.generate(**kwargs) with output_scores.
          - llava_ov : model.generate(**kwargs) with output_scores.

        Hooks placed on backend.layers fire naturally during the forward pass.
        Optional `stopping_criteria` (StoppingCriteriaList) is forwarded to
        the underlying generate; useful for early-stop on yes/no detection
        in VHs evaluation.
        """
        if self.family == 'internvl':
            return self._generate_internvl(inputs, max_new_tokens,
                                           capture_first_logits, stopping_criteria)
        return self._generate_hf_native(inputs, max_new_tokens,
                                        capture_first_logits, stopping_criteria)

    def _generate_internvl(self, inputs, max_new_tokens, capture_first_logits,
                            stopping_criteria=None):
        logits_cap = {}
        if capture_first_logits:
            def lm_hook(_m, args, out):
                if 'l' in logits_cap:
                    return
                x = args[0] if args else None
                # Match the first forward pass with seq_len == 1 (first decode step).
                if x is not None and x.shape[1] == 1:
                    logits_cap['l'] = out[0, 0].detach().cpu().float()
            h = self.model.language_model.lm_head.register_forward_hook(lm_hook)
        gen_cfg = dict(do_sample=False, num_beams=1,
                       max_new_tokens=max_new_tokens)
        if stopping_criteria is not None:
            gen_cfg['stopping_criteria'] = stopping_criteria
        try:
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=inputs['pixel_values'],
                question=inputs['_raw_question_with_placeholders'],
                generation_config=gen_cfg,
                num_patches_list=inputs['num_patches_list'],
            )
        finally:
            if capture_first_logits:
                h.remove()
        return {
            'response': response.strip(),
            'first_logits': logits_cap.get('l'),
            'generated_ids': None,
        }

    def _generate_hf_native(self, inputs, max_new_tokens, capture_first_logits,
                             stopping_criteria=None):
        gen_kwargs = dict(inputs['_generate_kwargs'])
        gen_kwargs['max_new_tokens'] = max_new_tokens
        gen_kwargs['do_sample'] = False
        if stopping_criteria is not None:
            gen_kwargs['stopping_criteria'] = stopping_criteria
        if capture_first_logits:
            gen_kwargs['return_dict_in_generate'] = True
            gen_kwargs['output_scores'] = True

        outputs = self.model.generate(**gen_kwargs)

        if capture_first_logits:
            seqs = outputs.sequences
            first_logits = outputs.scores[0][0].detach().cpu().float()
        else:
            seqs = outputs
            first_logits = None

        input_len = inputs['input_ids'].shape[1]
        new_tokens = seqs[0, input_len:]
        response = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        return {
            'response': response,
            'first_logits': first_logits,
            'generated_ids': new_tokens.detach().cpu(),
        }


# ── Module-level helpers ───────────────────────────────────────────────────

def _find_consecutive_ranges(ids_list: List[int], target_id: int
                             ) -> List[Tuple[int, int]]:
    """One range per maximal run of target_id."""
    ranges, i = [], 0
    n = len(ids_list)
    while i < n:
        if ids_list[i] == target_id:
            j = i
            while j + 1 < n and ids_list[j + 1] == target_id:
                j += 1
            ranges.append((i, j))
            i = j + 1
        else:
            i += 1
    return ranges


def _find_bracketed_ranges(ids_list: List[int], open_id: int, close_id: int
                           ) -> List[Tuple[int, int]]:
    """Tokens strictly between each (open_id … close_id) pair (exclusive bounds)."""
    ranges, i = [], 0
    n = len(ids_list)
    while i < n:
        if ids_list[i] == open_id:
            j = i + 1
            while j < n and ids_list[j] != close_id:
                j += 1
            if j < n:
                ranges.append((i + 1, j - 1))
                i = j + 1
                continue
        i += 1
    return ranges


def _internvl_pv_from_pil(img: Image.Image) -> torch.Tensor:
    """Single-image transform → (1, 3, 448, 448)."""
    from utils.model_utils import _internvl_transform
    return _internvl_transform(img).unsqueeze(0)


def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width,
                                        height, image_size):
    """Verbatim from official InternVL demo code (also lmms-eval internvl3)."""
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _internvl_dynamic_preprocess(image: Image.Image, min_num=1, max_num=12,
                                 image_size=448, use_thumbnail=True):
    """Official InternVL dynamic tiling (lmms-eval internvl3 defaults:
    max_num=12, use_thumbnail=True). Only used when MM_NATIVE_RES=1."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1)
        for i in range(1, n + 1) for j in range(1, n + 1)
        if min_num <= i * j <= max_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = _internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))
    return processed_images


def _internvl_pv_from_pil_native(img: Image.Image) -> torch.Tensor:
    """Dynamic-tiled transform → (n_tiles, 3, 448, 448)."""
    from utils.model_utils import _internvl_transform
    tiles = _internvl_dynamic_preprocess(img)
    return torch.stack([_internvl_transform(t) for t in tiles])


def _resolve_image_path(maybe_path: str) -> str:
    """If relative, prepend COCO_ROOT. Used only by InternVL path-taking helpers."""
    if os.path.isabs(maybe_path):
        return maybe_path
    return os.path.join(COCO_ROOT, maybe_path)


def _build_content_with_images(prompt: str, pil_images, interleave_marker,
                               image_entry):
    """Construct a chat-template content list with images at the right spots."""
    if interleave_marker is None or len(pil_images) == 0:
        content = [image_entry(p) for p in pil_images]
        content.append({'type': 'text', 'text': prompt})
        return content
    segments = prompt.split(interleave_marker)
    if len(segments) != len(pil_images) + 1:
        raise ValueError(
            f"interleave marker count mismatch: "
            f"{len(segments) - 1} markers vs {len(pil_images)} images")
    content = []
    for i, seg in enumerate(segments):
        if seg:
            content.append({'type': 'text', 'text': seg})
        if i < len(pil_images):
            content.append(image_entry(pil_images[i]))
    return content
