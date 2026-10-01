"""Generation-time utilities used across eval scripts.

Currently exports a single class, ``YesNoStop`` -- a stopping criterion
that ends generation once the model emits the first 'yes' or 'no' word.
Used by VHs Yes/No evaluations to cut decode time without affecting
parsed accuracy.
"""
import re

try:
    from transformers import StoppingCriteria, StoppingCriteriaList  # noqa: F401
except ImportError:
    StoppingCriteria = object
    StoppingCriteriaList = list


class YesNoStop(StoppingCriteria):
    """Stop generation as soon as 'yes' or 'no' appears as a word in the
    decoded tail. Cheap: only decodes the last ~8 tokens per check."""

    def __init__(self, tokenizer, tail=8, max_check_tokens=24):
        self.tok = tokenizer
        self.tail = tail
        self.max_check = max_check_tokens
        self._step = 0
        self._initial_len = None
        self._re = re.compile(r'\b(yes|no)\b', re.IGNORECASE)

    def __call__(self, input_ids, scores, **kwargs):
        if self._initial_len is None:
            # On first call, captured input_ids has prompt + 1 new token.
            self._initial_len = input_ids.shape[-1] - 1
        new_count = input_ids.shape[-1] - self._initial_len
        if new_count > self.max_check:
            return False  # give up early-stop attempts; let max_new_tokens cap
        # Decode only the new tokens (small).
        new_ids = input_ids[0, self._initial_len:]
        text = self.tok.decode(new_ids, skip_special_tokens=True)
        return bool(self._re.search(text))
