"""Resolver for head-ranking artifact paths.

## Why

The ablation and analysis code used to hard-code the pointer/writer ranking CSV paths in several places,
which made it impossible to run the paper code with a different ranking
(for example a ranking recomputed on a different sample set).

This module centralizes that decision.

## Contract

**Without a manifest the resolver behaves exactly like the canonical paths of the paper pipeline.**
Reproducibility is preserved; `selftest()` checks this mechanically.

With a manifest, its paths take precedence. The order is

    manifest > environment variable (WRITER_RANKING_OVERRIDE) > cfg default

## Manifest format (JSON)

    {
      "name": "camera_ready_6000",
      "note": "rankings recomputed on all 6,000 ICIR-MCQ samples",
      "rankings": {
        "internvl3_5_4b": {
          "pointer_dir": "/abs/.../head_rankings/attn_score_coco_mcq",
          "writer": "/abs/.../head_rankings/paper1L"
        }
      }
    }

`pointer_dir` is a directory containing `{phase}.csv`. `writer` has the same meaning as
the former `retriever_ranking` (a directory name or an absolute path) and may contain
`{phase}_{emb}_fullpool.csv` or point at the CSV file itself.

Models absent from the manifest fall back to the canonical paths; partial manifests are allowed.

## Usage

    from utils.ranking_manifest import get_resolver
    r = get_resolver()                      # uses the RANKING_MANIFEST environment variable
    csv_path = r.pointer_csv(model_dir, 'cattok')
    writer   = r.writer_ranking(model_dir, cfg.get('retriever_ranking'))

The resolution is printed once per process so that the run log alone shows which
rankings were used.
"""
from __future__ import annotations

import json
import os
from typing import Dict, Optional

from utils.model_utils import ATTN_ROOT

POINTER_SUBDIR = os.path.join('head_rankings', 'attn_score_coco_mcq')
ENV_MANIFEST = 'RANKING_MANIFEST'
ENV_WRITER_OVERRIDE = 'WRITER_RANKING_OVERRIDE'


class RankingResolver:
    """Resolve ranking paths; canonical paths when no manifest is given."""

    def __init__(self, manifest_path: Optional[str] = None):
        self.manifest_path = manifest_path
        self.name = 'canonical'
        self._rankings: Dict[str, dict] = {}
        self._logged: set = set()

        if manifest_path:
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f'manifest not found: {manifest_path}')
            with open(manifest_path) as f:
                m = json.load(f)
            self.name = m.get('name') or os.path.basename(manifest_path)
            self._rankings = m.get('rankings') or {}
            if not isinstance(self._rankings, dict):
                raise ValueError('manifest.rankings must be an object')
            print(f'[ranking] manifest "{self.name}" applied ({manifest_path})')
            print(f'[ranking]   overridden models: {sorted(self._rankings)}')

    # -- canonical paths (the single definition replacing the hard-coded ones) --
    @staticmethod
    def canonical_pointer_csv(model_dir: str, phase: str) -> str:
        return os.path.join(ATTN_ROOT, model_dir, POINTER_SUBDIR, f'{phase}.csv')

    # -- resolution --
    def pointer_csv(self, model_dir: str, phase: str, *, require: bool = True) -> str:
        entry = self._rankings.get(model_dir) or {}
        d = entry.get('pointer_dir')
        path = (os.path.join(d, f'{phase}.csv') if d
                else self.canonical_pointer_csv(model_dir, phase))
        self._log('pointer', model_dir, phase, path, overridden=bool(d))
        if require and not os.path.exists(path):
            raise FileNotFoundError(f'pointer ranking CSV not found: {path}')
        return path

    def writer_ranking(self, model_dir: str, cfg_default: Optional[str]) -> Optional[str]:
        """Return a value with the same meaning as the former `retriever_ranking`:

        a (relative) directory name or an absolute path, so callers can keep their isabs branch.
        """
        entry = self._rankings.get(model_dir) or {}
        w = entry.get('writer')
        src = 'manifest'
        if not w:
            w, src = os.environ.get(ENV_WRITER_OVERRIDE), 'env'
        if not w:
            w, src = cfg_default, 'cfg'
        self._log('writer', model_dir, '-', str(w), overridden=(src != 'cfg'), src=src)
        return w

    def _log(self, kind, model_dir, phase, path, *, overridden, src=None):
        key = (kind, model_dir, phase)
        if key in self._logged:
            return
        self._logged.add(key)
        tag = f' [{src}]' if src else ''
        mark = 'OVERRIDE' if overridden else 'canonical'
        print(f'[ranking] {kind:7s} {model_dir:16s} {phase:8s} {mark}{tag}: {path}')


_RESOLVER: Optional[RankingResolver] = None


def get_resolver(manifest_path: Optional[str] = None) -> RankingResolver:
    """Process-wide resolver: explicit argument > RANKING_MANIFEST environment variable > canonical."""
    global _RESOLVER
    if manifest_path is None:
        manifest_path = os.environ.get(ENV_MANIFEST) or None
    if _RESOLVER is None or _RESOLVER.manifest_path != manifest_path:
        _RESOLVER = RankingResolver(manifest_path)
    return _RESOLVER


def selftest(model_dirs=('internvl3_5_4b', 'internvl3_5_8b',
                         'qwen3_vl_4b', 'qwen3_vl_8b'),
             phases=('cattok', 'generation')) -> bool:
    """Check mechanically that, without a manifest, the resolver returns the canonical paths."""
    r = RankingResolver(None)
    ok = True
    for md in model_dirs:
        for ph in phases:
            got = r.pointer_csv(md, ph, require=False)
            want = os.path.join(ATTN_ROOT, md, 'head_rankings',
                                'attn_score_coco_mcq', f'{ph}.csv')
            if got != want:
                print(f'  [FAIL] {md}/{ph}\n    got  {got}\n    want {want}')
                ok = False
    print(f'[selftest] canonical paths match: {"OK" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if selftest() else 1)
