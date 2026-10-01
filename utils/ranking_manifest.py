"""Head-ranking 산출물 경로 해석기.

## 왜 필요한가

ablation·분석 코드가 pointer/writer 랭킹 CSV 경로를 여러 곳에 하드코딩하고 있었다.
그래서 "논문과 동일한 코드로, 다른 랭킹을 써서" 재현하는 것이 불가능했다
(camera-ready에서 표본 수를 6,000으로 통일하면 랭킹이 바뀐다).

이 모듈은 그 경로 결정을 한 곳으로 모은다.

## 계약

**manifest를 주지 않으면 논문 파이프라인의 canonical 경로와 완전히 동일하게 동작한다.**
기존 재현성은 보존된다. 이는 `selftest()`로 기계적으로 검증한다.

manifest를 주면 그 경로가 우선한다. 우선순위는

    manifest > 환경변수(WRITER_RANKING_OVERRIDE) > cfg 기본값

## manifest 형식 (JSON)

    {
      "name": "camera_ready_6000",
      "note": "ICIR-MCQ 6,000 전량 통일",
      "rankings": {
        "internvl3_5_4b": {
          "pointer_dir": "/abs/.../head_rankings/attn_score_coco_mcq",
          "writer": "/abs/.../head_rankings/paper1L"
        }
      }
    }

`pointer_dir`은 `{phase}.csv`를 담은 디렉토리. `writer`는 기존 `retriever_ranking`과
같은 의미(디렉토리명 또는 절대경로)로, 하위에 `{phase}_{emb}_fullpool.csv`가 있거나
CSV 파일 자체를 가리킬 수 있다.

manifest에 없는 모델은 canonical 경로로 떨어진다. 부분 지정이 가능하다.

## 사용

    from utils.ranking_manifest import get_resolver
    r = get_resolver()                      # RANKING_MANIFEST 환경변수 사용
    csv_path = r.pointer_csv(model_dir, 'cattok')
    writer   = r.writer_ranking(model_dir, cfg.get('retriever_ranking'))

프로세스당 한 번 해석 내역을 stdout에 남긴다. 실행 로그만 보고 어떤 랭킹을
썼는지 사후 확인할 수 있어야 하기 때문이다.
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
    """랭킹 경로 해석. manifest가 없으면 canonical 경로."""

    def __init__(self, manifest_path: Optional[str] = None):
        self.manifest_path = manifest_path
        self.name = 'canonical'
        self._rankings: Dict[str, dict] = {}
        self._logged: set = set()

        if manifest_path:
            if not os.path.exists(manifest_path):
                raise FileNotFoundError(f'manifest 없음: {manifest_path}')
            with open(manifest_path) as f:
                m = json.load(f)
            self.name = m.get('name') or os.path.basename(manifest_path)
            self._rankings = m.get('rankings') or {}
            if not isinstance(self._rankings, dict):
                raise ValueError('manifest.rankings 는 객체여야 한다')
            print(f'[ranking] manifest "{self.name}" 적용 ({manifest_path})')
            print(f'[ranking]   재지정 모델: {sorted(self._rankings)}')

    # ── canonical 경로 (하드코딩을 대체하는 단일 정의) ──────────────────────
    @staticmethod
    def canonical_pointer_csv(model_dir: str, phase: str) -> str:
        return os.path.join(ATTN_ROOT, model_dir, POINTER_SUBDIR, f'{phase}.csv')

    # ── 해석 ────────────────────────────────────────────────────────────────
    def pointer_csv(self, model_dir: str, phase: str, *, require: bool = True) -> str:
        entry = self._rankings.get(model_dir) or {}
        d = entry.get('pointer_dir')
        path = (os.path.join(d, f'{phase}.csv') if d
                else self.canonical_pointer_csv(model_dir, phase))
        self._log('pointer', model_dir, phase, path, overridden=bool(d))
        if require and not os.path.exists(path):
            raise FileNotFoundError(f'pointer 랭킹 CSV 없음: {path}')
        return path

    def writer_ranking(self, model_dir: str, cfg_default: Optional[str]) -> Optional[str]:
        """기존 `retriever_ranking`과 동일한 의미의 값을 돌려준다.

        디렉토리명(상대) 또는 절대경로. 호출 측의 isabs 분기를 그대로 쓸 수 있다.
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
    """프로세스 공용 resolver. 명시 인자 > RANKING_MANIFEST 환경변수 > canonical."""
    global _RESOLVER
    if manifest_path is None:
        manifest_path = os.environ.get(ENV_MANIFEST) or None
    if _RESOLVER is None or _RESOLVER.manifest_path != manifest_path:
        _RESOLVER = RankingResolver(manifest_path)
    return _RESOLVER


def selftest(model_dirs=('internvl3_5_4b', 'internvl3_5_8b',
                         'qwen3_vl_4b', 'qwen3_vl_8b'),
             phases=('cattok', 'generation')) -> bool:
    """manifest 없을 때 canonical 경로와 동일한지 기계적으로 확인한다."""
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
    print(f'[selftest] canonical 경로 일치: {"OK" if ok else "FAIL"}')
    return ok


if __name__ == '__main__':
    import sys
    sys.exit(0 if selftest() else 1)
