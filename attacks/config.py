"""공격 설정.

실험값의 출처는 configs/attack/*.yaml 하나뿐이다. 이 파일은 그 값을 읽어
검증만 하고, 실험용 숫자를 기본값으로 숨겨 두지 않는다.
"""
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Tuple

ATTACK_TYPES = ("none", "lf", "mp")          # 무공격 / 라벨 플리핑 / 모델 오염
SELECTION_MODES = ("fixed", "random")        # 고정 ID / 라운드마다 다시 뽑음
LF_MODES = ("reverse", "pairs", "all_to_one")


@dataclass(frozen=True)
class LabelFlipConfig:
    mode: str
    pairs: Tuple[Tuple[int, int], ...]
    target: Optional[int]


@dataclass(frozen=True)
class ModelPoisonConfig:
    scale: float          # 변화량 확대 배율
    sign_flip: bool       # 부호 반전 여부
    noise_std: float      # 가우시안 노이즈 표준편차 (0이면 끔)


@dataclass(frozen=True)
class AttackConfig:
    type: str
    ratio: float                  # 전체 클라이언트 중 악성 비율
    selection: str
    start_round: int              # 공격 시작 라운드 (포함)
    end_round: Optional[int]      # 공격 종료 라운드 (미포함). None이면 끝까지
    label_flip: LabelFlipConfig
    model_poison: ModelPoisonConfig

    @property
    def enabled(self) -> bool:
        return self.type != "none"

    @property
    def tag(self) -> str:
        """결과 파일 이름에 쓰는 공격 약어 (lf / mp / none)."""
        return self.type

    @property
    def ratio_pct(self) -> int:
        return int(round(self.ratio * 100))


def _get(cfg: Any, key: str):
    if isinstance(cfg, Mapping):
        if key not in cfg:
            raise KeyError(f"attack 설정에 '{key}' 키가 없다")
        return cfg[key]
    if not hasattr(cfg, key):
        raise KeyError(f"attack 설정에 '{key}' 키가 없다")
    return getattr(cfg, key)


def _pairs(raw) -> Tuple[Tuple[int, int], ...]:
    if raw is None:
        return ()
    out: List[Tuple[int, int]] = []
    for p in raw:
        src, dst = list(p)
        out.append((int(src), int(dst)))
    return tuple(out)


def parse_attack_config(cfg: Any) -> AttackConfig:
    """dict 또는 OmegaConf DictConfig(`args.attack`)를 AttackConfig로 바꾼다."""
    atype = str(_get(cfg, "type"))
    if atype not in ATTACK_TYPES:
        raise ValueError(f"attack.type은 {ATTACK_TYPES} 중 하나여야 한다: {atype}")

    ratio = float(_get(cfg, "ratio"))
    if not 0.0 <= ratio < 1.0:
        raise ValueError(f"attack.ratio는 [0, 1) 범위여야 한다: {ratio}")

    selection = str(_get(cfg, "selection"))
    if selection not in SELECTION_MODES:
        raise ValueError(f"attack.selection은 {SELECTION_MODES} 중 하나여야 한다: {selection}")

    start = int(_get(cfg, "start_round"))
    end_raw = _get(cfg, "end_round")
    end = None if end_raw is None else int(end_raw)
    if start < 0 or (end is not None and end <= start):
        raise ValueError(f"공격 구간이 잘못됐다: start={start}, end={end}")

    lf_raw = _get(cfg, "label_flip")
    lf = LabelFlipConfig(
        mode=str(_get(lf_raw, "mode")),
        pairs=_pairs(_get(lf_raw, "pairs")),
        target=None if _get(lf_raw, "target") is None else int(_get(lf_raw, "target")),
    )
    if lf.mode not in LF_MODES:
        raise ValueError(f"label_flip.mode는 {LF_MODES} 중 하나여야 한다: {lf.mode}")

    mp_raw = _get(cfg, "model_poison")
    mp = ModelPoisonConfig(
        scale=float(_get(mp_raw, "scale")),
        sign_flip=bool(_get(mp_raw, "sign_flip")),
        noise_std=float(_get(mp_raw, "noise_std")),
    )
    if mp.scale <= 0 or mp.noise_std < 0:
        raise ValueError(f"model_poison 값이 잘못됐다: {mp}")

    return AttackConfig(
        type=atype, ratio=ratio, selection=selection,
        start_round=start, end_round=end, label_flip=lf, model_poison=mp,
    )
