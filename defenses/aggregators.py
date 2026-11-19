"""강건 집계: Krum, Multi-Krum, Trimmed Mean, Median.

서버는 클라이언트 변화량 Δ_i = w_i − w_recv에 강건 집계를 적용해 D를 구하고,
새 모델과 서버 모멘텀을 D로 갱신한다.

    새 모델   w' = w_recv + D            (FedACG에서 w_recv는 lookahead 모델 w + λm)
    모멘텀    m' = λm + D                 (FedACG / FedAvgM)
    FedAvgM   w' = w + D + λm            (원본 ServerM과 같은 식)

D가 Δ의 평균이면 원본 FedACG 코드의 집계와 같다. 그래서 aggregator=mean은
원본 코드를 그대로 타고, 나머지 네 방식만 이 모듈을 거친다.

악성 수 f는 방어자가 가정하는 값이다: f = floor(assumed_ratio · n).
- Krum         : 점수 = 가장 가까운 n − f − 2개와의 거리 제곱합, 점수가 가장 낮은 1개 선택
- Multi-Krum   : 점수가 낮은 n − f개를 골라 평균
- Trimmed Mean : 좌표마다 위아래 f개씩 버리고 평균 (τ = f / n)
- Median       : 좌표마다 중앙값 (n이 짝수면 가운데 두 값의 평균)
Krum 계열은 n > 2f + 2일 때만 이론적 보장이 있다. 깨지면 경고를 남기고 그대로 진행한다.
"""
import logging
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)
Tensor = torch.Tensor

AGGREGATORS = ("mean", "krum", "multikrum", "trimmedmean", "median")


def normalize_name(name: Optional[str]) -> str:
    """'Multi-Krum', 'trimmed_mean' 같은 표기를 결과 파일 이름 규칙(attacks.naming)과 같게 맞춘다."""
    key = (name or "mean").lower().replace("-", "").replace("_", "")
    if key not in AGGREGATORS:
        raise ValueError(f"server.aggregator는 {AGGREGATORS} 중 하나여야 한다: {name}")
    return key


def aggregator_name(args: Any) -> str:
    return normalize_name(args.server.get("aggregator"))


def assumed_f(assumed_ratio: float, n: int) -> int:
    if not 0.0 <= assumed_ratio < 0.5:
        raise ValueError(f"defense.assumed_ratio는 [0, 0.5) 범위여야 한다: {assumed_ratio}")
    return int(math.floor(assumed_ratio * n + 1e-9))


def krum_condition_ok(n: int, f: int) -> bool:
    return n > 2 * f + 2


def _is_float(t: Tensor) -> bool:
    return torch.is_floating_point(t)


# ---------------- 벡터 단위 (Krum 계열) ----------------

def _flatten_clients(deltas: Mapping[str, Sequence[Tensor]], keys: List[str]) -> Tensor:
    """(n, d) 행렬. i행은 클라이언트 i의 실수 파라미터 Δ를 이어 붙인 벡터."""
    n = len(deltas[keys[0]])
    device = deltas[keys[0]][0].device
    rows = [torch.cat([deltas[k][i].detach().reshape(-1).float().to(device) for k in keys])
            for i in range(n)]
    return torch.stack(rows)


def krum_scores(flat: Tensor, f: int) -> Tensor:
    """클라이언트마다 가장 가까운 n − f − 2개(최소 1개)와의 거리 제곱합."""
    n = flat.shape[0]
    dist = torch.cdist(flat, flat).pow(2)
    dist.fill_diagonal_(float("inf"))
    k = max(n - f - 2, 1)
    return dist.topk(k, dim=1, largest=False).values.sum(dim=1)


def krum_select(flat: Tensor, f: int, multi: bool) -> List[int]:
    scores = krum_scores(flat, f)
    m = max(flat.shape[0] - f, 1) if multi else 1
    return scores.argsort()[:m].tolist()


# ---------------- 좌표 단위 ----------------

def trimmed_mean(stacked: Tensor, trim: int) -> Tensor:
    """좌표마다 위아래 trim개씩 버린 평균. stacked: (n, ...)."""
    n = stacked.shape[0]
    trim = min(trim, (n - 1) // 2)
    if trim == 0:
        return stacked.mean(dim=0)
    s = stacked.sort(dim=0).values
    return s[trim:n - trim].mean(dim=0)


def coordinate_median(stacked: Tensor) -> Tensor:
    """좌표별 중앙값. torch.median은 짝수일 때 아래쪽 값을 주므로 직접 계산한다."""
    n = stacked.shape[0]
    s = stacked.sort(dim=0).values
    if n % 2 == 1:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2


# ---------------- Δ 집계 ----------------

def aggregate_deltas(name: str, deltas: Mapping[str, Sequence[Tensor]], f: int
                     ) -> Tuple[Dict[str, Tensor], List[int]]:
    """(D, 집계에 쓴 클라이언트 위치)를 돌려준다. 위치는 이번 라운드 참여 순서 기준."""
    name = normalize_name(name)
    keys = list(deltas.keys())
    n = len(deltas[keys[0]])
    float_keys = [k for k in keys if _is_float(deltas[k][0])]
    out: Dict[str, Tensor] = {}
    used = list(range(n))

    if name in ("krum", "multikrum"):
        used = krum_select(_flatten_clients(deltas, float_keys), f, multi=(name == "multikrum"))

    for k in keys:
        if k not in float_keys:
            # 정수 버퍼(num_batches_tracked 등)는 원본처럼 평균만 낸다
            out[k] = sum(deltas[k]) / n
            continue
        stacked = torch.stack([d.float() for d in deltas[k]])
        if name in ("mean", "krum", "multikrum"):
            out[k] = stacked[used].mean(dim=0)
        elif name == "trimmedmean":
            out[k] = trimmed_mean(stacked, f)
        else:
            out[k] = coordinate_median(stacked)
        out[k] = out[k].to(deltas[k][0].dtype)
    return out, used


# ---------------- 서버 연결 ----------------

_warned = set()


def robust_aggregate(server: Any, local_weights: Mapping[str, Sequence[Tensor]],
                     local_deltas: Mapping[str, Sequence[Tensor]], client_ids: Sequence[int],
                     model_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    """servers/base.py의 Server.aggregate, ServerM.aggregate가 aggregator != mean일 때 부른다.

    model_dict는 이번 라운드에 클라이언트가 받은 모델(FedACG면 lookahead 모델)이다.
    """
    args = server.args
    name = aggregator_name(args)
    defense = args.get("defense")
    if defense is None:
        raise KeyError("강건 집계에는 defense 설정이 필요하다. 실행 명령에 +defense=default를 붙인다")
    n = len(client_ids)
    f = assumed_f(float(defense.assumed_ratio), n)
    if name in ("krum", "multikrum") and not krum_condition_ok(n, f) and (n, f) not in _warned:
        _warned.add((n, f))
        logger.warning(f"Krum 조건 n > 2f + 2 위반: n={n}, f={f}. 이론적 보장 없이 진행한다")

    D, used = aggregate_deltas(name, local_deltas, f)
    server.last_selected = [int(list(client_ids)[i]) for i in used]
    if name in ("krum", "multikrum"):
        logger.info(f"[Defense] {name} f={f} selected clients: {server.last_selected}")

    new_dict = {k: model_dict[k] + D[k].to(model_dict[k].device) for k in model_dict}

    lam = args.server.momentum
    if lam > 0 and hasattr(server, "global_momentum"):
        if not args.server.get("FedACG"):
            for k in new_dict:
                new_dict[k] = new_dict[k] + lam * server.global_momentum[k]
        for k in local_deltas:
            m = server.global_momentum[k]
            server.global_delta[k] = D[k].to(m.device)
            server.global_momentum[k] = lam * m + server.global_delta[k]
    return new_dict
