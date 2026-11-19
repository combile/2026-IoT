"""모델 오염: 악성 클라이언트가 보내는 변화량 Δ를 조작한다.

Δ' = s · Δ        (확대)
Δ' = -s · Δ       (부호 반전 + 확대)
Δ' += N(0, σ²)    (선택: 노이즈)

조작한 Δ'로 로컬 가중치도 다시 만든다(w' = w_global + Δ').
FedACG 코드는 서버가 local_weights와 local_deltas를 모두 받기 때문에
둘이 서로 어긋나지 않게 해야 한다.
"""
from typing import Dict, Mapping, Optional, Tuple

import torch

Tensor = torch.Tensor


def is_poisonable(key: str, value: Tensor) -> bool:
    """학습되는 실수 텐서만 조작한다. BN의 num_batches_tracked 같은 정수 버퍼는 제외."""
    return torch.is_floating_point(value) and not key.endswith("num_batches_tracked")


@torch.no_grad()
def poison_delta(delta: Mapping[str, Tensor], scale: float, sign_flip: bool,
                 noise_std: float = 0.0,
                 generator: Optional[torch.Generator] = None) -> Dict[str, Tensor]:
    factor = -scale if sign_flip else scale
    out: Dict[str, Tensor] = {}
    for key, value in delta.items():
        if not is_poisonable(key, value):
            out[key] = value.clone()
            continue
        poisoned = value * factor
        if noise_std > 0:
            noise = torch.randn(value.shape, generator=generator, dtype=value.dtype,
                                device="cpu").to(value.device)
            poisoned = poisoned + noise_std * noise
        out[key] = poisoned
    return out


@torch.no_grad()
def poison_state_dict(local_state: Mapping[str, Tensor], global_state: Mapping[str, Tensor],
                      scale: float, sign_flip: bool, noise_std: float = 0.0,
                      generator: Optional[torch.Generator] = None
                      ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    """(조작된 로컬 가중치, 조작된 Δ)를 돌려준다."""
    delta = {k: local_state[k] - global_state[k].to(local_state[k].device)
             if is_poisonable(k, local_state[k]) else local_state[k]
             for k in local_state}
    poisoned = poison_delta(delta, scale, sign_flip, noise_std, generator)
    new_state: Dict[str, Tensor] = {}
    for k, v in local_state.items():
        if is_poisonable(k, v):
            new_state[k] = global_state[k].to(v.device) + poisoned[k]
        else:
            new_state[k] = v.clone()
    return new_state, poisoned
