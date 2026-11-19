"""학습 루프가 부르는 공격 진입점.

학습 루프는 아래 세 곳에서만 공격 모듈을 부른다.

1. 참여 클라이언트를 뽑은 직후 : info = attack.begin_round(rnd, selected_ids)
2. 로컬 데이터셋을 만들 때      : ds = attack.maybe_flip_labels(client_id, rnd, ds)
3. 로컬 결과를 받은 직후       : state = attack.maybe_poison(client_id, rnd, state, global_state)

maybe_* 함수는 라운드 번호를 인자로 받고 내부 상태에 기대지 않는다.
FedACG의 multiprocessing 모드(spawn)에서 자식 프로세스가 가진 복사본으로 불러도
결과가 같다.
"""
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional

import torch
from torch.utils.data import Dataset

from .config import AttackConfig, parse_attack_config
from .label_flip import LabelFlippedDataset, build_label_map
from .model_poison import poison_state_dict
from .schedule import is_attack_round
from .selector import MaliciousSelector


@dataclass(frozen=True)
class RoundAttackInfo:
    round: int
    attack_active: bool           # 이번 라운드가 공격 구간인가
    n_selected: int               # n: 이번 라운드 참여 클라이언트 수
    n_malicious: int              # f_r: 참여자 중 악성 수 (공격 구간 밖이어도 센다)
    malicious_ids: FrozenSet[int]

    def as_log_row(self) -> Dict[str, Any]:
        """결과 CSV에 붙일 열. Krum 조건 위반 비율 계산에 n, f_r을 쓴다."""
        return {
            "round": self.round,
            "attack_active": int(self.attack_active),
            "n_selected": self.n_selected,
            "n_malicious": self.n_malicious,
            "malicious_ids": " ".join(str(i) for i in sorted(self.malicious_ids)),
        }


class AttackManager:
    def __init__(self, cfg: AttackConfig, num_clients: int, num_classes: int, seed: int):
        self.cfg = cfg
        self.num_clients = num_clients
        self.num_classes = num_classes
        self.seed = int(seed)
        self.selector = MaliciousSelector(num_clients, cfg.ratio if cfg.enabled else 0.0,
                                          cfg.selection, self.seed)
        self.label_map = None
        if cfg.type == "lf":
            lf = cfg.label_flip
            self.label_map = build_label_map(num_classes, lf.mode, lf.pairs, lf.target)

    # ---- 생성 ----
    @classmethod
    def from_args(cls, args: Any, num_classes: int) -> "AttackManager":
        """Hydra args에서 만든다. args.attack이 없으면 무공격으로 본다."""
        attack_cfg = args.get("attack") if hasattr(args, "get") else getattr(args, "attack", None)
        if attack_cfg is None:
            attack_cfg = _NO_ATTACK
        return cls(parse_attack_config(attack_cfg),
                   num_clients=int(args.trainer.num_clients),
                   num_classes=int(num_classes),
                   seed=int(args.seed))

    # ---- 판정 ----
    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def is_active_round(self, rnd: int) -> bool:
        return self.enabled and is_attack_round(rnd, self.cfg.start_round, self.cfg.end_round)

    def is_attacking(self, client_id: int, rnd: int) -> bool:
        return self.is_active_round(rnd) and int(client_id) in self.selector.malicious_set(rnd)

    def begin_round(self, rnd: int, selected_ids: Iterable[int]) -> RoundAttackInfo:
        selected = [int(c) for c in selected_ids]
        mal = self.selector.malicious_in(rnd, selected) if self.enabled else frozenset()
        return RoundAttackInfo(
            round=int(rnd),
            attack_active=self.is_active_round(rnd),
            n_selected=len(selected),
            n_malicious=len(mal),
            malicious_ids=mal,
        )

    # ---- 공격 적용 ----
    def maybe_flip_labels(self, client_id: int, rnd: int, dataset: Dataset) -> Dataset:
        if self.cfg.type != "lf" or not self.is_attacking(client_id, rnd):
            return dataset
        return LabelFlippedDataset(dataset, self.label_map)

    def maybe_poison(self, client_id: int, rnd: int,
                     local_state: Mapping[str, torch.Tensor],
                     global_state: Mapping[str, torch.Tensor]) -> Mapping[str, torch.Tensor]:
        """모델 오염 대상이면 조작한 state_dict를, 아니면 받은 것을 그대로 돌려준다.

        호출한 쪽은 돌려받은 state로 Δ = state - global_state를 다시 계산하면 된다.
        """
        if self.cfg.type != "mp" or not self.is_attacking(client_id, rnd):
            return local_state
        mp = self.cfg.model_poison
        gen: Optional[torch.Generator] = None
        if mp.noise_std > 0:
            gen = torch.Generator().manual_seed(_noise_seed(self.seed, rnd, client_id))
        new_state, _ = poison_state_dict(local_state, global_state, mp.scale, mp.sign_flip,
                                         mp.noise_std, gen)
        return new_state


def _noise_seed(seed: int, rnd: int, client_id: int) -> int:
    """(시드, 라운드, 클라이언트)마다 고정된 노이즈 시드. 실행할 때마다 같은 노이즈가 나온다."""
    prime = 1_000_003
    return ((int(seed) * prime + int(rnd)) * prime + int(client_id)) % (2 ** 31)


_NO_ATTACK = {
    "type": "none", "ratio": 0.0, "selection": "fixed",
    "start_round": 0, "end_round": None,
    "label_flip": {"mode": "reverse", "pairs": None, "target": None},
    "model_poison": {"scale": 1.0, "sign_flip": False, "noise_std": 0.0},
}
