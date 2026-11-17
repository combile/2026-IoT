"""악성 클라이언트 지정.

- fixed : 시드로 한 번 뽑은 ID 집합을 끝까지 쓴다.
- random: 라운드마다 전체 클라이언트에서 같은 수를 다시 뽑는다.

두 방식 모두 전역 np.random 상태를 건드리지 않는 별도 RNG를 쓴다.
그래야 같은 시드에서 무공격 실행과 공격 실행의 참여 클라이언트 표본이 같아진다.
"""
from typing import FrozenSet, Iterable

import numpy as np

# 전역 시드와 섞어 공격용 RNG 스트림을 분리하는 상수 (실험값이 아님)
_STREAM_TAG = 0xA77AC


class MaliciousSelector:
    def __init__(self, num_clients: int, ratio: float, mode: str, seed: int):
        if num_clients <= 0:
            raise ValueError("num_clients는 양수여야 한다")
        self.num_clients = num_clients
        self.mode = mode
        self.seed = int(seed)
        self.num_malicious = int(round(ratio * num_clients))
        self._fixed: FrozenSet[int] = frozenset()
        if mode == "fixed":
            rng = np.random.default_rng([_STREAM_TAG, self.seed])
            self._fixed = self._draw(rng)
        elif mode != "random":
            raise ValueError(f"알 수 없는 selection 방식: {mode}")

    def _draw(self, rng: np.random.Generator) -> FrozenSet[int]:
        if self.num_malicious == 0:
            return frozenset()
        ids = rng.choice(self.num_clients, self.num_malicious, replace=False)
        return frozenset(int(i) for i in ids)

    def malicious_set(self, rnd: int) -> FrozenSet[int]:
        """해당 라운드의 전체 악성 클라이언트 ID 집합."""
        if self.mode == "fixed":
            return self._fixed
        rng = np.random.default_rng([_STREAM_TAG, self.seed, int(rnd)])
        return self._draw(rng)

    def malicious_in(self, rnd: int, selected_ids: Iterable[int]) -> FrozenSet[int]:
        """이번 라운드 참여자 중 악성인 ID (크기가 f_r)."""
        mal = self.malicious_set(rnd)
        return frozenset(int(c) for c in selected_ids if int(c) in mal)
