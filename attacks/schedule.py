"""구간 공격: 정해진 라운드 구간에서만 공격한다."""
from typing import Optional


def is_attack_round(rnd: int, start_round: int, end_round: Optional[int]) -> bool:
    """[start_round, end_round) 구간이면 True. end_round가 None이면 끝까지 공격한다.

    구간 밖에서는 악성 클라이언트도 정상 클라이언트와 똑같이 학습하고 보고한다.
    """
    if rnd < start_round:
        return False
    return end_round is None or rnd < end_round
