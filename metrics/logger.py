"""결과 CSV 기록: 라운드마다 정확도, EMA 정확도, 서버 모멘텀 크기 ‖m‖.

EMA 계수 같은 실험값은 configs/metrics/default.yaml에서 받는다.
"""
import csv
import math
from pathlib import Path
from typing import Mapping, Optional

import torch

from attacks.naming import result_filename

FIELDS = ("round", "acc", "ema_acc", "m_norm")


def momentum_norm(momentum: Optional[Mapping[str, torch.Tensor]]) -> Optional[float]:
    """서버 모멘텀 m(파라미터 이름 → 텐서)의 전체 L2 노름.

    모멘텀이 없는 FedAvg는 None을 돌려준다. 정수 버퍼(num_batches_tracked 등)는 제외한다.
    텐서별 제곱합은 float32로 구하고(MPS는 float64 미지원), 텐서 사이 합은 파이썬 float로 더한다.
    """
    if momentum is None:
        return None
    total = 0.0
    for value in momentum.values():
        if torch.is_floating_point(value):
            total += float(value.detach().float().pow(2).sum())
    return math.sqrt(total)


class MetricsLogger:
    """평가한 라운드마다 한 줄씩 CSV에 바로 쓴다.

    학습 중간에 끊겨도(Colab 등) 그때까지의 라운드는 파일에 남는다.
    EMA는 첫 정확도에서 시작해 ema ← coef·ema + (1−coef)·acc로 갱신한다.
    """

    def __init__(self, path, ema_coef: float, start_round: int = 0, ema: Optional[float] = None):
        """start_round > 0이면 이어 돌리기다: round < start_round인 줄은 남기고 EMA는 ema에서 잇는다."""
        if not 0.0 <= ema_coef < 1.0:
            raise ValueError(f"metrics.ema는 [0, 1) 범위여야 한다: {ema_coef}")
        self.path = Path(path)
        self.ema_coef = ema_coef
        self.ema: Optional[float] = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if start_round > 0:
            from resume.csv_rows import truncate_rows
            truncate_rows(self.path, start_round, FIELDS)
            self.ema = None if ema is None else float(ema)
        else:
            with open(self.path, "w", newline="") as f:
                csv.writer(f).writerow(FIELDS)

    def log(self, rnd: int, acc: float, m_norm: Optional[float]) -> float:
        acc = float(acc)
        if self.ema is None:
            self.ema = acc
        else:
            self.ema = self.ema_coef * self.ema + (1.0 - self.ema_coef) * acc
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([rnd, acc, self.ema, "" if m_norm is None else m_norm])
        return self.ema


def result_csv_name(args, attack_cfg) -> str:
    """실행 설정에서 결과 파일 이름(알고리즘_집계_공격_비율_시드.csv)을 만든다.

    - 알고리즘: server.FedACG가 켜져 있으면 fedacg, 아니면 fedavg
    - 집계: server.aggregator (없으면 mean). 설정 이름은 우은식 님 집계 구현에 맞춘다.
    """
    algorithm = "fedacg" if args.server.get("FedACG") else "fedavg"
    aggregator = args.server.get("aggregator") or "mean"
    return result_filename(algorithm, aggregator, attack_cfg.tag, attack_cfg.ratio_pct, args.seed)
