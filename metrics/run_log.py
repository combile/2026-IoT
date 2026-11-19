"""화면 출력은 줄이고 전체 로그는 실행마다 파일로 남긴다.

- 화면(콘솔): console_level 이상만 출력한다. 기본 WARNING이면 라운드 요약 한 줄과 경고만 보인다.
- 파일: INFO 이상 전부를 실행별 파일(results/logs/<결과 이름>.log)에 새로 쓴다.
- Hydra 기본 파일 로그(federated_train.log)는 모든 실행이 한 파일에 이어 붙어 섞이므로 뗀다.
"""
import logging
from pathlib import Path
from typing import Optional

FILE_FORMAT = "[%(asctime)s][%(name)s][%(levelname)s] - %(message)s"


def setup_run_logging(path, console_level: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in root.handlers[:]:
        if isinstance(handler, logging.FileHandler):   # FileHandler는 StreamHandler의 하위 클래스라 먼저 본다
            root.removeHandler(handler)
            handler.close()
        elif isinstance(handler, logging.StreamHandler):
            handler.setLevel(console_level)
    file_handler = logging.FileHandler(path, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
    root.addHandler(file_handler)
    return path


def round_summary(rnd: int, acc: float, ema: float, m_norm: Optional[float]) -> str:
    """화면에 찍는 라운드 요약 한 줄. FedAvg는 ‖m‖을 생략한다."""
    line = f"[Round {rnd}] acc {acc:.2f}% | EMA {ema:.2f}%"
    if m_norm is not None:
        line += f" | ‖m‖ {m_norm:.4f}"
    return line
