"""이어 돌릴 때 라운드별 CSV(지표, 공격 기록)를 체크포인트 시점까지 잘라 낸다."""
import csv
from pathlib import Path
from typing import Sequence


def truncate_rows(path, start_round: int, header: Sequence[str]) -> int:
    """round < start_round인 줄만 남기고 다시 쓴다. 남긴 줄 수를 돌려준다.

    체크포인트 뒤에 기록된 라운드는 다시 돌면서 새로 쓰이므로 지운다.
    파일이 없으면 헤더만 있는 새 파일을 만든다.
    """
    path = Path(path)
    rows = []
    if path.exists():
        with open(path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if r.get("round") and int(r["round"]) < start_round]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(header))
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in header} for r in rows)
    return len(rows)
