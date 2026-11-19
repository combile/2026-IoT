"""여러 사람이 같은 실험 목록을 나눠 돌릴 때 쓰는 실행 대기열.

코랩 노트북 5번 셀이 쓴다.
- 우선순위 정렬 : 핵심 조합(우선 비율, 시드 1)이 먼저 나오게 한다. 나눠 돌려도 핵심 결과부터 모인다
- 몫 나누기     : PART = (내 번호, 전체 인원)이면 정렬된 목록에서 번갈아 가져간다
- 잠금 파일     : 실행을 시작할 때 results/locks/<결과 이름>.lock을 만든다. 다른 사람이 돌리는 중인
                  실행은 건너뛴다. 결과 CSV·체크포인트가 오래 갱신되지 않으면(런타임이 죽음) 잠금을 무시한다
"""
import csv
import json
import socket
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.make_sweep import iter_runs  # noqa: E402

Run = Tuple[str, str]   # (결과 파일 이름, 실행 명령)


# ---------------- 정렬과 나누기 ----------------
def iter_run_specs(spec) -> Iterator[Dict]:
    """iter_runs와 같은 순서로 (name, cmd)에 알고리즘·집계·비율·시드를 붙여 준다."""
    import itertools
    combos = itertools.product(spec["algorithms"], spec["aggregators"], spec["ratios"], spec["seeds"])
    for (alg, agg, ratio, seed), (name, cmd) in zip(combos, iter_runs(spec)):
        yield {"name": name, "cmd": cmd, "algorithm": alg, "aggregator": agg,
               "ratio": float(ratio), "seed": int(seed)}


def prioritized(spec) -> List[Run]:
    """시드 순서가 먼저, 같은 시드 안에서는 우선 비율(spec['priority_ratio'])이 먼저 오게 정렬한다.

    예: 시드 1·20% (10개) → 시드 1·10%, 30% (20개) → 시드 2 ... 알고리즘·집계 순서는 그대로 둔다.
    """
    first = spec.get("priority_ratio")
    seeds = [int(s) for s in spec["seeds"]]
    ratios = [float(r) for r in spec["ratios"]]

    def key(r):
        ratio_rank = 0 if first is not None and abs(r["ratio"] - float(first)) < 1e-9 else 1
        return (seeds.index(r["seed"]), ratio_rank, ratios.index(r["ratio"]))

    specs = list(iter_run_specs(spec))
    order = sorted(range(len(specs)), key=lambda i: (key(specs[i]), i))   # 같은 키는 원래 순서 유지
    return [(specs[i]["name"], specs[i]["cmd"]) for i in order]


def my_share(runs: Sequence[Run], part: Optional[Tuple[int, int]]) -> List[Run]:
    """part = (내 번호, 전체 인원). None이면 전부(도와주기 모드)."""
    if part is None:
        return list(runs)
    idx, total = part
    if total < 1 or not 0 <= idx < total:
        raise ValueError(f"PART는 (0 ~ 전체 인원-1, 전체 인원)이어야 한다: {part}")
    return [r for i, r in enumerate(runs) if i % total == idx]


# ---------------- 진행 상태 ----------------
def finished(results_dir, name: str, rounds: int) -> bool:
    """결과 CSV의 마지막 라운드가 rounds - 1 이상이면 끝난 실행이다."""
    path = Path(results_dir) / name
    if not path.exists():
        return False
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return bool(rows) and int(rows[-1]["round"]) >= rounds - 1


def _last_activity(results_dir, name: str) -> float:
    """잠금 파일, 결과 CSV, 체크포인트 중 가장 최근 수정 시각. 학습 중이면 라운드마다 CSV가 갱신된다."""
    results_dir = Path(results_dir)
    stem = Path(name).stem
    paths = [lock_path(results_dir, name), results_dir / name, results_dir / "checkpoints" / f"{stem}.pt"]
    return max((p.stat().st_mtime for p in paths if p.exists()), default=0.0)


# ---------------- 잠금 ----------------
def lock_path(results_dir, name: str) -> Path:
    return Path(results_dir) / "locks" / f"{Path(name).stem}.lock"


def read_lock(results_dir, name: str) -> Optional[Dict]:
    path = lock_path(results_dir, name)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {"worker": "?"}


def held_by_other(results_dir, name: str, worker: str, stale_minutes: float,
                  now: Optional[float] = None) -> Optional[str]:
    """다른 사람이 돌리는 중이면 그 사람 이름을, 아니면 None을 돌려준다.

    내 잠금(같은 worker)이거나, stale_minutes 동안 아무 파일도 갱신되지 않은 잠금은 무시한다.
    """
    lock = read_lock(results_dir, name)
    if lock is None or lock.get("worker") == worker:
        return None
    now = time.time() if now is None else now
    if now - _last_activity(results_dir, name) > stale_minutes * 60:
        return None
    return str(lock.get("worker"))


def acquire(results_dir, name: str, worker: str) -> Path:
    path = lock_path(results_dir, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"worker": worker, "host": socket.gethostname(), "started": time.time()}))
    return path


def release(results_dir, name: str, worker: str) -> None:
    lock = read_lock(results_dir, name)
    if lock is not None and lock.get("worker") == worker:
        lock_path(results_dir, name).unlink(missing_ok=True)


# ---------------- 대기열 ----------------
def todo(spec, results_dir, rounds: int, part: Optional[Tuple[int, int]]) -> List[Run]:
    """내 몫 중 아직 끝나지 않은 실행 (우선순위 순)."""
    return [r for r in my_share(prioritized(spec), part) if not finished(results_dir, r[0], rounds)]
