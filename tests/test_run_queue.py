import os
import time
from pathlib import Path

import pytest
import yaml

from scripts.run_queue import (acquire, finished, held_by_other, my_share, prioritized, release,
                               todo)

EXP = Path(__file__).resolve().parents[1] / "configs" / "experiments"


def spec(name="lf_sweep"):
    return yaml.safe_load((EXP / f"{name}.yaml").read_text())


def write_csv(path, last_round):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = ["round,acc,ema_acc,m_norm"] + [f"{r},10,10," for r in range(last_round + 1)]
    path.write_text("\n".join(rows) + "\n")


# ---------------- 정렬 ----------------
def test_prioritized_keeps_all_runs_once():
    s = spec()
    runs = prioritized(s)
    assert len(runs) == 90 and len({n for n, _ in runs}) == 90


def test_priority_ratio_seed1_first():
    runs = prioritized(spec())
    first10 = [n for n, _ in runs[:10]]
    assert all(n.endswith("_lf_20_s1.csv") for n in first10)          # 2 알고리즘 × 5 집계
    next20 = [n for n, _ in runs[10:30]]
    assert all("_s1.csv" in n and "_lf_20_" not in n for n in next20)  # 시드 1의 10%, 30%
    assert all(n.endswith("_s3.csv") for n, _ in runs[60:])


def test_baseline_spec():
    runs = prioritized(spec("baseline_sweep"))
    assert len(runs) == 30 and runs[0][0] == "fedavg_mean_none_0_s1.csv"


# ---------------- 나누기 ----------------
@pytest.mark.parametrize("total", [1, 2, 3])
def test_shares_partition_the_list(total):
    runs = prioritized(spec())
    shares = [my_share(runs, (i, total)) for i in range(total)]
    merged = [r for s in shares for r in s]
    assert sorted(merged) == sorted(runs)                             # 빠짐도 겹침도 없음
    assert max(len(s) for s in shares) - min(len(s) for s in shares) <= 1


def test_shares_interleave_priority():
    runs = prioritized(spec())
    a, b = my_share(runs, (0, 2)), my_share(runs, (1, 2))
    assert all(n.endswith("_lf_20_s1.csv") for n, _ in a[:5] + b[:5])  # 둘 다 핵심 조합부터


def test_share_none_is_everything_and_bad_part_rejected():
    runs = prioritized(spec())
    assert my_share(runs, None) == runs
    with pytest.raises(ValueError):
        my_share(runs, (2, 2))


# ---------------- 진행 상태 ----------------
def test_finished_and_todo(tmp_path):
    s = spec()
    runs = prioritized(s)
    write_csv(tmp_path / runs[0][0], last_round=299)     # 끝남
    write_csv(tmp_path / runs[1][0], last_round=120)     # 중간에 끊김
    assert finished(tmp_path, runs[0][0], rounds=300)
    assert not finished(tmp_path, runs[1][0], rounds=300)
    assert not finished(tmp_path, runs[2][0], rounds=300)
    names = [n for n, _ in todo(s, tmp_path, rounds=300, part=None)]
    assert runs[0][0] not in names and names[0] == runs[1][0]     # 끊긴 실행부터 다시
    assert finished(tmp_path, runs[0][0], rounds=200)             # 라운드를 줄여도 끝난 것으로 봄


# ---------------- 잠금 ----------------
def test_lock_by_other_is_respected_then_goes_stale(tmp_path):
    name = "fedacg_krum_lf_20_s1.csv"
    acquire(tmp_path, name, "강윤서")
    assert held_by_other(tmp_path, name, "예다은", stale_minutes=30) == "강윤서"
    assert held_by_other(tmp_path, name, "강윤서", stale_minutes=30) is None    # 내 잠금
    later = time.time() + 31 * 60
    assert held_by_other(tmp_path, name, "예다은", stale_minutes=30, now=later) is None


def test_lock_stays_fresh_while_csv_updates(tmp_path):
    name = "fedacg_krum_lf_20_s1.csv"
    acquire(tmp_path, name, "강윤서")
    old = time.time() - 3600
    os.utime(tmp_path / "locks" / "fedacg_krum_lf_20_s1.lock", (old, old))   # 잠금은 오래됐지만
    write_csv(tmp_path / name, last_round=50)                                 # 결과 CSV가 방금 갱신됨
    assert held_by_other(tmp_path, name, "예다은", stale_minutes=30) == "강윤서"


def test_release_only_own_lock(tmp_path):
    name = "a.csv"
    acquire(tmp_path, name, "우은식")
    release(tmp_path, name, "예다은")
    assert (tmp_path / "locks" / "a.lock").exists()
    release(tmp_path, name, "우은식")
    assert not (tmp_path / "locks" / "a.lock").exists()
