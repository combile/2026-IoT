import csv
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.krum_violation import expected_violation, run_violation, summarize

ROOT = Path(__file__).resolve().parents[1]


def write_log(path, rows):
    """rows: (attack_active, n_selected, n_malicious)"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "attack_active", "n_selected", "n_malicious", "malicious_ids"])
        for r, (a, n, m) in enumerate(rows):
            w.writerow([r, a, n, m, ""])


# ---------- 실행 하나 ----------

def test_violation_when_n_not_greater_than_2f_plus_2(tmp_path):
    # n = 10: f ≤ 3이면 만족(10 > 8), f = 4면 위반(10 > 10 아님)
    path = tmp_path / "fedacg_krum_lf_20_s1.csv"
    write_log(path, [(0, 10, 3), (1, 10, 4), (1, 10, 2), (0, 10, 5)])
    r = run_violation(path)
    assert (r["rounds"], r["violated"]) == (4, 2)
    assert r["ratio_pct"] == pytest.approx(50.0)
    assert (r["attack_rounds"], r["attack_violated"]) == (2, 1)
    assert r["attack_ratio_pct"] == pytest.approx(50.0)


def test_no_attack_rounds_leaves_attack_ratio_blank(tmp_path):
    path = tmp_path / "fedacg_krum_lf_20_s1.csv"
    write_log(path, [(0, 10, 4), (0, 10, 1)])
    r = run_violation(path)
    assert r["attack_rounds"] == 0 and r["attack_ratio_pct"] is None


# ---------- 이론값 (초기하분포) ----------

def test_expected_violation_matches_brute_force():
    # 전체 6명 중 악성 2명, 라운드당 4명: 4 > 2f + 2 → f ≥ 1이면 위반. P(f ≥ 1) = 1 − C(4,4)/C(6,4) = 14/15
    assert expected_violation(num_clients=6, n=4, ratio=2 / 6) == pytest.approx(14 / 15)


def test_expected_violation_zero_without_malicious():
    assert expected_violation(num_clients=100, n=10, ratio=0.0) == 0.0


def test_expected_violation_plan_setting_values():
    # 100명 중 10명 참여: 악성 20%면 f ≥ 4일 때 위반. 이항 근사 12.1%보다 약간 작다(비복원 추출)
    p20 = expected_violation(num_clients=100, n=10, ratio=0.2)
    p30 = expected_violation(num_clients=100, n=10, ratio=0.3)
    assert 0.10 < p20 < 0.121 and p20 < p30


# ---------- 결과 폴더 전체 ----------

def test_summarize_groups_by_attack_and_ratio(tmp_path):
    logs = tmp_path / "attack_logs"
    write_log(logs / "fedacg_krum_lf_20_s1.csv", [(1, 10, 4), (1, 10, 1)])
    write_log(logs / "fedavg_mean_lf_20_s1.csv", [(1, 10, 4), (1, 10, 4)])
    write_log(logs / "fedacg_krum_mp_30_s1.csv", [(1, 10, 1)])
    write_log(logs / "fedacg_krum_none_0_s1.csv", [(0, 10, 0)])          # 무공격은 제외
    runs, groups = summarize(logs, num_clients=100)
    assert {r["name"] for r in runs} == {"fedacg_krum_lf_20_s1.csv", "fedavg_mean_lf_20_s1.csv",
                                         "fedacg_krum_mp_30_s1.csv"}
    g = {(x["attack"], x["ratio"]): x for x in groups}
    assert (g[("lf", 20)]["rounds"], g[("lf", 20)]["violated"]) == (4, 3)
    assert g[("lf", 20)]["ratio_pct"] == pytest.approx(75.0)
    assert g[("lf", 20)]["expected_pct"] == pytest.approx(100 * expected_violation(100, 10, 0.2))


def test_cli_writes_two_csvs(tmp_path):
    logs = tmp_path / "results" / "attack_logs"
    write_log(logs / "fedacg_krum_lf_20_s1.csv", [(1, 10, 4)])
    subprocess.run([sys.executable, str(ROOT / "scripts" / "krum_violation.py"), str(tmp_path / "results")],
                   check=True, capture_output=True)
    assert (tmp_path / "results" / "krum_violation_runs.csv").exists()
    assert (tmp_path / "results" / "krum_violation.csv").exists()


def test_cli_out_dir_writes_there_not_into_results(tmp_path):
    logs = tmp_path / "results" / "attack_logs"
    write_log(logs / "fedacg_krum_lf_20_s1.csv", [(1, 10, 4)])
    out = tmp_path / "work" / "new"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "krum_violation.py"), str(tmp_path / "results"),
                    "--out-dir", str(out)], check=True, capture_output=True)
    assert (out / "krum_violation_runs.csv").exists() and (out / "krum_violation.csv").exists()
    assert not (tmp_path / "results" / "krum_violation.csv").exists()
