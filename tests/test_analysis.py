import csv
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from analysis import (attack_window, parse_name, read_rows, recovery_round, summarize, target_round,
                      write_summary)

ROOT = Path(__file__).resolve().parents[1]


def metrics_rows(emas, accs=None):
    accs = accs or emas
    return [{"round": r, "acc": a, "ema_acc": e} for r, (a, e) in enumerate(zip(accs, emas))]


def write_metrics(path, emas, accs=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "acc", "ema_acc", "m_norm"])
        for row in metrics_rows(emas, accs):
            w.writerow([row["round"], row["acc"], row["ema_acc"], ""])


def write_attack_log(path, active):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "attack_active", "n_selected", "n_malicious", "malicious_ids"])
        for r, a in enumerate(active):
            w.writerow([r, a, 10, 2, "1 2"])


# ---------- 이름 ----------

def test_parse_name():
    assert parse_name("fedacg_krum_lf_20_s1.csv") == {
        "algorithm": "fedacg", "aggregator": "krum", "attack": "lf", "ratio": 20, "seed": 1}


def test_parse_name_rejects_other_files():
    assert parse_name("summary.csv") is None


# ---------- 목표 도달 라운드 ----------

def test_target_round_is_first_round_ema_reaches_target():
    assert target_round(metrics_rows([10, 40, 55, 50, 60]), 50.0) == 2


def test_target_round_none_when_never_reached():
    assert target_round(metrics_rows([10, 20]), 50.0) is None


# ---------- 공격 구간 ----------

def test_attack_window_from_active_flags():
    rows = [{"round": r, "attack_active": a} for r, a in enumerate([0, 0, 1, 1, 1, 0, 0])]
    assert attack_window(rows) == (2, 5)


def test_attack_window_none_without_attack():
    rows = [{"round": r, "attack_active": 0} for r in range(5)]
    assert attack_window(rows) is None


# ---------- 회복 라운드 ----------

def test_recovery_is_first_round_after_attack_within_tolerance():
    base = metrics_rows([50, 50, 50, 50, 50, 50, 50])
    run = metrics_rows([50, 50, 30, 30, 40, 48, 49.5])
    # 공격 구간 [2, 4): 4라운드부터 본다. 기준선 50 - 허용치 1 = 49 이상인 첫 라운드는 6
    assert recovery_round(run, base, attack_end=4, tolerance=1.0) == 6


def test_recovery_can_be_attack_end_itself():
    base = metrics_rows([50, 50, 50, 50])
    run = metrics_rows([50, 40, 49.5, 50])
    assert recovery_round(run, base, attack_end=2, tolerance=1.0) == 2


def test_recovery_none_when_not_recovered():
    base = metrics_rows([50, 50, 50, 50])
    run = metrics_rows([50, 30, 30, 30])
    assert recovery_round(run, base, attack_end=2, tolerance=1.0) is None


def test_recovery_only_compares_rounds_both_runs_have():
    base = metrics_rows([50, 50, 50])            # 기준선이 아직 2라운드까지만 있음
    run = metrics_rows([50, 30, 30, 30, 50])
    assert recovery_round(run, base, attack_end=2, tolerance=1.0) is None


# ---------- 전체 요약 ----------

def make_results(tmp_path):
    res = tmp_path / "results"
    write_metrics(res / "fedacg_krum_none_0_s1.csv", [40, 50, 50, 50, 50, 50], accs=[40, 60, 50, 50, 50, 50])
    write_attack_log(res / "attack_logs" / "fedacg_krum_none_0_s1.csv", [0] * 6)
    write_metrics(res / "fedacg_krum_lf_20_s1.csv", [40, 50, 30, 30, 45, 49.5], accs=[40, 55, 30, 30, 45, 49.5])
    write_attack_log(res / "attack_logs" / "fedacg_krum_lf_20_s1.csv", [0, 0, 1, 1, 0, 0])
    write_metrics(res / "fedavg_mean_lf_20_s1.csv", [10, 20, 30, 40, 50, 60])   # 기준선 없음
    write_attack_log(res / "attack_logs" / "fedavg_mean_lf_20_s1.csv", [0, 0, 1, 1, 0, 0])
    (res / "notes.csv").write_text("x\n1\n")                                  # 규칙 밖 이름은 무시
    return res


def test_summarize_attack_run(tmp_path):
    rows = {r["name"]: r for r in summarize(make_results(tmp_path), target_acc=50.0, tolerance=1.0)}
    r = rows["fedacg_krum_lf_20_s1.csv"]
    assert r["last_round"] == 5
    assert r["max_acc"] == pytest.approx(55.0)
    assert r["baseline_max_acc"] == pytest.approx(60.0)
    assert r["acc_drop_pp"] == pytest.approx(5.0)       # 서베이 AIm: 기준선 최고 정확도 − 공격 시 최고 정확도
    assert r["target_round"] == 1
    assert (r["attack_start"], r["attack_end"]) == (2, 4)
    assert r["recovery_round"] == 5
    assert r["recovery_rounds"] == 1                     # 공격 종료(4) 뒤 1라운드 걸림


def test_summarize_baseline_has_no_attack_columns(tmp_path):
    rows = {r["name"]: r for r in summarize(make_results(tmp_path), target_acc=50.0, tolerance=1.0)}
    r = rows["fedacg_krum_none_0_s1.csv"]
    assert r["attack_start"] is None and r["recovery_rounds"] is None and r["acc_drop_pp"] is None
    assert "notes.csv" not in rows


def test_summarize_missing_baseline_leaves_blank(tmp_path):
    rows = {r["name"]: r for r in summarize(make_results(tmp_path), target_acc=50.0, tolerance=1.0)}
    r = rows["fedavg_mean_lf_20_s1.csv"]
    assert r["baseline_max_acc"] is None and r["recovery_rounds"] is None
    assert (r["attack_start"], r["attack_end"]) == (2, 4)


def test_write_summary_blank_cells_for_none(tmp_path):
    rows = summarize(make_results(tmp_path), target_acc=50.0, tolerance=1.0)
    out = tmp_path / "summary.csv"
    write_summary(rows, out)
    back = {r["name"]: r for r in read_rows(out)}
    assert back["fedavg_mean_lf_20_s1.csv"]["recovery_rounds"] == ""
    assert back["fedacg_krum_lf_20_s1.csv"]["recovery_rounds"] == "1"


def test_cli_writes_summary(tmp_path):
    res = make_results(tmp_path)
    out = tmp_path / "summary.csv"
    subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze.py"), str(res), "--out", str(out)],
                   check=True, capture_output=True)
    assert len(read_rows(out)) == 3


def test_config_has_definitions():
    cfg = yaml.safe_load((ROOT / "configs" / "analysis" / "default.yaml").read_text())
    assert cfg["target_acc"] > 0 and cfg["recovery_tolerance"] >= 0


def test_attack_window_none_end_while_attack_still_running():
    """학습이 공격 구간 도중에 끊기면 공격 종료 라운드를 알 수 없다."""
    rows = [{"round": r, "attack_active": a} for r, a in enumerate([0, 0, 1, 1])]
    assert attack_window(rows) == (2, None)


def test_summarize_no_recovery_while_attack_still_running(tmp_path):
    res = make_results(tmp_path)
    write_attack_log(res / "attack_logs" / "fedacg_krum_lf_20_s1.csv", [0, 0, 1, 1, 1, 1])
    rows = {r["name"]: r for r in summarize(res, target_acc=50.0, tolerance=1.0)}
    r = rows["fedacg_krum_lf_20_s1.csv"]
    assert (r["attack_start"], r["attack_end"]) == (2, None) and r["recovery_rounds"] is None
