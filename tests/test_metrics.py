import csv
import math
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from attacks import parse_attack_config
from metrics import MetricsLogger, momentum_norm, result_csv_name

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------- ‖m‖ ----------

def test_momentum_norm_is_l2_over_all_tensors():
    m = {"a": torch.tensor([3.0, 0.0]), "b": torch.tensor([[4.0]])}
    assert momentum_norm(m) == pytest.approx(5.0)


def test_momentum_norm_skips_integer_buffers():
    m = {"w": torch.tensor([3.0, 4.0]), "num_batches_tracked": torch.tensor(1000)}
    assert momentum_norm(m) == pytest.approx(5.0)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS(Mac GPU) 없음")
def test_momentum_norm_on_mps():
    """MPS는 float64를 지원하지 않는다. Mac GPU로 돌려도 계산돼야 한다."""
    m = {"w": torch.tensor([3.0, 4.0], device="mps")}
    assert momentum_norm(m) == pytest.approx(5.0)


def test_momentum_norm_none_for_fedavg():
    assert momentum_norm(None) is None


# ---------- EMA와 CSV ----------

def test_ema_starts_from_first_accuracy_then_uses_coef(tmp_path):
    logger = MetricsLogger(tmp_path / "r.csv", ema_coef=0.9)
    assert logger.log(0, 10.0, None) == pytest.approx(10.0)
    assert logger.log(1, 20.0, None) == pytest.approx(0.9 * 10.0 + 0.1 * 20.0)


def test_csv_has_one_row_per_logged_round(tmp_path):
    path = tmp_path / "r.csv"
    logger = MetricsLogger(path, ema_coef=0.9)
    logger.log(0, 10.0, 1.5)
    logger.log(1, 20.0, 2.5)
    rows = read_rows(path)
    assert [r["round"] for r in rows] == ["0", "1"]
    assert float(rows[1]["acc"]) == pytest.approx(20.0)
    assert float(rows[1]["ema_acc"]) == pytest.approx(11.0)
    assert float(rows[1]["m_norm"]) == pytest.approx(2.5)


def test_m_norm_column_empty_for_fedavg(tmp_path):
    path = tmp_path / "r.csv"
    MetricsLogger(path, ema_coef=0.9).log(0, 10.0, None)
    assert read_rows(path)[0]["m_norm"] == ""


def test_rows_are_on_disk_before_training_ends(tmp_path):
    """Colab 세션이 끊겨도 그때까지의 라운드는 남아 있어야 한다."""
    path = tmp_path / "r.csv"
    logger = MetricsLogger(path, ema_coef=0.9)
    logger.log(0, 10.0, None)
    assert len(read_rows(path)) == 1


def test_new_run_overwrites_old_file(tmp_path):
    path = tmp_path / "r.csv"
    MetricsLogger(path, ema_coef=0.9).log(0, 10.0, None)
    MetricsLogger(path, ema_coef=0.9)
    assert read_rows(path) == []


def test_creates_results_dir(tmp_path):
    path = tmp_path / "results" / "r.csv"
    MetricsLogger(path, ema_coef=0.9)
    assert path.exists()


@pytest.mark.parametrize("coef", [-0.1, 1.0, 1.5])
def test_rejects_invalid_ema_coef(tmp_path, coef):
    with pytest.raises(ValueError):
        MetricsLogger(tmp_path / "r.csv", ema_coef=coef)


def test_config_file_has_guide_values():
    cfg = yaml.safe_load((ROOT / "configs" / "metrics" / "default.yaml").read_text())
    assert cfg["ema"] == 0.9
    assert cfg["results_dir"]


# ---------- 결과 파일 이름 ----------

def make_args(server, seed=1, aggregator=None):
    server = dict(server)
    if aggregator is not None:
        server["aggregator"] = aggregator
    return OmegaConf.create({"server": server, "seed": seed})


def attack_cfg(name, ratio):
    cfg = yaml.safe_load((ROOT / "configs" / "attack" / f"{name}.yaml").read_text())
    cfg["ratio"] = ratio
    return parse_attack_config(cfg)


def test_result_name_fedacg_krum_lf():
    args = make_args({"type": "ServerM", "momentum": 0.85, "FedACG": True}, seed=1, aggregator="krum")
    assert result_csv_name(args, attack_cfg("lf", 0.2)) == "fedacg_krum_lf_20_s1.csv"


def test_result_name_fedavg_baseline_defaults_to_mean():
    args = make_args({"type": "Server", "momentum": -1}, seed=3)
    assert result_csv_name(args, attack_cfg("none", 0.0)) == "fedavg_mean_none_0_s3.csv"
