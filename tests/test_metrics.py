import csv
import io
import logging
import math
from pathlib import Path

import pytest
import torch
import yaml
from omegaconf import OmegaConf

from attacks import parse_attack_config
from metrics import MetricsLogger, momentum_norm, result_csv_name, round_summary, setup_run_logging

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


# ---------- 화면/파일 로그 분리 ----------

@pytest.fixture
def clean_root_logger():
    root = logging.getLogger()
    saved_handlers, saved_level = root.handlers[:], root.level
    root.handlers = []
    root.setLevel(logging.INFO)
    yield root
    for h in root.handlers:
        h.close()
    root.handlers = saved_handlers
    root.setLevel(saved_level)


def test_console_shows_only_warnings_file_keeps_everything(tmp_path, clean_root_logger):
    console = io.StringIO()
    clean_root_logger.addHandler(logging.StreamHandler(console))
    path = tmp_path / "logs" / "r.log"
    setup_run_logging(path, console_level="WARNING")

    log = logging.getLogger("clients.base_client")
    log.info("client detail")
    log.warning("round summary")

    assert "client detail" not in console.getvalue()
    assert "round summary" in console.getvalue()
    text = path.read_text()
    assert "client detail" in text and "round summary" in text


def test_removes_shared_hydra_log_file(tmp_path, clean_root_logger):
    """Hydra 기본 federated_train.log는 모든 실행이 한 파일에 섞이므로 뗀다."""
    shared = tmp_path / "federated_train.log"
    clean_root_logger.addHandler(logging.FileHandler(shared))
    setup_run_logging(tmp_path / "r.log", console_level="WARNING")
    logging.getLogger("x").info("only in run log")
    assert "only in run log" not in shared.read_text()


def test_run_log_overwritten_on_rerun(tmp_path, clean_root_logger):
    path = tmp_path / "r.log"
    setup_run_logging(path, console_level="WARNING")
    logging.getLogger("x").info("first run")
    for h in clean_root_logger.handlers:
        h.close()
    clean_root_logger.handlers = []
    setup_run_logging(path, console_level="WARNING")
    assert "first run" not in path.read_text()


def test_round_summary_line():
    assert round_summary(12, 45.1, 43.2, 1.92345) == "[Round 12] acc 45.10% | EMA 43.20% | ‖m‖ 1.9235"
    assert round_summary(3, 10.0, 10.0, None) == "[Round 3] acc 10.00% | EMA 10.00%"


def test_config_has_console_level():
    cfg = yaml.safe_load((ROOT / "configs" / "metrics" / "default.yaml").read_text())
    assert cfg["console_level"] == "WARNING"
