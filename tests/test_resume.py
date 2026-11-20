import csv
import random

import numpy as np
import pytest
import torch

from resume import ResumeManager, capture_rng, config_diff, restore_rng, truncate_rows

omegaconf = pytest.importorskip("omegaconf")
OmegaConf = omegaconf.OmegaConf


class FakeServer:
    """FedACG ServerM처럼 global_momentum, global_delta를 가진 서버."""

    def __init__(self, model):
        self.global_momentum = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.global_delta = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}


def make_args(tmp_path, **resume):
    cfg = {"enabled": True, "dir": None, "every": 2, "keep_final": False, "strict": True}
    cfg.update(resume)
    return OmegaConf.create({
        "seed": 1, "trainer": {"global_rounds": 10, "num_clients": 100},
        "server": {"momentum": 0.85, "FedACG": True},
        "metrics": {"results_dir": str(tmp_path / "results"), "console_level": "WARNING"},
        "resume": cfg,
    })


def model_and_server(seed=0):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.GroupNorm(1, 3))
    server = FakeServer(model)
    for v in server.global_momentum.values():
        v.normal_()
    return model, server


def read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


# ---------------- 생성 ----------------
def test_disabled_or_missing_config_returns_none(tmp_path):
    assert ResumeManager.from_args(make_args(tmp_path, enabled=False), "x.csv") is None
    assert ResumeManager.from_args(OmegaConf.create({"seed": 1}), "x.csv") is None


def test_default_path_under_results(tmp_path):
    rm = ResumeManager.from_args(make_args(tmp_path), "fedacg_krum_lf_20_s1.csv")
    assert rm.path == tmp_path / "results" / "checkpoints" / "fedacg_krum_lf_20_s1.pt"
    rm2 = ResumeManager.from_args(make_args(tmp_path, dir=str(tmp_path / "ck")), "a.csv")
    assert rm2.path == tmp_path / "ck" / "a.pt"


# ---------------- 저장 시점 ----------------
def test_should_save_schedule():
    rm = ResumeManager("x.pt", every=3, keep_final=False, strict=True, config={})
    assert [r for r in range(10) if rm.should_save(r, last_round=9)] == [2, 5, 8]
    rm_keep = ResumeManager("x.pt", every=3, keep_final=True, strict=True, config={})
    assert [r for r in range(10) if rm_keep.should_save(r, last_round=9)] == [2, 5, 8, 9]


# ---------------- 저장/복원 ----------------
def test_save_load_restore_roundtrip(tmp_path):
    args = make_args(tmp_path)
    rm = ResumeManager.from_args(args, "run.csv")
    assert rm.load() is None and ResumeManager.start_round(None) == 0

    model, server = model_and_server(0)
    rm.save(5, model, server, ema=42.5)
    assert rm.files() == [rm.round_path(5)]                       # 라운드 번호 파일 하나만

    state = ResumeManager.from_args(args, "run.csv").load()
    assert ResumeManager.start_round(state) == 6 and state["ema"] == 42.5

    model2, server2 = model_and_server(1)                      # 다른 값으로 시작
    ResumeManager.restore(state, model2, server2)
    for k, v in model.state_dict().items():
        assert torch.equal(model2.state_dict()[k], v)
    for k, v in server.global_momentum.items():
        assert torch.equal(server2.global_momentum[k], v)        # m이 0으로 돌아가지 않음


def test_save_is_a_copy(tmp_path):
    rm = ResumeManager.from_args(make_args(tmp_path), "run.csv")
    model, server = model_and_server(0)
    rm.save(1, model, server, ema=None)
    before = {k: v.clone() for k, v in server.global_momentum.items()}
    for v in server.global_momentum.values():
        v.add_(1.0)                                              # 저장 뒤에 바뀌어도
    state = rm.load()
    for k, v in before.items():
        assert torch.equal(state["server"]["global_momentum"][k], v)


def test_fedavg_server_without_momentum(tmp_path):
    rm = ResumeManager.from_args(make_args(tmp_path), "run.csv")
    model, _ = model_and_server(0)
    rm.save(1, model, object(), ema=10.0)
    state = rm.load()
    assert state["server"] == {}
    ResumeManager.restore(state, model, object())


def test_finish_removes_checkpoint_unless_keep_final(tmp_path):
    model, server = model_and_server(0)
    rm = ResumeManager.from_args(make_args(tmp_path), "a.csv")
    rm.save(1, model, server, ema=None)
    rm.finish()
    assert rm.files() == []
    rm_keep = ResumeManager.from_args(make_args(tmp_path, keep_final=True), "b.csv")
    rm_keep.save(1, model, server, ema=None)
    rm_keep.finish()
    assert rm_keep.files() == [rm_keep.round_path(1)]


# ---------------- 설정 비교 ----------------
def test_config_mismatch_strict_raises(tmp_path):
    model, server = model_and_server(0)
    ResumeManager.from_args(make_args(tmp_path), "run.csv").save(1, model, server, ema=None)
    changed = make_args(tmp_path)
    changed.server.momentum = 0.9
    with pytest.raises(RuntimeError, match="server.momentum"):
        ResumeManager.from_args(changed, "run.csv").load()
    changed.resume.strict = False
    assert ResumeManager.from_args(changed, "run.csv").load() is not None


def test_config_ignores_rounds_paths_and_resume(tmp_path):
    model, server = model_and_server(0)
    ResumeManager.from_args(make_args(tmp_path), "run.csv").save(1, model, server, ema=None)
    more = make_args(tmp_path, every=5)
    more.trainer.global_rounds = 2000                          # 라운드를 늘려 이어 돌리기 허용
    more.metrics.console_level = "INFO"
    assert ResumeManager.from_args(more, "run.csv").load() is not None


def test_config_diff_lists_changed_keys():
    diff = config_diff({"a": {"b": 1}, "c": 2}, {"a": {"b": 3}, "c": 2, "d": 4})
    assert diff == ["a.b: 1 -> 3", "d: <없음> -> 4"]


# ---------------- 난수 상태 ----------------
def test_rng_restore_reproduces_sequence():
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    state = capture_rng()
    expected = (random.random(), np.random.choice(100, 10, replace=False).tolist(), torch.rand(3))
    random.seed(9); np.random.seed(9); torch.manual_seed(9)
    restore_rng(state)
    got = (random.random(), np.random.choice(100, 10, replace=False).tolist(), torch.rand(3))
    assert expected[0] == got[0] and expected[1] == got[1] and torch.equal(expected[2], got[2])


# ---------------- CSV 이어 쓰기 ----------------
def test_truncate_rows_keeps_rounds_before_start(tmp_path):
    path = tmp_path / "m.csv"
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["round", "acc"])
        for r in range(6):
            w.writerow([r, r * 10])
    assert truncate_rows(path, 4, ["round", "acc"]) == 4
    assert [r["round"] for r in read_rows(path)] == ["0", "1", "2", "3"]
    assert truncate_rows(tmp_path / "new.csv", 4, ["round", "acc"]) == 0   # 파일이 없으면 헤더만


def test_metrics_logger_resume_continues_ema(tmp_path):
    from metrics import MetricsLogger
    path = tmp_path / "m.csv"
    full = MetricsLogger(path, ema_coef=0.9)
    emas = [full.log(r, acc, 1.0) for r, acc in enumerate([10.0, 20.0, 30.0, 40.0])]
    expected = read_rows(path)

    # 라운드 1까지 체크포인트(EMA = emas[1]), 라운드 2까지 기록된 뒤 끊겼다고 가정
    resumed = MetricsLogger(path, ema_coef=0.9, start_round=2, ema=emas[1])
    assert [r["round"] for r in read_rows(path)] == ["0", "1"]
    for r, acc in [(2, 30.0), (3, 40.0)]:
        resumed.log(r, acc, 1.0)
    assert read_rows(path) == expected


def test_attack_log_resume_keeps_earlier_rounds(tmp_path):
    from attacks import AttackManager
    import yaml
    from pathlib import Path
    cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs/attack/mp.yaml").read_text())
    args = OmegaConf.create({"seed": 1, "trainer": {"num_clients": 100}, "server": {"FedACG": True},
                             "attack": cfg, "metrics": {"results_dir": str(tmp_path)}})
    m = AttackManager.from_args(args, num_classes=10)
    for r in range(4):
        m.begin_round(r, range(10))
    expected = read_rows(m.log_path)
    m2 = AttackManager.from_args(args, num_classes=10, start_round=2)
    assert [r["round"] for r in read_rows(m2.log_path)] == ["0", "1"]
    for r in (2, 3):
        m2.begin_round(r, range(10))
    assert read_rows(m2.log_path) == expected


def test_run_log_append(tmp_path):
    import logging
    from metrics import setup_run_logging
    path = tmp_path / "run.log"
    log = logging.getLogger("resume-test")
    root = logging.getLogger()
    old_level = root.level
    root.setLevel(logging.INFO)
    try:
        setup_run_logging(path, console_level="WARNING")
        log.info("first")
        setup_run_logging(path, console_level="WARNING", append=True)
        log.info("second")
        text = path.read_text()
        assert "first" in text and "second" in text
    finally:
        for h in root.handlers[:]:
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                h.close()
        root.setLevel(old_level)


# ---------------- 구글 드라이브 복사본 ----------------
def test_save_replaces_older_round_files(tmp_path):
    rm = ResumeManager.from_args(make_args(tmp_path), "run.csv")
    model, server = model_and_server(0)
    for r in (9, 19, 29):
        rm.save(r, model, server, ema=None)
    assert rm.files() == [rm.round_path(29)]
    assert rm.round_path(29).name == "run.r00029.pt"


def test_drive_duplicates_are_read_newest_and_cleaned(tmp_path):
    """드라이브가 덮어쓰기 대신 "이름 (n).pt" 복사본을 만들었던 예전 방식의 파일이 남아 있어도
    라운드가 가장 큰 것을 읽고, 다음 저장 때 모두 지운다."""
    args = make_args(tmp_path)
    rm = ResumeManager.from_args(args, "run.csv")
    model, server = model_and_server(0)
    folder = rm.path.parent
    for i, r in enumerate([9, 19, 29, 39]):     # run.pt(라운드 9), run (1).pt(19), ...
        rm.save(r, model, server, ema=float(r))
        name = "run.pt" if i == 0 else f"run ({i}).pt"
        rm.round_path(r).rename(folder / name)
    (folder / "run.pt.tmp").write_bytes(b"partial")
    state = ResumeManager.from_args(args, "run.csv").load()
    assert state["round"] == 39 and state["ema"] == 39.0

    rm.save(49, model, server, ema=None)
    assert rm.files() == [rm.round_path(49)]
    rm.finish()
    assert rm.files() == []


def test_truncated_newest_file_falls_back_to_previous(tmp_path):
    args = make_args(tmp_path)
    rm = ResumeManager.from_args(args, "run.csv")
    model, server = model_and_server(0)
    rm.save(19, model, server, ema=19.0)
    rm.round_path(29).write_bytes(b"cut off while writing")   # 저장 중에 끊김
    rm2 = ResumeManager.from_args(args, "run.csv")
    state = rm2.load()
    assert state["round"] == 19 and rm2.loaded_path == rm.round_path(19)


def test_files_do_not_match_other_runs(tmp_path):
    args = make_args(tmp_path)
    model, server = model_and_server(0)
    a = ResumeManager.from_args(args, "fedacg_mean_lf_20_s1.csv")
    b = ResumeManager.from_args(args, "fedacg_mean_lf_20_s10.csv")
    a.save(9, model, server, ema=None)
    b.save(19, model, server, ema=None)
    assert a.files() == [a.round_path(9)] and b.files() == [b.round_path(19)]
    a.finish()
    assert b.files() == [b.round_path(19)]
