import sys
import time
from pathlib import Path

from scripts import run_local, run_queue as rq

FAKE = """
import sys
from pathlib import Path
args = dict(a.split('=', 1) for a in sys.argv[1:] if '=' in a)
work = Path(args['metrics.results_dir'])
name = args['exp_name'] + '.csv'
rounds = int(args['trainer.global_rounds'])
(work / name).write_text('round,acc,ema_acc,m_norm\\n' + ''.join(f'{r},10,10,\\n' for r in range(rounds)))
(work / 'attack_logs').mkdir(exist_ok=True)
(work / 'attack_logs' / name).write_text('round\\n')
"""


def setup(tmp_path):
    fed = tmp_path / "fedacg"
    fed.mkdir()
    (fed / "federated_train.py").write_text(FAKE)
    spec = tmp_path / "spec.yaml"
    spec.write_text("attack: lf\ndataset: cifar10\nratios: [0.2]\nseeds: [1]\n"
                    "algorithms: {fedavg: 'server=base'}\naggregators: [mean, krum]\n"
                    "aggregator_override: '+server.aggregator={agg}'\ncommon: ''\n")
    return fed, spec, tmp_path / "results", tmp_path / "work"


def args(fed, spec, results, work, *more):
    return ["--fedacg", str(fed), "--spec", str(spec), "--results", str(results), "--work", str(work),
            "--worker", "mac", "--rounds", "3", "--extra", "", *more]


def test_runs_publishes_and_releases(tmp_path):
    fed, spec, results, work = setup(tmp_path)
    results.mkdir()
    assert run_local.main(args(fed, spec, results, work)) == 0
    for n in ["fedavg_mean_lf_20_s1.csv", "fedavg_krum_lf_20_s1.csv"]:
        assert rq.finished(results, n, 3)                      # 공유 폴더로 복사됨
        assert (results / "attack_logs" / n).exists()
        assert rq.read_lock(results, n) is None                # 잠금 해제


def test_skips_run_locked_by_colab(tmp_path):
    fed, spec, results, work = setup(tmp_path)
    rq.acquire(results, "fedavg_mean_lf_20_s1.csv", "예다은")   # 코랩이 돌리는 중
    run_local.main(args(fed, spec, results, work))
    assert not rq.finished(results, "fedavg_mean_lf_20_s1.csv", 3)
    assert rq.finished(results, "fedavg_krum_lf_20_s1.csv", 3)


def test_stop_file_and_max_runs(tmp_path):
    fed, spec, results, work = setup(tmp_path)
    run_local.main(args(fed, spec, results, work, "--max-runs", "1"))
    assert len(list(results.glob("*.csv"))) == 1
    (work / "STOP").touch()
    run_local.main(args(fed, spec, results, work))
    assert len(list(results.glob("*.csv"))) == 1


def test_build_command_replaces_python_and_appends_overrides():
    cmd = run_local.build_command("python federated_train.py seed=1", "/venv/python", 80,
                                  "attack.start_round=30 attack.end_round=50", "./data", "/w", 5)
    assert cmd[0] == "/venv/python" and cmd[1] == "federated_train.py"
    assert "trainer.global_rounds=80" in cmd and "attack.end_round=50" in cmd
    assert "metrics.results_dir=/w" in cmd and "resume.every=5" in cmd


def test_heartbeat_keeps_lock_fresh(tmp_path):
    name = "a.csv"
    rq.acquire(tmp_path, name, "mac")
    lock = rq.lock_path(tmp_path, name)
    before = lock.stat().st_mtime
    with run_local.Heartbeat(tmp_path, name, "mac", minutes=0.002):
        time.sleep(0.5)
    assert lock.stat().st_mtime > before
