"""코랩이 아닌 컴퓨터(Mac MPS, 연구실 GPU 등)에서 코랩 노트북 5번 셀과 같은 대기열을 돌린다.

코랩과 동시에 돌려도 된다. 같은 공유 결과 폴더의 잠금 파일(results/locks/)을 보고, 다른 곳에서 돌리는 실행은
건너뛴다. 공유 폴더는 Google Drive 데스크톱 앱으로 연결한 경로를 쓴다.

    python scripts/run_local.py --fedacg ~/fedacg-local \\
        --results "~/Library/CloudStorage/GoogleDrive-<계정>/My Drive/2026-IoT/results" \\
        --work "~/Library/CloudStorage/GoogleDrive-<계정>/My Drive/2026-IoT-work" \\
        --worker 예다은-mac --spec configs/experiments/lf_sweep.yaml

- --work를 코랩의 WORK(내 드라이브 2026-IoT-work)와 같은 폴더로 두면, 코랩에서 끊긴 실행을 이 컴퓨터가 이어 받고
  그 반대도 된다(체크포인트가 같은 곳에 있음).
- 실행이 오래 걸려도(Mac은 80라운드에 약 3시간 40분) 다른 사람이 잠금을 오래된 것으로 보지 않도록 잠금 파일을
  --heartbeat분마다 다시 쓴다.
- --work에 STOP 파일을 만들면 지금 실행을 끝낸 뒤 멈춘다.
- ROUNDS·EXTRA 기본값은 README 1차 실험 계획(80라운드, 공격 30~49)이다. 팀 전체가 같은 값을 쓴다.
"""
import argparse
import shlex
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import yaml  # noqa: E402
from scripts import run_queue as rq  # noqa: E402

DEFAULT_ROUNDS = 80
DEFAULT_EXTRA = "attack.start_round=30 attack.end_round=50"


def publish(work, results, name: str) -> list:
    """work에서 끝난 실행의 결과 CSV·공격 기록·로그를 공유 results로 한 번 복사한다 (노트북 publish와 같음)."""
    stem = Path(name).stem
    copied = []
    for rel in [name, f"attack_logs/{name}", f"logs/{stem}.log"]:
        src = Path(work) / rel
        if src.exists():
            dst = Path(results) / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
            copied.append(rel)
    return copied


def build_command(cmd: str, python: str, rounds: int, extra: str, data: str, work: str, every: int) -> list:
    """sweep의 'python federated_train.py ...' 명령을 이 컴퓨터용으로 바꾼다."""
    args = shlex.split(cmd)
    if args and args[0] == "python":
        args[0] = python
    args += [f"trainer.global_rounds={rounds}", *shlex.split(extra), "wandb=False",
             f"dataset.path={data}", "dataset.client_path=./client_data",
             f"metrics.results_dir={work}", f"resume.every={every}"]
    return args


class Heartbeat:
    """실행 중에 잠금 파일을 주기적으로 다시 써서 '돌고 있음'을 알린다."""

    def __init__(self, results, name, worker, minutes: float):
        self.args = (results, name, worker)
        self.interval = minutes * 60
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.wait(self.interval):
            try:
                rq.acquire(*self.args)
            except OSError:
                pass   # 드라이브가 잠깐 끊겨도 학습은 계속한다

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--fedacg", required=True, help="팀 코드와 패치를 붙인 FedACG 폴더 (scripts/setup_local.sh)")
    p.add_argument("--results", required=True, help="공유 결과 폴더 (드라이브의 2026-IoT/results)")
    p.add_argument("--work", required=True, help="실행 중 결과·체크포인트를 둘 폴더")
    p.add_argument("--worker", required=True, help="잠금 파일에 남는 이름. 코랩과 다르게 쓴다 (예: 예다은-mac)")
    p.add_argument("--spec", default=str(ROOT / "configs/experiments/lf_sweep.yaml"))
    p.add_argument("--data", default="./data", help="CIFAR-10 폴더")
    p.add_argument("--part", default=None, help="'0,2'처럼 (내 번호, 인원). 기본은 전부")
    p.add_argument("--rounds", type=int, default=DEFAULT_ROUNDS)
    p.add_argument("--extra", default=DEFAULT_EXTRA)
    p.add_argument("--stale-minutes", type=float, default=150, help="노트북 STALE_MINUTES와 같은 값")
    p.add_argument("--heartbeat", type=float, default=10, help="잠금 파일을 다시 쓰는 간격(분)")
    p.add_argument("--every", type=int, default=5, help="체크포인트 간격(라운드)")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--max-runs", type=int, default=None, help="이만큼 돌리고 멈춘다")
    a = p.parse_args(argv)

    results, work = Path(a.results).expanduser(), Path(a.work).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    part = tuple(int(x) for x in a.part.split(",")) if a.part else None
    spec = yaml.safe_load(open(a.spec))
    queue = rq.todo(spec, results, a.rounds, part)
    print(f"{a.worker}: {len(queue)}개 남음", flush=True)

    done = 0
    for i, (name, cmd) in enumerate(queue, 1):
        if (work / "STOP").exists():
            print("STOP 파일이 있어 멈춘다", flush=True)
            break
        if a.max_runs is not None and done >= a.max_runs:
            break
        if rq.finished(results, name, a.rounds):
            continue
        other = rq.held_by_other(results, name, a.worker, a.stale_minutes)
        if other:
            print(f"[{i}/{len(queue)}] {name}: {other} 님이 돌리는 중이라 건너뜀", flush=True)
            continue
        if rq.finished(work, name, a.rounds):
            publish(work, results, name)
            print(f"[{i}/{len(queue)}] {name}: 이미 끝나서 공유 폴더로 복사만 함", flush=True)
            continue
        rq.acquire(results, name, a.worker)
        print(f"[{i}/{len(queue)}] {name} 시작 {time.strftime('%m-%d %H:%M')}", flush=True)
        try:
            with Heartbeat(results, name, a.worker, a.heartbeat):
                args = build_command(cmd, a.python, a.rounds, a.extra, a.data, str(work), a.every)
                with open(work / f"{Path(name).stem}.stdout", "a") as out:
                    rc = subprocess.run(args, cwd=Path(a.fedacg).expanduser(), stdout=out,
                                        stderr=subprocess.STDOUT).returncode
            if rc != 0:
                print(f"[{i}/{len(queue)}] {name}: 실패 (종료 코드 {rc}), {work / (Path(name).stem + '.stdout')} 확인",
                      flush=True)
                return rc
            publish(work, results, name)
            done += 1
            print(f"[{i}/{len(queue)}] {name} 끝 {time.strftime('%m-%d %H:%M')}", flush=True)
        finally:
            rq.release(results, name, a.worker)
    return 0


if __name__ == "__main__":
    sys.exit(main())
