"""Krum 조건 위반 비율: 공격 기록(results/attack_logs/)에서 n > 2f + 2가 깨진 라운드의 비율을 구한다.

n은 라운드 참여 인원(n_selected), f는 그 라운드의 실제 악성 수(n_malicious)다.
- 실행별 : 전체 라운드 기준과 공격 구간(attack_active = 1) 기준을 함께 낸다 → krum_violation_runs.csv
- 비율별 : 공격 종류·악성 비율마다 모든 실행의 라운드를 합쳐 낸다 → krum_violation.csv
           이론값(expected_pct)도 함께 낸다. 악성 round(ratio · N)명 중에서 n명을 비복원으로 뽑을 때(초기하분포)
           위반이 나올 확률이다. 악성 지정 방식(fixed / random) 모두 라운드마다 이 분포를 따른다.
무공격 실행은 악성이 없어 제외한다.

사용 예:
    python scripts/krum_violation.py results/
    python scripts/krum_violation.py /content/drive/MyDrive/2026-IoT/results
"""
import argparse
import csv
import re
from collections import Counter
from math import comb
from pathlib import Path
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
NAME_RE = re.compile(r"^[a-z]+_[a-z]+_(?P<attack>[a-z]+)_(?P<ratio>\d+)_s\d+\.csv$")

RUN_FIELDS = ("name", "attack", "ratio", "rounds", "violated", "ratio_pct",
              "attack_rounds", "attack_violated", "attack_ratio_pct")
GROUP_FIELDS = ("attack", "ratio", "runs", "n", "rounds", "violated", "ratio_pct",
                "attack_rounds", "attack_violated", "attack_ratio_pct", "expected_pct")


def violated(n: int, f: int) -> bool:
    return not n > 2 * f + 2


def _pct(part: int, whole: int) -> Optional[float]:
    return 100.0 * part / whole if whole else None


def expected_violation(num_clients: int, n: int, ratio: float) -> float:
    """전체 num_clients명 중 악성 round(ratio · num_clients)명, 라운드당 n명을 비복원으로 뽑을 때 위반 확률."""
    k = int(round(ratio * num_clients))
    total = comb(num_clients, n)
    return sum(comb(k, f) * comb(num_clients - k, n - f)
               for f in range(min(k, n) + 1) if violated(n, f)) / total


def _read(path) -> List[Tuple[int, int, int]]:
    with open(path, newline="") as f:
        return [(int(r["attack_active"]), int(r["n_selected"]), int(r["n_malicious"])) for r in csv.DictReader(f)]


def _counts(rows) -> Dict:
    attack = [r for r in rows if r[0] == 1]
    out = {"rounds": len(rows), "violated": sum(violated(n, f) for _, n, f in rows),
           "attack_rounds": len(attack), "attack_violated": sum(violated(n, f) for _, n, f in attack)}
    out["ratio_pct"] = _pct(out["violated"], out["rounds"])
    out["attack_ratio_pct"] = _pct(out["attack_violated"], out["attack_rounds"])
    return out


def run_violation(path) -> Dict:
    path = Path(path)
    m = NAME_RE.match(path.name)
    row = {"name": path.name, "attack": m["attack"] if m else None, "ratio": int(m["ratio"]) if m else None}
    row.update(_counts(_read(path)))
    return row


def summarize(log_dir, num_clients: int) -> Tuple[List[Dict], List[Dict]]:
    runs, rows_by_group = [], {}
    for path in sorted(Path(log_dir).glob("*.csv")):
        m = NAME_RE.match(path.name)
        if not m or m["attack"] == "none":
            continue
        rows = _read(path)
        run = {"name": path.name, "attack": m["attack"], "ratio": int(m["ratio"])}
        run.update(_counts(rows))
        runs.append(run)
        rows_by_group.setdefault((run["attack"], run["ratio"]), []).append(rows)

    groups = []
    for (attack, ratio), logs in sorted(rows_by_group.items()):
        rows = [r for log in logs for r in log]
        n = Counter(r[1] for r in rows).most_common(1)[0][0]
        g = {"attack": attack, "ratio": ratio, "runs": len(logs), "n": n}
        g.update(_counts(rows))
        g["expected_pct"] = 100 * expected_violation(num_clients, n, ratio / 100)
        groups.append(g)
    return runs, groups


def _write(rows, fields, path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in fields})


def _num_clients_from_sweep() -> int:
    """실험 설정(configs/experiments/lf_sweep.yaml)의 trainer.num_clients를 읽는다."""
    text = (ROOT / "configs" / "experiments" / "lf_sweep.yaml").read_text()
    return int(re.search(r"trainer\.num_clients=(\d+)", text).group(1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", help="attack_logs/가 들어 있는 결과 폴더")
    parser.add_argument("--num-clients", type=int, default=None, help="기본값: 실험 설정의 trainer.num_clients")
    a = parser.parse_args()
    num_clients = a.num_clients or _num_clients_from_sweep()

    results_dir = Path(a.results_dir)
    runs, groups = summarize(results_dir / "attack_logs", num_clients)
    _write(runs, RUN_FIELDS, results_dir / "krum_violation_runs.csv")
    _write(groups, GROUP_FIELDS, results_dir / "krum_violation.csv")
    for g in groups:
        measured = "-" if g["ratio_pct"] is None else f"{g['ratio_pct']:.1f}%"
        print(f"{g['attack']} {g['ratio']}%: n={g['n']}, 실행 {g['runs']}개, 위반 {g['violated']}/{g['rounds']}라운드 "
              f"({measured}), 이론값 {g['expected_pct']:.1f}%")
    print(f"→ {results_dir / 'krum_violation.csv'}, {results_dir / 'krum_violation_runs.csv'}")


if __name__ == "__main__":
    main()
