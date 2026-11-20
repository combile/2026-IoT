"""결과 CSV로 실행별 요약 지표를 계산한다: 최고 정확도, 정확도 하락, 목표 도달 라운드, 회복 라운드.

정의 (3장 "평가 지표"에 옮길 수 있게 적음, 숫자는 configs/analysis/default.yaml):
- 최고 정확도      : 그 실행의 테스트 정확도(acc) 최댓값
- 정확도 하락(%p)  : 같은 조건(알고리즘·집계·시드) 무공격 기준선의 최고 정확도 − 공격 실행의 최고 정확도 (서베이 AIm)
- 목표 도달 라운드 : EMA 정확도가 처음으로 target_acc 이상이 된 라운드
- 공격 구간        : 공격 기록(attack_logs)에서 attack_active = 1인 첫 라운드 ~ 마지막 라운드 + 1, [start, end)
                     기록의 마지막 라운드까지 공격 중이면(학습이 공격 도중 끊김) end를 알 수 없어 빈 칸
- 회복 라운드      : 공격 종료 라운드(end)부터 보아, EMA 정확도가 같은 라운드의 기준선 EMA − recovery_tolerance
                     이상이 된 첫 라운드. 회복에 걸린 라운드 수는 그 라운드 − end
값을 구할 수 없으면(도달 못 함, 회복 못 함, 기준선 없음) 빈 칸으로 둔다.
"""
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

NAME_RE = re.compile(r"^(?P<algorithm>[a-z]+)_(?P<aggregator>[a-z]+)_(?P<attack>[a-z]+)_(?P<ratio>\d+)_s(?P<seed>\d+)\.csv$")

FIELDS = ("name", "algorithm", "aggregator", "attack", "ratio", "seed", "last_round",
          "max_acc", "final_ema_acc", "target_round", "baseline_max_acc", "acc_drop_pp",
          "attack_start", "attack_end", "recovery_round", "recovery_rounds")


def parse_name(name: str) -> Optional[Dict]:
    """결과 파일 이름(알고리즘_집계_공격_비율_시드.csv)을 나눈다. 규칙 밖 이름이면 None."""
    m = NAME_RE.match(name)
    if not m:
        return None
    d = m.groupdict()
    return {"algorithm": d["algorithm"], "aggregator": d["aggregator"], "attack": d["attack"],
            "ratio": int(d["ratio"]), "seed": int(d["seed"])}


def read_rows(path) -> List[Dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _metrics(path) -> List[Dict]:
    return [{"round": int(r["round"]), "acc": float(r["acc"]), "ema_acc": float(r["ema_acc"])}
            for r in read_rows(path)]


def target_round(rows: Sequence[Dict], target: float) -> Optional[int]:
    return next((r["round"] for r in rows if r["ema_acc"] >= target), None)


def attack_window(log_rows: Sequence[Dict]) -> Optional[Tuple[int, Optional[int]]]:
    active = [int(r["round"]) for r in log_rows if int(r["attack_active"]) == 1]
    if not active:
        return None
    last_logged = max(int(r["round"]) for r in log_rows)
    return (min(active), None if max(active) == last_logged else max(active) + 1)


def recovery_round(rows: Sequence[Dict], baseline: Sequence[Dict], attack_end: int,
                   tolerance: float) -> Optional[int]:
    base = {r["round"]: r["ema_acc"] for r in baseline}
    for r in rows:
        if r["round"] >= attack_end and r["round"] in base and r["ema_acc"] >= base[r["round"]] - tolerance:
            return r["round"]
    return None


def summarize(results_dir, target_acc: float, tolerance: float) -> List[Dict]:
    results_dir = Path(results_dir)
    runs = {p.name: parse_name(p.name) for p in sorted(results_dir.glob("*.csv"))}
    runs = {n: info for n, info in runs.items() if info}
    metrics = {n: _metrics(results_dir / n) for n in runs}

    out = []
    for name, info in runs.items():
        rows = metrics[name]
        row = dict.fromkeys(FIELDS)
        row.update(name=name, **info)
        if rows:
            row.update(last_round=rows[-1]["round"], max_acc=max(r["acc"] for r in rows),
                       final_ema_acc=rows[-1]["ema_acc"], target_round=target_round(rows, target_acc))
        if info["attack"] != "none":
            base_name = f"{info['algorithm']}_{info['aggregator']}_none_0_s{info['seed']}.csv"
            baseline = metrics.get(base_name)
            log_path = results_dir / "attack_logs" / name
            window = attack_window(read_rows(log_path)) if log_path.exists() else None
            if window:
                row.update(attack_start=window[0], attack_end=window[1])
            if baseline and rows:
                row["baseline_max_acc"] = max(r["acc"] for r in baseline)
                row["acc_drop_pp"] = row["baseline_max_acc"] - row["max_acc"]
                if window and window[1] is not None:
                    rec = recovery_round(rows, baseline, window[1], tolerance)
                    if rec is not None:
                        row.update(recovery_round=rec, recovery_rounds=rec - window[1])
        out.append(row)
    return out


def write_summary(rows: Sequence[Dict], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: "" if r.get(k) is None else r[k] for k in FIELDS})
