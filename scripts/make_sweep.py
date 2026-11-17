"""실험 조합 설정(configs/experiments/*.yaml)을 읽어 실행 명령을 한 줄씩 출력한다.

사용 예:
    python scripts/make_sweep.py configs/experiments/lf_sweep.yaml > runs_lf.sh
    python scripts/make_sweep.py configs/experiments/lf_sweep.yaml --check-results results/

--check-results를 주면 명령 대신 아직 결과 CSV가 없는 조합만 출력한다
(가이드 체크리스트: "실험 조합이 빠짐없이 돌았는지", "시드 3회 결과가 모두 있는지").
"""
import argparse
import itertools
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attacks.naming import result_filename  # noqa: E402


def iter_runs(spec):
    for alg, agg, ratio, seed in itertools.product(
            spec["algorithms"], spec["aggregators"], spec["ratios"], spec["seeds"]):
        ratio_pct = int(round(float(ratio) * 100))
        name = result_filename(alg, agg, spec["attack"], ratio_pct, seed)
        overrides = " ".join([
            spec["algorithms"][alg],
            spec["aggregator_override"].format(agg=agg),
            f"dataset={spec['dataset']}",
            f"attack={spec['attack']}",
            f"attack.ratio={ratio}",
            f"seed={seed}",
            f"exp_name={Path(name).stem}",
            " ".join(spec["common"].split()),
        ])
        yield name, f"python federated_train.py {overrides}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("spec")
    parser.add_argument("--check-results", metavar="DIR")
    a = parser.parse_args()
    spec = yaml.safe_load(Path(a.spec).read_text())
    runs = list(iter_runs(spec))
    if a.check_results:
        missing = [n for n, _ in runs if not (Path(a.check_results) / n).exists()]
        print(f"# {len(runs) - len(missing)}/{len(runs)} done, {len(missing)} missing")
        for n in missing:
            print(n)
        sys.exit(1 if missing else 0)
    for _, cmd in runs:
        print(cmd)


if __name__ == "__main__":
    main()
