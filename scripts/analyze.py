"""결과 폴더의 CSV를 읽어 실행별 요약(summary.csv)을 만든다.

사용 예:
    python scripts/analyze.py /content/drive/MyDrive/2026-IoT/results
    python scripts/analyze.py results/ --out results/summary.csv --target-acc 55

정의와 기본값은 analysis/summary.py, configs/analysis/default.yaml에 있다.
"""
import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis import summarize, write_summary  # noqa: E402


def main():
    cfg = yaml.safe_load((ROOT / "configs" / "analysis" / "default.yaml").read_text())
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir")
    parser.add_argument("--out", help="기본값: <results_dir>/summary.csv")
    parser.add_argument("--target-acc", type=float, default=cfg["target_acc"])
    parser.add_argument("--tolerance", type=float, default=cfg["recovery_tolerance"])
    a = parser.parse_args()

    rows = summarize(a.results_dir, a.target_acc, a.tolerance)
    out = Path(a.out) if a.out else Path(a.results_dir) / "summary.csv"
    write_summary(rows, out)
    print(f"{len(rows)}개 실행 요약 → {out} (목표 정확도 {a.target_acc}%, 회복 허용치 {a.tolerance}%p)")


if __name__ == "__main__":
    main()
