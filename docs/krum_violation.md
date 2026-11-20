# Krum 조건 위반 비율 (scripts/krum_violation.py)

담당: 강윤서 · 가이드 "강윤서 가이드 > 실험" 2번(Krum 조건 위반 비율)에 해당한다.

## 실행

```bash
python scripts/krum_violation.py results/
```
코랩에서는 `!python /content/2026-IoT/scripts/krum_violation.py /content/drive/MyDrive/2026-IoT/results --out-dir /content/drive/MyDrive/2026-IoT-work`.
공유 폴더를 바로가기로 쓰는 경우 결과 CSV를 공유 폴더에 다시 쓰면 같은 이름 파일이 새로 생기므로, `--out-dir`로 내 드라이브 폴더에 쓴다. 생략하면 `results_dir`에 쓴다.
공격 기록(`results/attack_logs/`)만 읽으므로 실험 도중에도 돌려 볼 수 있다.

## 정의 (3장에 옮길 수 있게 적음)

- 라운드 r의 참여 인원 n(`n_selected`)과 그 라운드의 실제 악성 수 f_r(`n_malicious`)이 n > 2f_r + 2를 만족하지 않으면 위반이다.
- **실측 위반 비율**: 위반 라운드 수 / 전체 라운드 수. 공격 구간(`attack_active = 1`)만 센 값도 함께 낸다(구간 밖 악성 클라이언트는 정상 행동하므로 Krum에 영향이 없다).
- **이론값**: 전체 N명 중 악성 round(ratio · N)명에서 n명을 비복원으로 뽑을 때(초기하분포) 위반이 나올 확률. N은 실험 설정의 `trainer.num_clients`.
- 무공격 실행은 악성이 없어 제외한다.
- 방어 쪽 Krum은 실제 f_r이 아니라 가정한 f = floor(assumed_ratio · n)을 쓴다(`configs/defense/default.yaml`). 이 비율은 "실제 악성 수가 Krum의 이론적 보장 범위를 넘은 라운드가 얼마나 되는가"를 뜻한다.

## 출력

| 파일 | 내용 |
|---|---|
| `krum_violation.csv` | 공격 종류·악성 비율별: `runs`, `n`, `rounds`, `violated`, `ratio_pct`, `attack_rounds`, `attack_violated`, `attack_ratio_pct`, `expected_pct` |
| `krum_violation_runs.csv` | 실행별: 위와 같은 열(이론값 제외) |

## 참고: 현재 설정의 이론값 (N = 100, n = 10)

| 악성 비율 | 10% | 20% | 30% |
|---|---|---|---|
| 위반 확률 | 0.8% | 11.0% | 34.6% |

이항분포 근사(1.3%, 12.1%, 35.0%)보다 약간 낮다. 클라이언트를 비복원으로 뽑기 때문이다.

## 테스트

```bash
python -m pytest tests/test_krum_violation.py
```
