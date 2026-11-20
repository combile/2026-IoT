# 결과 분석 (analysis/)

담당: 강윤서 · 가이드 "강윤서 가이드 > 구현" 3번(목표 도달 라운드·회복 라운드 계산 스크립트)에 해당한다.

## 실행

```bash
python scripts/analyze.py results/                       # results/summary.csv 생성
python scripts/analyze.py results/ --target-acc 55 --tolerance 0.5
```
코랩에서는 `!python /content/2026-IoT/scripts/analyze.py /content/drive/MyDrive/2026-IoT/results`.
끝나지 않은 실행도 그때까지의 라운드로 계산하므로 실험 도중에도 돌려 볼 수 있다.

## 구성

| 파일 | 내용 |
|---|---|
| `analysis/summary.py` | 실행별 요약 계산, `summary.csv` 쓰기 |
| `scripts/analyze.py` | 명령줄 실행 |
| `configs/analysis/default.yaml` | 목표 정확도, 회복 허용치 (팀 확정 전 임시값, `[TBD]`) |

## summary.csv 열

| 열 | 뜻 |
|---|---|
| `name`, `algorithm`, `aggregator`, `attack`, `ratio`, `seed` | 결과 파일 이름과 그 구성 |
| `last_round` | 기록된 마지막 라운드 |
| `max_acc` | 최고 테스트 정확도 (%) |
| `final_ema_acc` | 마지막 라운드의 EMA 정확도 (%) |
| `target_round` | 목표 도달 라운드 |
| `baseline_max_acc` | 같은 조건 무공격 기준선의 최고 정확도 (%) |
| `acc_drop_pp` | 정확도 하락 (%p) = `baseline_max_acc − max_acc` |
| `attack_start`, `attack_end` | 공격 구간 [start, end) |
| `recovery_round` | 회복한 라운드 |
| `recovery_rounds` | 회복에 걸린 라운드 수 = `recovery_round − attack_end` |

구할 수 없는 값(도달 못 함, 회복 못 함, 기준선 없음, 공격 도중 끊김)은 빈 칸이다.

## 동작 정의 (3장 "평가 지표"에 옮길 수 있게 적음)

- **같은 조건 기준선**: 알고리즘·집계·시드가 같은 무공격 실행(`<알고리즘>_<집계>_none_0_s<시드>.csv`).
- **정확도 하락**: 기준선 최고 정확도 − 공격 실행 최고 정확도. 서베이의 공격 영향(AIm) 정의를 따른다.
- **목표 도달 라운드**: EMA 정확도가 처음으로 `target_acc` 이상이 된 라운드.
- **공격 구간**: 공격 기록(`results/attack_logs/`)에서 `attack_active = 1`인 첫 라운드부터 마지막 라운드 + 1까지. 기록의 마지막 라운드까지 공격 중이면(학습이 공격 도중 끊김) 종료 라운드를 알 수 없어 회복도 계산하지 않는다.
- **회복 라운드**: 공격 종료 라운드부터 보아, EMA 정확도가 같은 라운드의 기준선 EMA − `recovery_tolerance`(%p) 이상이 된 첫 라운드. 두 실행 모두 기록이 있는 라운드만 비교한다.

## 팀에서 정할 값

- `target_acc`: 1차 계획이 80라운드라 FedACG 논문 목표치(CIFAR-10 81%, 85%)에는 닿지 않는다. 무공격 기준선 결과를 보고 정한다.
- `recovery_tolerance`: 기준선과 몇 %p 이내면 회복으로 볼지.

## 테스트

```bash
python -m pytest tests/test_analysis.py
```
