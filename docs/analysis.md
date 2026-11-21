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
| `configs/analysis/default.yaml` | 목표 정확도, 회복 허용치 (무공격 기준선으로 정한 값) |

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

## 기준값과 근거

무공격 기준선 30회(FedAvg·FedACG × 집계 5종 × 시드 1~3, 80라운드)의 EMA 정확도로 정하였다.

### 목표 정확도 `target_acc` = 35%

목표값별 도달 라운드(시드 1, 2, 3 순, `–`는 80라운드 안에 도달하지 못함):

| 조합 | 30% | 35% | 40% | 45% | 50% |
|---|---|---|---|---|---|
| FedAvg + mean | 33, 33, 33 | 45, 44, 44 | 67, 63, 65 | – | – |
| FedAvg + multikrum | 39, 40, 38 | 63, 54, 52 | –, 74, 78 | – | – |
| FedAvg + trimmedmean | 37, 36, 35 | 52, 50, 47 | 72, 69, 67 | – | – |
| FedAvg + median | 38, 36, 35 | 54, 51, 48 | 74, 72, 70 | – | – |
| FedACG + mean | 24, 25, 22 | 32, 33, 29 | 41, 42, 38 | 55, 56, 48 | 67, 67, 60 |
| FedACG + multikrum | 30, 30, 29 | 38, 40, 38 | 50, 53, 50 | 66, 65, 63 | –, 79, 76 |
| FedACG + trimmedmean | 27, 30, 27 | 35, 37, 36 | 45, 47, 45 | 61, 61, 58 | 72, 74, 67 |
| FedACG + median | 29, 31, 28 | 36, 40, 37 | 45, 50, 47 | 61, 63, 60 | 71, 76, 70 |
| FedAvg·FedACG + krum | – | – | – | – | – |

- 35%는 Krum을 제외한 8개 조합이 시드 3회 모두 도달하는 최댓값이다. 50%에는 FedAvg가 한 번도 도달하지 못하고, 40%는 FedAvg + multikrum이 시드 1에서 도달하지 못한다.
- 35%에서도 알고리즘 간 차이가 드러난다(FedACG 29~40라운드, FedAvg 44~63라운드).
- Krum은 무공격에서도 30%에 도달하지 못하므로(최종 정확도 FedAvg 29.8%, FedACG 16.7%) 목표 도달 라운드는 "도달 못 함"으로 표기하고 원인은 본문에서 논의한다.

### 회복 허용치 `recovery_tolerance` = 2.0%p

- 회복은 같은 시드 기준선과 비교하므로, 허용치는 공격이 없어도 생기는 차이보다 커야 한다.
- 50~79라운드에서 같은 라운드 EMA 정확도의 시드 간 표준편차는 FedAvg 0.4~0.9%p, FedACG 0.8~2.0%p이다. 라운드 간 EMA 변동은 0.24~0.40%p이다.
- 1.0%p는 FedACG의 자연 변동보다 작아 공격과 무관한 차이로 회복 실패가 판정될 수 있다. 2.0%p는 두 알고리즘의 자연 변동을 모두 포함한다.

## 테스트

```bash
python -m pytest tests/test_analysis.py
```
