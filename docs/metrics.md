# 지표 로깅 (metrics/)

담당: 강윤서 · 가이드 "강윤서 가이드 > 구현" 1·2번에 해당한다. 3번(목표 도달 라운드·회복 라운드 계산 스크립트)은 회복 라운드 정의가 확정된 뒤 별도 PR로 올린다.

## 구성

| 파일 | 내용 | 가이드 항목 |
|---|---|---|
| `metrics/logger.py` | `MetricsLogger`(라운드별 CSV 기록, EMA 계산), `momentum_norm`(‖m‖), `result_csv_name`(결과 파일 이름) | 1. 정확도·EMA, 2. ‖m‖ |
| `configs/metrics/default.yaml` | EMA 계수(0.9), 결과 폴더 | 공통 지침 |
| `integration/metrics_integration.patch` | FedACG 학습 루프 연결. `fedacg_integration.patch` 다음에 적용한다 | - |

## 결과 CSV

파일 이름은 공통 지침대로 `알고리즘_집계_공격_비율_시드.csv`이고 `results/` 아래에 생긴다(예: `results/fedacg_krum_lf_20_s1.csv`, 무공격은 `results/fedavg_mean_none_0_s1.csv`).

| 열 | 뜻 |
|---|---|
| `round` | 라운드 번호 (0부터). `attack_log.csv`와 이 열로 합친다 |
| `acc` | 테스트 정확도 (%) |
| `ema_acc` | EMA 정확도 (%) |
| `m_norm` | 서버 모멘텀 크기 ‖m‖. FedAvg는 빈 칸 |

## 동작 정의 (3장 "평가 지표"에 옮길 수 있게 적음)

- **EMA 정확도**: 첫 평가 라운드의 정확도에서 시작해 `ema ← 0.9·ema + 0.1·acc`로 갱신한다.
- **‖m‖**: 그 라운드의 집계가 끝난 직후 서버 모멘텀 m의 전체 L2 노름이다. 모든 층의 실수 텐서를 하나의 벡터로 보고 계산하며, 정수 버퍼는 제외한다.
- **기록 시점**: 평가하는 라운드(`eval.freq`)마다 한 줄씩 바로 파일에 쓴다. 학습이 중간에 끊겨도 그때까지의 라운드는 남는다.
- **알고리즘·집계 이름**: `server.FedACG`가 켜져 있으면 `fedacg`, 아니면 `fedavg`. 집계는 `server.aggregator`를 쓰고, 없으면 `mean`이다.

## 적용 순서

```bash
git clone https://github.com/geehokim/FedACG.git && cd FedACG
git apply ../2026-IoT/integration/fedacg_integration.patch
git apply ../2026-IoT/integration/metrics_integration.patch
cp -r ../2026-IoT/attacks ../2026-IoT/metrics .
cp -r ../2026-IoT/configs/attack ../2026-IoT/configs/metrics configs/
```

## 확인할 점

- **집계 설정 이름**: 우은식 님 집계 구현의 설정 키가 `server.aggregator`가 아니면 `result_csv_name`을 맞춰 고친다.
- **이어서 학습(resume)**: 현재 FedACG 체크포인트는 모델 가중치만 저장한다. `load_model_path`로 이어 돌리면 결과 CSV가 새로 만들어지고 EMA와 m도 처음부터 다시 시작하므로, 이어 붙인 결과는 쓰지 않는다.

## 테스트

```bash
python -m pytest tests/test_metrics.py
```
