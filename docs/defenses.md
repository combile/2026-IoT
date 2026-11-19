# 방어 모듈 (defenses/)

담당: 우은식 · 가이드 "우은식 가이드 > 구현" 2~4번에 해당한다.

## 구성

| 파일 | 내용 | 가이드 항목 |
|---|---|---|
| `defenses/aggregators.py` | Krum, Multi-Krum, Trimmed Mean, Median과 서버 연결 함수 `robust_aggregate` | 2. 집계 함수 분리 |
| `configs/defense/default.yaml` | 방어자가 가정하는 악성 비율 `assumed_ratio` | 4. f를 설정에서 받음 |
| `integration/defense_integration.patch` | FedACG `servers/base.py`의 `Server`, `ServerM` 집계에 연결. 다른 패치와 겹치지 않으므로 적용 순서는 상관없다 | - |
| `integration/colab_memory.patch` | 코랩 무료판(메모리 12.7GB)용. FedACG 원본은 처음에 클라이언트 100명분 모델(1명당 2개)을 모두 만들다가 메모리가 부족해 멈춘다. 이 패치는 클라이언트 객체 하나를 돌려 쓰게 바꾼다. 순차 실행(`multiprocessing=False`)에서는 결과가 원본과 같다 | - |
| `notebooks/colab_run.ipynb` | 코랩 실행 순서 (설치, 패치, 테스트, 2라운드 점검, 실험) | - |
| `tests/test_defenses.py` | 집계 규칙과 원본 집계 일치 여부 테스트 | - |

## 집계 방식 고르기

```bash
# 평균(원본 FedACG 코드 그대로)
python federated_train.py server=FedACG client=ACG dataset=cifar10 attack=mp
# Krum
python federated_train.py server=FedACG client=ACG dataset=cifar10 attack=mp +defense=default +server.aggregator=krum
```

`server.aggregator`: `mean`, `krum`, `multikrum`, `trimmedmean`, `median`. 값이 없으면 `mean`이다. 결과 파일 이름의 집계 부분(`attacks/naming.py`, `metrics/logger.py`)도 이 값을 따른다.

## 동작 정의 (3장 "방어 방법"에 그대로 옮길 수 있게 적음)

- **집계 대상**: 클라이언트 변화량 Δ_i = w_i − w_recv. FedACG에서 w_recv는 lookahead 모델 w + λm이다.
- **갱신 순서**: 강건 집계로 D를 구한 뒤 새 모델 w' = w_recv + D, 서버 모멘텀 m' = λm + D로 갱신한다. FedACG에서는 w' = w + m'와 같다.
- **악성 수 가정**: 라운드 참여자 n명 중 f = floor(assumed_ratio · n). 기본값 0.3, n = 10이면 f = 3이다.
  - 방어자는 실제 악성 수(attack.ratio)를 모른다고 보고 모든 실행에 같은 f를 쓴다.
- **Krum**: 각 Δ_i에 대해 가장 가까운 n − f − 2개까지의 거리 제곱합을 점수로 매긴다. 점수가 가장 낮은 Δ 하나를 D로 쓴다.
- **Multi-Krum**: 점수가 낮은 n − f개의 평균을 D로 쓴다.
- **Trimmed Mean**: 좌표마다 위아래 f개씩 버리고 평균한다. 잘라 내는 비율 τ = f / n이다(FedACG의 β와 구분).
- **Median**: 좌표마다 중앙값을 쓴다. n이 짝수면 가운데 두 값의 평균이다.
- **Krum 조건**: n > 2f + 2가 깨지면 경고를 한 번 남기고 그대로 진행한다. n = 10, f = 3이면 10 > 8로 조건을 만족한다.
- **정수 버퍼**(num_batches_tracked 등)는 원본처럼 평균만 낸다.

## 로그

Krum과 Multi-Krum은 라운드마다 고른 클라이언트 ID를 전체 로그 파일에 남긴다.

```
[Defense] krum f=3 selected clients: [51]
```

`attack_log.csv`의 `malicious_ids`와 맞춰 보면 악성 클라이언트가 걸러졌는지 4장에서 확인할 수 있다.

## 확인한 것

- `python -m pytest`: 39개 통과 (공격 24개 + 방어 15개)
- f = 0인 Multi-Krum(전원 평균)의 새 모델과 모멘텀이 원본 `ServerM.aggregate` 결과와 같다(FedACG, FedAvgM, FedAvg).
- 부호 반전 ×10 공격자 3명/10명에서 네 방식 모두 평균보다 정상 방향에 가깝다. Krum 계열은 공격자를 고르지 않는다.
- FedACG 원본 코드에 `fedacg_integration.patch`, `defense_integration.patch`를 적용한 상태에서 ResNet-18 실제 크기(파라미터 1,117만 개)로 네 방식이 모두 돌아간다.
- `colab_memory.patch`를 적용하기 전과 후에 CIFAR-10 1라운드를 같은 시드로 돌렸다(FedACG + Krum, 모델 오염 30%). 클라이언트 10명의 손실, Krum이 고른 클라이언트([54]), 정확도, ‖m‖이 모두 같았다.

## 첫 회의에서 확정할 값 ([TBD])

| 항목 | 임시 기본값 | 위치 |
|---|---|---|
| 방어자가 가정하는 악성 비율 | 0.3 (실험 최대 악성 비율) | `configs/defense/default.yaml` |
