# 공격 모듈 (attacks/)

담당: 예다은 · 가이드 "예다은 가이드 > 구현" 1~4번에 해당한다.

## 구성

| 파일 | 내용 | 가이드 항목 |
|---|---|---|
| `attacks/label_flip.py` | 라벨 매핑(reverse / pairs / all_to_one), 라벨만 바꾸는 데이터셋 래퍼 | 1. 라벨 플리핑 |
| `attacks/model_poison.py` | Δ' = ±s·Δ (+ 선택적 노이즈), 조작한 Δ로 로컬 가중치 재구성 | 2. 모델 오염 |
| `attacks/schedule.py` | `[start_round, end_round)` 구간 판정 | 3. 구간 공격 |
| `attacks/selector.py` | 악성 클라이언트 지정 (fixed / random) | 4. 지정 방식 |
| `attacks/manager.py` | 학습 루프가 부르는 진입점 `AttackManager` | - |
| `attacks/naming.py` | 결과 파일 이름 `알고리즘_집계_공격_비율_시드.csv` | 공통 지침 |
| `configs/attack/{none,lf,mp}.yaml` | 공격 설정 (실험값은 여기에만 있다) | 공통 지침 |
| `configs/experiments/{lf,mp}_sweep.yaml` | 실험 조합 2×5×3×3 | 실험 |
| `scripts/make_sweep.py` | 실행 명령 생성, 누락 결과 확인 | 체크리스트 |
| `integration/fedacg_integration.patch` | FedACG 공개 코드에 연결하는 패치 | - |

## 동작 정의 (3장 "공격 방법"에 그대로 옮길 수 있게 적음)

- **악성 클라이언트 수**: 전체 N명 중 f_total = round(ratio · N)명. 라운드 r의 참여자 n명 중 악성 수를 f_r로 기록한다.
- **지정 방식**
  - `fixed`: 시드로 한 번 뽑은 f_total명이 학습 내내 악성이다.
  - `random`: 라운드마다 전체에서 f_total명을 다시 뽑는다.
  - 두 방식 모두 별도 RNG를 쓴다. 그래서 같은 시드면 무공격 실행과 공격 실행의 참여 클라이언트 표본이 같다.
- **구간 공격**: 라운드 r이 `start_round ≤ r < end_round`일 때만 공격한다. 구간 밖에서는 악성 클라이언트도 정상 학습한 결과를 그대로 보낸다.
- **라벨 플리핑**: 악성 클라이언트는 라벨을 `map[y]`로 바꾼 로컬 데이터로 정상 절차대로 학습한다. 기본값 `reverse`는 y → C−1−y이다.
- **모델 오염**: 악성 클라이언트는 정상 학습 후 Δ = w_local − w_recv를 Δ' = −s·Δ(`sign_flip: true`) 또는 s·Δ로 바꿔 보낸다.
  - FedACG에서 w_recv는 lookahead 모델(w + λm)이다.
  - 서버의 강건 집계는 조작된 Δ'를 그대로 받는다.
  - 정수 버퍼(num_batches_tracked)는 조작하지 않는다.

## 학습 루프 연결 지점 (3곳)

```python
# 1) Trainer.__init__
self.attack = AttackManager.from_args(self.args, num_classes=len(self.datasets['train'].classes))

# 2) Trainer.local_update: 로컬 데이터셋 생성 직후 / 로컬 학습 직후
local_dataset = self.attack.maybe_flip_labels(client_idx, rnd, local_dataset)
local_model   = self.attack.maybe_poison(client_idx, rnd, local_model, task['state_dict'])

# 3) Trainer.train: 참여 클라이언트 선택 직후
attack_info = self.attack.begin_round(rnd, selected_client_ids)   # n, f_r 기록
```

- 공격은 모두 클라이언트 쪽(`local_update`)에서 적용한다. 그래서 `multiprocessing=True`(spawn)에서도 그대로 동작한다.
- 서버 쪽 코드(집계 함수)는 공격 모듈을 몰라도 된다.

## 로그

`attack_log.csv`에 라운드마다 다음 열을 남긴다.
- `round`, `attack_active`, `n_selected`(n), `n_malicious`(f_r), `malicious_ids`

강윤서 님 담당:
- 결과 CSV(정확도, EMA, ‖m‖)와 `round` 기준으로 합칠 수 있다.
- Krum 조건 위반 비율은 `n_selected > 2 * n_malicious + 2`가 깨지는 라운드의 비율로 바로 계산할 수 있다.

## 실행 예

```bash
# 무공격
python federated_train.py server=FedACG client=ACG dataset=cifar10 attack=none
# 라벨 플리핑 20%, 100~199 라운드 공격
python federated_train.py server=FedACG client=ACG dataset=cifar10 attack=lf attack.ratio=0.2
# 모델 오염, 배율/구간 변경
python federated_train.py server=FedACG client=ACG dataset=cifar10 attack=mp \
    attack.model_poison.scale=5 attack.start_round=50 attack.end_round=150

# 라벨 플리핑 실험 90회 명령 생성 / 누락 확인
python scripts/make_sweep.py configs/experiments/lf_sweep.yaml > runs_lf.sh
python scripts/make_sweep.py configs/experiments/lf_sweep.yaml --check-results results/
```

## 첫 회의에서 확정할 값 ([TBD])

| 항목 | 임시 기본값 | 위치 |
|---|---|---|
| 데이터셋 | CIFAR-10 | `configs/experiments/*_sweep.yaml` |
| 라벨 매핑 | reverse (y → 9−y) | `configs/attack/lf.yaml` |
| 모델 오염 방식 | 부호 반전 + 배율 3 | `configs/attack/mp.yaml` |
| 악성 지정 방식 | fixed | `configs/attack/*.yaml` |
| 공격 구간 | 100 ≤ r < 200 | `configs/attack/*.yaml` |
| 클라이언트 수·참여율·라운드 수 | 100명 · 0.1 (n=10) · 미정 | `configs/experiments/*_sweep.yaml` |
| 집계 override 이름 | `+server.aggregator={mean,krum,...}` | 우은식 님 구현에 맞춰 수정 |

## 테스트

```bash
pip install pytest pyyaml omegaconf torch numpy
python -m pytest
```
