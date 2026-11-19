# 이어 돌리기 (resume/)

코랩 런타임이 끊겨도 같은 명령을 다시 실행하면 마지막 체크포인트부터 이어 돈다. 이어 돌린 결과는 끊기지 않은 실행과 같다.

## 왜 필요한가

- FedACG 원본 체크포인트(`res18.pt`)에는 모델 가중치만 들어 있다.
- 그래서 원본 방식(`load_model_path`)으로 이어 돌리면 서버 모멘텀 m이 0부터 다시 시작하고, EMA와 난수 상태도 처음으로 돌아간다.
- 이렇게 이어 돌린 FedACG 결과는 끊기지 않은 실행과 다르므로 논문에 쓸 수 없다.

## 구성

| 파일 | 내용 |
|---|---|
| `resume/state.py` | `ResumeManager`: 체크포인트 저장, 불러오기, 설정 비교, 끝나면 삭제 |
| `resume/csv_rows.py` | 이어 돌릴 때 라운드별 CSV를 체크포인트 시점까지 잘라 냄 |
| `configs/resume/default.yaml` | 저장 주기, 위치, 끝난 뒤 삭제 여부, 설정 비교 엄격도 |
| `integration/resume_integration.patch` | 학습 루프 연결. `colab_memory.patch` 다음에 적용한다 |
| `tests/test_resume.py` | 단위 테스트 15개 |

패치 적용 순서는 다음과 같다.
1. `fedacg_integration`
2. `metrics_integration`
3. `defense_integration`
4. `colab_memory`
5. `resume_integration`

## 체크포인트 내용

위치는 `<metrics.results_dir>/checkpoints/<결과 이름>.r<라운드>.pt`이다(예: 라운드 19까지 끝낸 체크포인트는 `results/checkpoints/fedacg_krum_lf_20_s1.r00019.pt`). 코랩에서는 결과 폴더가 드라이브에 있어서 체크포인트도 드라이브에 남는다.

| 항목 | 내용 |
|---|---|
| `round` | 마지막으로 끝낸 라운드 r. 다시 시작하면 r+1부터 돈다 |
| `model` | 그 라운드 집계가 끝난 뒤의 글로벌 모델 |
| `server` | `global_momentum`(m), `global_delta`, `global_v` 중 서버에 있는 것. FedAvg는 비어 있다 |
| `ema` | EMA 정확도 |
| `rng` | python `random`, numpy, torch(CPU, CUDA) 난수 상태. 클라이언트 선택과 데이터 섞기가 끊기지 않은 실행과 같아진다 |
| `config` | 실행 설정. 다시 시작할 때 비교한다 |

## 동작

- **저장**
  - `resume.every`라운드마다 저장한다(기본 10).
  - 라운드 번호가 붙은 새 파일에 다 쓴 뒤 이 실행의 예전 체크포인트를 지운다. 저장 중에 끊겨도 이전 체크포인트는 남는다.
  - 덮어쓰지 않는 이유: 구글 드라이브는 이름이 같은 파일을 여러 개 허용한다. 공유받은 폴더에서 임시 파일을 같은 이름으로 바꿔 덮어쓰자(`os.replace`) 교체되지 않고 `이름 (1).pt`, `이름 (2).pt` 복사본이 쌓였다(2026-11-19 확인).
  - 불러올 때는 이 실행의 체크포인트 파일(예전 이름과 드라이브 복사본 포함) 중 읽을 수 있고 라운드가 가장 큰 것을 쓴다. 저장 중에 잘린 파일은 건너뛴다.
- **이어 돌리기**
  - 같은 이름의 체크포인트가 있으면 자동으로 이어 돈다.
  - 결과 CSV, 공격 기록, 실행 로그는 덮어쓰지 않는다.
  - 체크포인트 이후에 기록된 라운드는 지우고 다시 쓴다. 그래서 같은 라운드가 두 번 기록되지 않는다.
- **끝나면**: `keep_final: false`(기본)면 이 실행의 체크포인트 파일(드라이브 복사본 포함)을 모두 지운다. 체크포인트는 실행당 약 130MB라 90회분을 남기면 드라이브 무료 용량(15GB)을 넘는다.
- **설정 비교**
  - 체크포인트를 저장할 때와 결과에 영향을 주는 설정이 다르면 멈춘다(`strict: true`).
  - 라운드 수(`trainer.global_rounds`), 경로, 화면 출력 수준은 비교하지 않는다. 라운드 수를 늘려 이어 돌릴 수 있다.
- **처음부터 다시 돌리기**: 체크포인트 파일을 지우거나 `resume.enabled=false`를 붙인다.

## 실행 예

```bash
# 평소와 같다. 끊기면 같은 명령을 다시 실행한다
python federated_train.py server=FedACG client=ACG dataset=cifar10 attack=lf attack.ratio=0.2 seed=1
# 5라운드마다 저장
python federated_train.py ... resume.every=5
```

## 확인한 것

- 단위 테스트를 작성했다.
  - 저장과 복원 후 모델과 m이 같다.
  - 난수 상태를 되돌리면 같은 수열이 나온다.
  - EMA가 끊기지 않은 실행과 같은 값으로 이어진다.
  - 공격 기록과 로그 파일을 이어 쓴다.
  - 설정이 다르면 멈춘다.
- 실제 FedACG 실행으로 비교했다(CIFAR-10, CPU, 모델 오염 30%, 3라운드). 끊기지 않은 실행(A)과, 라운드 1 체크포인트 직후 강제 종료(SIGKILL) 뒤 다시 실행한 것(B)의 결과 CSV와 공격 기록이 바이트 단위로 같았다.

## 한계

- `multiprocessing=True` 모드에서는 자식 프로세스의 난수 상태를 저장하지 않는다. 코랩은 순차 실행이라 해당 없다.
- GPU 연산 자체의 비결정성(cuDNN)이 있으면 이어 돌린 결과가 마지막 자리에서 조금 다를 수 있다. 끊기지 않은 실행끼리도 생기는 차이다.
- 원본 코드가 학습이 끝날 때 `checkpoints/.../attack_log.csv`에 저장하는 예전 공격 기록은 이어 돌린 뒤의 라운드만 담는다. 공격 기록은 `results/attack_logs/`를 쓴다.
