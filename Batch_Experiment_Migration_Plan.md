# Batch Experiment Migration Plan

- Date: 2026-02-13
- Branch: `extensive_experiment`
- Scope: 지금은 구현하지 않고, 노트북 기반 실험을 일괄 실행 가능한 형태로 전환하기 위한 실행 계획만 정의한다.

## 1. Goal

`Instruction.md`의 실험 요구사항(Generality, Heterogeneity, Negative transfer 억제, Dynamics 측정)을 충족할 수 있도록,
현재 `.ipynb` 중심 파이프라인을 CLI 기반 배치 실험 파이프라인으로 전환한다.

## 2. Current State Snapshot

1. 실험 로직이 노트북에 집중되어 있다.
2. 주요 노트북:
- `ssmo_Classification/ssmo-exercise-MNIST-Seed(0-4).ipynb`
- `ssmo_Classification/ssmo-exercise-CIFAR10-Seed(0-4).ipynb`
- `ssmo_OperatorLearning/ssmo-Burgers1d-Seed(0-4)).ipynb`
- `ssmo_OperatorLearning/ssmo-Darcy-Seed(0-4).ipynb`
3. `models/`에는 모델 정의(FNO/DeepONet/GNOT)가 분리되어 있으나, 학습/실험 오케스트레이션은 노트북에 중복 구현되어 있다.
4. `results/burgers1d/*.csv`는 현재 곡선 배열을 직접 쓰는 방식이라 스키마가 깨져(헤더 반복) 후처리가 어렵다.
5. 의존성 잠금 파일(`requirements.txt`, `pyproject.toml`)이 없다.

## 3. Target Deliverables

1. 노트북 없이 실행 가능한 CLI 엔트리포인트
2. 설정 파일 기반(run matrix) 실험 실행
3. 시드/모델/방법별 자동 반복 실행
4. 재현 가능한 결과 저장 포맷(집계 가능한 CSV + 상세 로그)
5. `Instruction.md` 기준 4개 태스크 동등 중요도 실행 세트 정의

## 4. Step-by-Step Plan

### Step 0. Baseline freeze and guardrails

1. 노트북 결과를 "baseline reference"로 보존한다(수정하지 않음).
2. 실험 공통 규칙 정의:
- seed 고정
- deterministic 옵션
- device 선택(cpu/cuda)
- 출력 디렉토리 규칙
3. 의존성 명세 파일 초안 작성:
- PyTorch, torchvision, numpy, scipy, tqdm, matplotlib, neuralop

Done criteria:
- 같은 seed로 같은 설정 실행 시 주요 metric 재현 가능(허용 오차 범위 정의).

### Step 1. Project structure for batch runs

다음 구조를 신설한다.

```text
src/
  methods/
    independent.py
    naive_mutual.py
    ssmo_hard.py
    ssmo_soft.py
  tasks/
    classification.py
    operator.py
    time_series.py
    segmentation.py
  utils/
    seed.py
    io.py
    metrics.py
    logging.py
  configs/
    classification/
    operator/
    time_series/
    segmentation/
runners/
  run_single.py
  run_pair.py
  sweep.py
```

Done criteria:
- `python -m runners.sweep --help` 호출 가능.
- 4개 태스크(`classification`, `operator`, `time_series`, `segmentation`) 모두 동일 인터페이스로 실행 가능.

### Step 2. Notebook logic extraction

1. 노트북에서 아래 공통 함수들을 모듈로 추출:
- `set_seed`
- dataset/dataloader 생성
- `train_one_epoch`, `evaluate`
- `run_experiment`, `run_experiment_ssmo`
2. DeepONet wrapper(1D/2D)와 task별 전처리 코드를 노트북 밖으로 이동.
3. 중복 제거:
- Burgers/Darcy의 공통 trainer 합치기
- MNIST/CIFAR의 공통 분류 trainer 합치기
4. 4개 태스크 정렬:
- time-series/segmentation용 dataset adapter와 method hook 포인트를 classification/operator와 같은 수준으로 구현

Done criteria:
- 노트북 없이 동일 태스크 1회 학습 실행 가능.

### Step 3. Method unification

방법 축을 공통 인터페이스로 통일한다.

1. `Independent`
2. `Naive mutual` (imitation always ON)
3. `SSMO-hard` (loss 비교 기반 hard mask)
4. `SSMO-soft` (sigmoid gating, 옵션)

Done criteria:
- method 교체가 config의 문자열 변경만으로 가능.

### Step 4. Config-driven run matrix

1. YAML/JSON config로 아래 파라미터 선언:
- task, dataset, model pair, method
- epochs, lr, batch_size
- seeds
- OOD/noise/weak-peer 설정
2. run matrix 자동 확장:
- seed x model_pair x method x condition

Done criteria:
- 단일 명령으로 여러 실험 자동 실행.

### Step 5. Result schema standardization

현재 CSV 깨짐 문제를 해결하기 위해 결과 포맷을 분리한다.

1. `summary.csv` (집계용 scalar only):
- run_id, seed, task, dataset, method, model1, model2,
- best_metric_1, best_metric_2, final_metric_1, final_metric_2,
- condition_tag, wall_time_sec
2. `curves.npz` (epoch curve 저장):
- train_curve_1, test_curve_1, train_curve_2, test_curve_2
3. `metrics.jsonl` (epoch별 상세 로그):
- conflict rate, cosine similarity, mask activation ratio 포함

Done criteria:
- pandas로 바로 읽어서 mean/std 집계 가능.

### Step 6. Instruction-aligned equal-importance coverage

4개 태스크를 동등 중요도로 구성한다.

1. Classification: CIFAR-10/100 + CIFAR-C + label noise + weak peer
2. Operator: Burgers/Darcy/Navier-Stokes + parameter shift + resolution shift
3. Time-series: ETT/Electricity/Weather + temporal/missingness shift
4. Segmentation: VOC (+Cityscapes option) + corruption/resolution/domain shift
5. Dynamics 로그(conflict/cosine/mask/gradient norm ratio) 4개 태스크 공통 수집

Done criteria:
- `Instruction.md`의 4개 태스크 동등 중요도 세트를 config로 재현 가능.

### Step 7. Batch execution UX

1. 실행 커맨드 표준화:
- single run
- sweep run
- aggregate run
2. 실패 복구:
- run 단위 checkpoint
- 이미 완료된 run skip 옵션
3. 로그 표준화:
- 콘솔 요약 + 파일 로그 동시 저장

Done criteria:
- 장시간 sweep 중단 후 재개 가능.

### Step 8. Verification and acceptance

1. simple 검증:
- epoch 2~5로 전 task simple 검증
2. Regression check:
- 노트북 baseline 대비 metric 방향성 일치 확인
3. Sanity plots:
- curve plot, summary table 자동 생성

Done criteria:
- "한 번의 명령으로 전 실험 실행 + 결과 집계" 파이프라인 동작 확인.

## 5. Immediate Execution Order (when implementation starts)

1. Step 1 + Step 2 (구조 생성, 노트북 로직 추출)
2. Step 3 + Step 5 (방법 통합, 결과 스키마 정리)
3. Step 4 + Step 7 (config matrix + 배치 실행)
4. Step 6 + Step 8 (Instruction 정합성 확보 + 검증)

## 6. Definition of Done

1. `.ipynb`를 열지 않고도 전체 실험이 실행된다.
2. seed sweep 결과가 자동 집계되고 표/그래프로 변환 가능하다.
3. hetero pair + negative transfer + dynamics 로그가 재현 가능하다.
4. `Instruction.md` 4개 태스크 동등 중요도 세트가 커맨드 수준에서 재실행 가능하다.
