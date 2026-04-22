# final_code

`final_code/`는 `/home/namkyeong/ssmo/final_code` 아래에 둔 standalone 재현 번들입니다.

- 실험 실행은 전부 shell script로 합니다.
- 결과 집계와 최종 figure 생성은 전부 Jupyter notebook에서 합니다.
- 외부 결과를 복사해서 쓰지 않고, `final_code/results/`에 생성된 결과만 사용합니다.
- notebook은 `final_code/results/`만 읽고, 표/그림/요약은 `final_code/artifacts/`와 `final_code/Results_Summary.md`로 export합니다.
- 기본 실행 환경은 conda `ssml`입니다.

## 폴더 구조

```text
final_code/
├── artifacts/
│   ├── figures/
│   └── tables/
├── config/
├── data/
├── logs/
├── models/
├── notebooks/
├── results/
├── runners/
├── scripts/
└── src/
```

## 환경

```bash
source /home/namkyeong/anaconda3/etc/profile.d/conda.sh
conda activate ssml
```

스크립트 내부에서도 `ssml` 환경을 다시 활성화하도록 되어 있습니다.

## 데이터

현재 번들에는 wrap-up 재현에 필요한 데이터가 `final_code/data/` 아래로 직접 복사되어 있습니다.

- Time-series: `ETTh1`, `electricity`, `weather`
- Operator: `burgers_data_R10.mat`, `darcy`
- Classification: `cifar-10-batches-py`, `cifar-100-python`

다른 머신으로 `final_code/`만 따로 옮겨 쓰는 경우에도, 아래 경로가 그대로 유지되면 추가 fallback 없이 실행됩니다.

### Time-series

```text
final_code/data/time_series/ETT-small/ETTh1.csv
final_code/data/time_series/electricity/electricity.csv
final_code/data/time_series/weather/weather.csv
```

### Operator

```text
final_code/data/burgers_data_R10.mat
final_code/data/darcy/
```

`run_burgers.sh`는 `burgers_data_R10.mat`를 먼저 확인하고, `final_code/data/burgers_data_R10.mat.zip`가 있으면 거기서 복원합니다.

Darcy는 `neuralop` 경로를 우선 시도하고, 현재 `ssml` 환경에서 해당 import가 불안정할 경우 로컬 `.pt` 데이터셋 경로로 fallback 하도록 정리해 두었습니다.

### Classification

현재 번들에는 `CIFAR-10`, `CIFAR-100`도 이미 포함되어 있습니다. 기본 스크립트는 그대로 써도 되고, 필요하면 아래처럼 다운로드를 막고 로컬 데이터만 쓰게 할 수 있습니다.

```bash
DOWNLOAD=0 bash final_code/scripts/run_experiment.sh cifar10
```

## 실행

전체 실행:

```bash
bash final_code/scripts/run_all.sh
```

2개 GPU 병렬 실행:

```bash
bash final_code/scripts/run_all_parallel.sh
```

멀티 호스트 분산 실행 (`gpu0`는 비워 두고 `gpu1 + worker1 + worker2` 사용):

```bash
bash final_code/scripts/run_all_distributed.sh
```

장시간 분산 런은 현재 사용 중인 셸이나 `tmux` 세션에서 직접 실행하는 것을 권장합니다. 실행이 시작되면 `final_code/results/_dispatch/<run_id>/summary.txt`에 실험별 host/GPU 배치가 바로 기록됩니다.

기본 분배는 아래와 같습니다.

- `localhost gpu1 (RTX 4090)`: `cifar100_cifarstem`
- `worker2 gpu0 (RTX 3090 Ti)`: `etth1`, `electricity`
- `worker1 gpu0 (RTX 2080 Ti)`: `weather`, `darcy`
- `worker1 gpu1 (RTX 2080 Ti)`: `burgers`, `cifar10`

분산 실행 기록은 아래에 남습니다.

```text
final_code/results/_dispatch/<run_id>/plan.tsv
final_code/results/_dispatch/<run_id>/events.tsv
final_code/results/_dispatch/<run_id>/summary.txt
```

`events.tsv`에는 각 experiment의 시작/종료 시각, host, GPU, 상태, scheduler log 경로가 기록됩니다.

분산 실행도 dry-run으로 먼저 확인할 수 있습니다.

```bash
DRY_RUN=1 bash final_code/scripts/run_all_distributed.sh
```

기본값은 `GPU 0`, `GPU 1`에 queue를 나누고, 각 GPU에서 assigned된 experiment를 가능한 만큼 전부 동시에 올립니다.

또한 각 experiment 내부에서도 의존성이 없는 leaf run은 동시에 실행합니다. 예를 들어:

- `weather`, `darcy`: `independent / dml / ssml`와 seed별 run 동시 실행
- `cifar10`: `core(independent+ssml)`와 `dml` 동시 실행
- `etth1`: `independent`와 `dml` 동시 실행 후 `ssml`
- `burgers`: `followup` 후 `ctrl`와 `ssml` 동시 실행
- `cifar100_cifarstem`: `pool` 후 `independent`, `dml`, `ssml` 동시 실행

```bash
GPU0_EXPERIMENTS="weather etth1 cifar100_cifarstem" \
GPU1_EXPERIMENTS="electricity burgers darcy cifar10" \
  bash final_code/scripts/run_all_parallel.sh
```

동시 실행 수를 제한하고 싶으면:

```bash
JOBS_PER_GPU=3 bash final_code/scripts/run_all_parallel.sh
```

GPU별로 따로 주고 싶으면:

```bash
JOBS_PER_GPU0=2 JOBS_PER_GPU1=3 bash final_code/scripts/run_all_parallel.sh
```

experiment 내부 leaf run 병렬 수를 제한하고 싶으면:

```bash
MAX_PARALLEL_RUNS=4 bash final_code/scripts/run_all_parallel.sh
```

wrapper stage 병렬 수를 제한하고 싶으면:

```bash
CIFAR10_STAGE_MAX_PARALLEL_RUNS=1 \
ETTH1_STAGE_MAX_PARALLEL_RUNS=2 \
  bash final_code/scripts/run_all_parallel.sh
```

GPU 번호를 바꾸고 싶으면:

```bash
GPU_IDS="1 0" bash final_code/scripts/run_all_parallel.sh
```

개별 실행:

```bash
bash final_code/scripts/run_experiment.sh weather
bash final_code/scripts/run_experiment.sh electricity
bash final_code/scripts/run_experiment.sh etth1
bash final_code/scripts/run_experiment.sh burgers
bash final_code/scripts/run_experiment.sh darcy
bash final_code/scripts/run_experiment.sh cifar10
bash final_code/scripts/run_experiment.sh cifar100_cifarstem
```

GPU 지정 예시:

```bash
GPU=1 bash final_code/scripts/run_experiment.sh weather
```

실행 전 명령만 확인:

```bash
DRY_RUN=1 bash final_code/scripts/run_experiment.sh weather
```

## Notebook 분석

```bash
jupyter lab final_code/notebooks
```

권장 순서:

1. `00_index.ipynb`
2. `01_time_series_wrapup.ipynb`
3. `02_operator_wrapup.ipynb`
4. `03_classification_wrapup.ipynb`
5. `04_final_wrapup.ipynb`

Notebook에서 직접 수정 가능한 항목:

- seed 선택 / 제외
- epoch window
- legend 이름
- export on / off

## 산출물 경로

실험 로그:

```text
final_code/logs/
```

실험 결과:

```text
final_code/results/
```

Notebook export:

```text
final_code/artifacts/figures/
final_code/artifacts/tables/
```

최종 notebook에서 export하면 아래 파일이 생성됩니다.

```text
final_code/artifacts/figures/final_wrapup_time_series.png
final_code/artifacts/figures/final_wrapup_operator.png
final_code/artifacts/figures/final_wrapup_classification.png
final_code/artifacts/tables/final_wrapup_summary.csv
final_code/artifacts/tables/final_wrapup_summary.md
```

## 빠른 스모크 체크

번들 정리 전에 각 wrapper는 아래 조건으로 한 번씩 스모크 실행을 통과했습니다.

- 공통: `SEEDS=0`, `EPOCHS=1`
- classification: `TRAIN_SUBSET_SIZE=64`, `VAL_SUBSET_SIZE=64`
- smoke 산출물은 배포 상태에서 다시 비워 두었습니다.

예시:

```bash
SEEDS=0 EPOCHS=1 BATCH_SIZE=512 NUM_WORKERS=0 DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
  bash final_code/scripts/run_experiment.sh weather
```

```bash
SEEDS=0 EPOCHS=1 BATCH_SIZE=64 NUM_WORKERS=0 DEVICE=cuda CUDA_VISIBLE_DEVICES=0 \
DOWNLOAD=0 TRAIN_SUBSET_SIZE=64 VAL_SUBSET_SIZE=64 \
  bash final_code/scripts/run_experiment.sh cifar100_cifarstem
```

`cifar100_cifarstem` smoke에서는 pool run의 `best_model.pt`가 아직 생기지 않은 경우 같은 run 디렉터리의 `model.pt`를 임시 초기화 체크포인트로 fallback 하도록 해 두었습니다. full rerun에서는 `best_model.pt`가 있으면 그 파일을 그대로 사용합니다.

## 현재 상태

- 실험 결과와 분산 실행 기록은 `final_code/results/` 아래에만 둡니다.
- 실행 로그는 `final_code/logs/` 아래에만 둡니다.
- 최종 숫자, 표, 그림은 `04_final_wrapup.ipynb`가 `final_code/artifacts/`로 export합니다.
- 요약 문서는 `final_code/Results_Summary.md`이며, `04_final_wrapup.ipynb`의 `EXPORT_RESULTS_SUMMARY = True` 설정으로 다시 생성됩니다.
