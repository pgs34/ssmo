# Execution Run Matrix

- Date: 2026-03-30
- Scope: `Instruction.md` 기반 논문용 실행 계획을 현재 코드베이스 기준으로 정리
- Principle: `ready now`와 `needs patch`를 분리한다

## 1. Current Snapshot

- `operator`: strongest result already exists
  - `FNO x DeepONet / Darcy`
  - `SSML < independent < DML`
- `classification`: partially good
  - `CIFAR-10 / homogeneous` is already a clean win
  - `CIFAR-100 / homogeneous` has signal and best clean SSML is now `0.532567 > 0.528533` independent, but comparable clean homogeneous `DML` reference is still not pinned down
  - `heterogeneous CNN x ViT` is still weak on clean IID
- `time-series`: weakest overall
  - `Weather` is already positive
  - `ETTh1` early-stop reruns removed the curve issue, but `independent` is still best
  - `Electricity` still needs a strongest-baseline win

## 2. Ready-Now Matrix

These can run with the current repo without further feature patches.

### 2.1 Classification

- Datasets:
  - `cifar10`
  - `cifar100`
- Pairs:
  - homogeneous: `resnet18:resnet18`
  - heterogeneous: `resnet18:vit_b16`
- Methods:
  - `independent`
  - `dml`
  - `ssml`
- Conditions:
  - clean: `none:0.0`
  - symmetric noise: `0.2`, `0.4`
  - asymmetric noise: `0.2`, `0.4`
- Why this is ready:
  - label noise is already supported in [run_classification.py](/home/namkyeong/ssmo/runners/run_classification.py)

### 2.2 Operator

- Datasets:
  - `burgers`
  - `darcy`
  - `navier_stokes`
- Pairs:
  - heterogeneous: `fno:deeponet`
  - homogeneous follow-up: `fno:fno`, `deeponet:deeponet` if needed
- Methods:
  - `independent`
  - `dml`
  - `ssml`
- Why this is ready:
  - datasets and pairs are already supported in [run_operator.py](/home/namkyeong/ssmo/runners/run_operator.py)

### 2.3 Time-Series

- Datasets:
  - `etth1`
  - `electricity`
  - `weather`
- Pairs:
  - `transformer:dlinear`
- Methods:
  - `independent`
  - `dml`
  - `ssml`
- Why this is ready:
  - standard hetero forecasting pair already works in [run_time_series.py](/home/namkyeong/ssmo/runners/run_time_series.py)

## 3. Needs Patch

These are in `Instruction.md` but not fully wired yet.

### 3.1 Classification

- `CIFAR-10-C`, `CIFAR-100-C`
- peer-specific weak-peer stress:
  - peer-only data down
  - peer-only noise injection
  - explicit small CNN peer like `resnet8`
- full dynamics logs:
  - conflict rate
  - cosine similarity
  - `||g_im|| / ||g_sup||`

### 3.2 Operator

- parameter shift config as first-class CLI options
- resolution shift config
- weak-peer stress by capacity/data/noise
- teacher mask visualization export

### 3.3 Time-Series

- temporal shift as explicit evaluation split variants
- missingness shift
- seasonal subperiod test
- common dynamics logs across all methods

## 4. Worker Allocation

### worker1

- `GPU0`: classification homogeneous matrix
- `GPU1`: classification heterogeneous matrix

### worker2

- `GPU0`: time-series matrix

### worker3

- `GPU0`: operator matrix

## 5. Expected Paper Use

### Immediate paper-ready candidates

- operator heterogeneous result
- classification homogeneous clean win on CIFAR-10
- time-series weather comparison

### Still exploratory

- classification hetero clean IID
- electricity SSML strongest-baseline gap
- CIFAR-100 homogeneous clean claim: larger margin and a comparable `DML` baseline
- ETTh1 early-stop comparison as a diagnostic/control, not a win case

## 6. Output Layout

- classification homogeneous:
  - `results/instruction_matrix_v1/classification_homo`
- classification heterogeneous:
  - `results/instruction_matrix_v1/classification_hetero`
- time-series:
  - `results/instruction_matrix_v1/time_series`
- operator:
  - `results/instruction_matrix_v1/operator`

## 7. Launchers

- worker1:
  - [launch_worker1_instruction_matrix_v1.sh](/home/namkyeong/ssmo/scripts/paper_rerun/cluster/launch_worker1_instruction_matrix_v1.sh)
  - [launch_worker1_cifar100_alt_focus_v1.sh](/home/namkyeong/ssmo/scripts/paper_rerun/cluster/launch_worker1_cifar100_alt_focus_v1.sh)
- worker2:
  - [launch_worker2_instruction_matrix_v1.sh](/home/namkyeong/ssmo/scripts/paper_rerun/cluster/launch_worker2_instruction_matrix_v1.sh)
  - [launch_worker2_electricity_followup_v1.sh](/home/namkyeong/ssmo/scripts/paper_rerun/cluster/launch_worker2_electricity_followup_v1.sh)
- worker3:
  - [launch_worker3_instruction_matrix_v1.sh](/home/namkyeong/ssmo/scripts/paper_rerun/cluster/launch_worker3_instruction_matrix_v1.sh)
