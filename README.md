# ssml

Selective-Supervised Mutual Operator Learning (SSMO/SSML) 실험 리포지토리입니다.

## Project Structure

- `src/`: 실험 핵심 코드
  - `src/tasks/`: task별 데이터 로더
  - `src/models/`: task별 모델 빌더
  - `src/utils/`: 공통 유틸(시드, 저장 등)
- `runners/`: CLI 실행 엔트리포인트
  - `run_classification.py`
  - `run_operator.py`
  - `run_time_series.py`
  - `run_segmentation.py`
  - `sweep.py` (task/dataset/model 매트릭스 일괄 실행)
- `models/`: 기존 코어 연산자 모델 구현(FNO/DeepONet/GNOT)
- `notebook/`: 기존 노트북 자산
- `Instruction.md`: 실험 요구사항
- `Batch_Experiment_Migration_Plan.md`: 마이그레이션 계획

## Task / Dataset / Model Matrix

| Task | Datasets | Models |
|---|---|---|
| Classification | `mnist`, `cifar10`, `cifar100` | `simple_cnn`, `simple_mlp`, `resnet18`, `vit_b16` |
| Operator learning | `burgers`, `darcy`, `navier_stokes` | `fno`, `deeponet`, `gnot`, `neuralop_fno`, `neuralop_tfno`, `neuralop_uno`(2D only) |
| Time-series forecasting | `etth1`, `etth2`, `ettm1`, `ettm2`, `electricity`, `weather`, `traffic`, `exchange_rate`, `illness` | `dlinear`, `transformer` |
| Semantic segmentation | `voc`, `cityscapes` | `unet`, `deeplabv3_resnet50` |

## Data Layout

- Classification: `torchvision` 데이터셋 자동 다운로드 가능 (`--download`)
- Operator:
  - Burgers: `data/burgers_data_R10.mat` 수동 배치
  - Darcy/NS: `--download`로 `neuralop` dataset 다운로드
  - `neuralop_*` 모델 사용 시 `conda activate ssml` 권장
- Time-series CSV 경로:
  - `data/time_series/ETT-small/ETTh1.csv`
  - `data/time_series/ETT-small/ETTh2.csv`
  - `data/time_series/ETT-small/ETTm1.csv`
  - `data/time_series/ETT-small/ETTm2.csv`
  - `data/time_series/electricity/electricity.csv`
  - `data/time_series/weather/weather.csv`
  - `data/time_series/traffic/traffic.csv`
  - `data/time_series/exchange_rate/exchange_rate.csv`
  - `data/time_series/illness/national_illness.csv`
- Segmentation:
  - VOC: `--download-voc`
  - Cityscapes: `data/cityscapes/leftImg8bit/...`, `data/cityscapes/gtFine/...`

## Run Commands

### Single run

```bash
python -m runners.run_classification --dataset cifar10 --model resnet18 --epochs 20 --batch-size 128 --seed 0 --device cuda --download
python -m runners.run_operator --dataset darcy --model neuralop_fno --epochs 20 --batch-size 16 --seed 0 --device cuda --download
python -m runners.run_time_series --dataset etth1 --model dlinear --seq-len 96 --pred-len 24 --epochs 20 --batch-size 64 --seed 0 --device cuda
python -m runners.run_segmentation --train-dataset voc --val-dataset voc --model unet --epochs 20 --batch-size 8 --seed 0 --height 512 --width 512 --device cuda --download-voc
```

### Single-model method comparison

`runners.run_*`는 동일한 모델/데이터셋에서 `--method`만 바꿔 실험합니다.

```bash
python -m runners.run_classification \
  --dataset cifar100 --model resnet18 --method naive \
  --epochs 20 --batch-size 64 --seed 0 --device cuda --download

python -m runners.run_classification \
  --dataset cifar100 --model resnet18 --method dml \
  --epochs 20 --batch-size 64 --seed 0 --device cuda --download \
  --classification-imitation-loss kl

python -m runners.run_classification \
  --dataset cifar100 --model resnet18 --method studygroup \
  --epochs 20 --batch-size 64 --seed 0 --device cuda --download \
  --lambda-imitation 1.0 --margin 0.0 --warmup-epochs 5
```

분류 imitation loss 타입:
- `--classification-imitation-loss kl` (기본)
- `--classification-imitation-loss js`
- `--classification-imitation-loss mse_logits`

공정 비교 규칙:
- `task`, `dataset`, `model`, `seed`, `epochs`, `batch-size`를 동일하게 유지
- method(`independent`/`naive`/`dml`/`studygroup`)만 변경
- simple 검증은 `--max-train-batches`, `--max-val-batches`로 배치 수 제한 가능

### Shell scripts (`simple` / `sweep`)

`scripts/`는 2단계로 사용합니다.
1. `simple`: 빠른 디버깅 기본값(단일 seed, 짧은 epoch, 배치 제한)
2. `sweep`: 디버깅 완료 후 큰 기본값으로 빡세게 튜닝

`simple`:
- `scripts/simple/run_simple_classification.sh`
- `scripts/simple/run_simple_operator.sh`
- `scripts/simple/run_simple_time_series.sh`
- `scripts/simple/run_simple_segmentation.sh`
- 기본 결과 경로: `results/run_simple/<task>/<dataset>/<model>/...`
- 실행 종료 시 `src.utils.visualization` 자동 호출
- 시각화 결과(`runs.csv`, `aggregate.csv`, `best_methods.csv`)는 기본적으로 `results/run_simple/`에 함께 저장

`hyperparameter_tuning` (`scripts/sweep`):
- `scripts/sweep/tune_classification.sh`
- `scripts/sweep/tune_operator.sh`
- `scripts/sweep/tune_time_series.sh`
- `scripts/sweep/tune_segmentation.sh`
- 기본 결과 경로: `results/sweep/<task>/hyper_parameter/...`
- 실행 종료 시 `src.utils.visualization` 자동 호출
- 시각화 결과(`runs.csv`, `aggregate.csv`, `best_methods.csv`)는 기본적으로 `results/sweep/<task>/hyper_parameter/`에 함께 저장

예시:
```bash
# 1) simple 디버깅
bash scripts/simple/run_simple_classification.sh

# 2) OK면 sweep 튜닝
bash scripts/sweep/tune_classification.sh
```

### Sweep

```bash
python -m runners.sweep \
  --tasks classification operator time_series segmentation \
  --seeds 0 1 2 \
  --epochs 20 \
  --device cuda \
  --include-neuralop-models \
  --download-classification \
  --download-operator \
  --download-voc \
  --continue-on-error
```

```bash
python -m runners.sweep --dry-run --max-runs 4
python -m runners.sweep --tasks segmentation --include-cityscapes
```

## Outputs

각 run 결과는 아래를 저장합니다.
- `summary.json`
- `curves.npz`
- single: `model.pt`

single 기본 저장 경로: `results/experiments/<task>/<dataset>/<model>/seed<seed>/...`
scripts/simple 시각화/집계 파일 저장: `results/run_simple/runs.csv`, `results/run_simple/aggregate.csv`, `results/run_simple/best_methods.csv`
scripts/sweep 시각화/집계 파일 저장: `results/sweep/<task>/hyper_parameter/runs.csv`, `results/sweep/<task>/hyper_parameter/aggregate.csv`, `results/sweep/<task>/hyper_parameter/best_methods.csv`

## Current Scope

현재는 아래 범위를 제공합니다.
- single-model supervised baseline (`runners.run_*`)

`Instruction.md`의 full mutual/SSMO 확장(`SSMO-soft` 등 추가 ablation)은 다음 구현 단계입니다.

## Practical Notes

- `--epochs 20`은 기본 simple 검증값입니다.
- 본 실험은 태스크별로 epoch를 더 길게 주는 것을 권장합니다.
  - 예시: classification 100, operator 150, time-series 60, segmentation 80

```bash
python -m runners.sweep \
  --tasks classification operator time_series segmentation \
  --seeds 0 1 2 3 4 \
  --classification-epochs 100 \
  --operator-epochs 150 \
  --time-series-epochs 60 \
  --segmentation-epochs 80 \
  --device cuda \
  --include-neuralop-models \
  --include-cityscapes \
  --download-operator \
  --download-voc \
  --continue-on-error
```
