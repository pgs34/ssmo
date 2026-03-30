# Results Summary

현재 워크스페이스 기준에서 **SSML이 실제로 제일 좋은 결과만** 상단에 남긴 요약이다.

## Confirmed SSML Wins

| Task | Dataset | Pair | Best SSML | Independent | DML | Ordering | Note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Operator | Darcy | `FNO x DeepONet` | `0.003148` | `0.003200` | `0.004758` | `SSML < independent < DML` | 현재 전체 실험 중 가장 깔끔한 승리 |
| Time-Series | Weather | `transformer x dlinear` | `0.261344` | `0.272783` | `0.281672` | `SSML < independent < DML` | `instruction_matrix_v1` 기준, activation 이후 실제 개선 |
| Time-Series | ETTh1 | `transformer x dlinear` | `0.321898` | `0.322104` | `0.323888` | `SSML < independent < DML` | best epoch가 매우 이른 early-best 패턴 |

## Needs Fix

| Task | Dataset | Pair | SSML | Best baseline | Problem |
| --- | --- | --- | ---: | ---: | --- |
| Time-Series | Electricity | `transformer x dlinear` | `0.168784` | `transformer independent = 0.165290`, `dlinear independent = 0.152387` | SSML이 DML보다만 좋고 strongest baseline은 못 넘음 |
| Classification | CIFAR-10 | `resnet18` | `0.856300` | `DML = 0.856933` | SSML이 근소하게 짐 |
| Classification | CIFAR-100 | `resnet34_gelu x resnet34_gelu` | `0.531233` | `independent = 0.528533` | independent는 넘지만 동일 조건 DML 우위가 아직 확인되지 않아 paper-grade 승리 아님 |

## Detail Tables

### Operator

| Item | Value |
| --- | --- |
| Best setting | `Darcy / FNO x DeepONet` |
| SSML | `0.003148` |
| FNO independent | `0.003200` |
| DML | `0.004758` |
| Ordering | `SSML < independent < DML` |
| SSML reference | [summary.json](/home/namkyeong/ssmo/results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_ssml_mse_seed0/summary.json) |
| Independent reference | [summary.json](/home/namkyeong/ssmo/results/operator_ssml_tuned_v1/operator/darcy/fno_independent_mse_seed0/summary.json) |
| DML reference | [summary.json](/home/namkyeong/ssmo/results/operator_ssml_tuned_v1/operator/darcy/fno__deeponet_dml_mse_seed0/summary.json) |

### Time-Series

| Dataset | Pair | SSML | Independent | DML | Extra Baseline | Ordering | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Weather | `transformer x dlinear` | `0.261344` | `0.272783` | `0.281672` | - | `SSML < independent < DML` | `instruction_matrix_v1` 기준 |
| ETTh1 | `transformer x dlinear` | `0.321898` | `0.322104` | `0.323888` | - | `SSML < independent < DML` | seed best epoch가 `3, 5, 5` |
| Electricity | `transformer x dlinear` | `0.168784` | `0.165290` | `0.175607` | `dlinear independent = 0.152387` | `independent < SSML < DML` | DML보다 낫지만 strongest single baseline은 못 넘음 |

| Dataset | SSML reference | Independent reference | DML reference |
| --- | --- | --- | --- |
| Weather | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed0/summary.json) |
| ETTh1 | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_rescue_v3/a0p5/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer_independent_huber_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_ssml_confirm_v2/time_series/etth1/transformer__dlinear_dml_huber_seed0/summary.json) |
| Electricity | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_ssml_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/electricity/transformer_independent_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/electricity/transformer__dlinear_dml_mse_seed0/summary.json) |

### Classification

| Dataset | Pair | Run | SSML | Independent | DML | Ordering | Note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| CIFAR-100 | `resnet34_gelu x resnet34_gelu` | `v17 alt / conf_pb25_aw5e4` | `0.531233` | `0.528533` | - | `independent < SSML` | clean classification에서 현재 메인 승리 신호 |
| CIFAR-10 | `resnet18` | `v4` | `0.856300` | `0.853600` | `0.856933` | `independent < SSML < DML` | SSML은 strong, DML이 근소 우위 |

| Dataset | SSML reference | Independent reference | DML reference |
| --- | --- | --- | --- |
| CIFAR-100 | [summary.json](/home/namkyeong/ssmo/results/classification_ssml_reweight_cifar100_v17_alt/conf_pb25_aw5e4/classification/cifar100/resnet34_gelu_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed0/summary.json) | - |
| CIFAR-10 | [classification_ssml_reweight_cifar10_v4](/home/namkyeong/ssmo/results/classification_ssml_reweight_cifar10_v4) | [classification_ssml_topk_sweep_cifar10_v3](/home/namkyeong/ssmo/results/classification_ssml_topk_sweep_cifar10_v3) | [classification_ssml_topk_sweep_cifar10_v3](/home/namkyeong/ssmo/results/classification_ssml_topk_sweep_cifar10_v3) |

## Carry Forward

| Task | Setting | Why keep it |
| --- | --- | --- |
| Operator | `Darcy / FNO x DeepONet / SSML` | 가장 깔끔한 승리 |
| Time-Series | `Weather / transformer x dlinear / instruction_matrix_v1` | 실제로 `SSML < independent < DML` |
| Time-Series | `ETTh1 / transformer x dlinear / rescue_v3 a0p5` | 근소 승리, `early_best_v1` 재실행은 기존 best 미갱신 |
| Classification | `CIFAR-100 / resnet34_gelu x resnet34_gelu / v17 alt conf_pb25_aw5e4` | 아직 미완이지만 clean classification에서 제일 유망한 축 |

## Drop / Ignore

| Setting | Reason |
| --- | --- |
| `weather rescue v17~v18` | 성능 실패 |
| `classification worker3 backup` | baseline 이하 |
| `classification clean hetero ResNet x ViT` | clean IID에서는 아직 약함 |
| `classification v17 main` | baseline 근처이거나 아래 |

## Latest Checked Runs

| Run | Status | Key result | Interpretation |
| --- | --- | --- | --- |
| `time_series_etth1_early_best_v1 / a025_fastdecay` | 완료 | mean `0.323241` | 기존 ETTh1 best `0.321898`보다 나쁨. `DML`보단 낫지만 `independent`는 못 넘음 |
| `time_series_etth1_early_best_v1 / a0p5_fastdecay` | 완료 | mean `0.323374` | 기존 ETTh1 best 미갱신 |
| `classification_cifar100_alt_focus_v1` | 실패 런 | summary `0개` | 상속된 `INIT_CHECKPOINT_TEMPLATE` 값이 깨져 시작 직후 `FileNotFoundError`로 종료 |
| `paper_gap_v1 / classification_homo_noise` | 부분 진행 | `independent seed0 = 0.526 @ epoch 48` | baseline 한 점만 나옴, SSML/DML 비교 아직 불가 |
| `paper_gap_v1 / classification_hetero_noise` | 미시작 또는 결과 없음 | summary `0개` | 아직 판단 불가 |

## Plot Files

| Task | Plot |
| --- | --- |
| Classification | [test_error_classification.png](/home/namkyeong/ssmo/test_error_classification.png) |
| Time-Series | [test_error_time_series.png](/home/namkyeong/ssmo/test_error_time_series.png) |
| Operator | [test_error_operator.png](/home/namkyeong/ssmo/test_error_operator.png) |
