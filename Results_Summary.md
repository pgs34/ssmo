# Results Summary

현재 워크스페이스 기준에서 **paper candidate win**과 **latest control / non-win**을 다시 정리한 요약이다.
공식 비교는 **같은 epoch budget으로 맞춘 completed 결과**를 우선 사용하고, `partial / exploratory / 다른 epoch` 결과는 참고용으로만 적는다.

## Confirmed SSML Wins

| Task | Dataset | Pair | Best SSML | Independent | DML | Ordering | Note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Operator | Burgers | `FNO x DeepONet` | `0.0000009653` | `0.0000009846` | `0.0000093153` | `SSML < independent < DML` | `operator_burgers_polish_aggressive_v4 / cos_relay_full_l0012_s20_70_40_sample_lr4e4` 3-seed mean best 기준. same-campaign `ctrl_cos = 0.0000009846`보다 약 `1.96%` 낮고, prior best `fair_v3 = 0.0000029749`보다도 크게 개선됐다. `DML`은 아직 old completed `followup_v1` reference다 |
| Operator | Darcy | `FNO x DeepONet` | `0.003148` | `0.003200` | `0.004758` | `SSML < independent < DML` | 현재 전체 실험 중 가장 깔끔한 승리 |
| Time-Series | Weather | `transformer x dlinear` | `0.261344` | `0.272783` | `0.281672` | `SSML < independent < DML` | `instruction_matrix_v1` 3-seed mean 기준, activation 이후 실제 개선 |
| Time-Series | Electricity | `transformer x dlinear` | `0.100142` | `0.152387` | `0.164034` | `SSML < independent < DML` | `corrective_v1 / corr_gate64_l15_sp5e4` 3-seed mean best checkpoint 기준. strongest independent는 `dlinear`이고, stable final 대체 설정은 아직 확정 못 했다 |
| Classification | CIFAR-10 | `resnet18 x resnet18` | `0.864233` | `0.809233` | `0.849667` | `independent < DML < SSML` | SSML은 `instruction_matrix_v1`, DML은 `classification_cifar10_homo_dml_long_v1` 3-seed mean 기준. long rerun 후에도 SSML이 여전히 앞섬 |

## Needs Fix

| Task | Dataset | Pair | SSML | Best baseline | Problem |
| --- | --- | --- | ---: | ---: | --- |
| Time-Series | ETTh1 | `transformer x dlinear` | `0.271835` | `dlinear independent = 0.272032`, `transformer independent best = 0.321354`, `transformer rerun = 0.323224`, `DML = 0.327667` | `teacher_ft_v1` best setting을 pair-deployed `best_branch`로 다시 읽으면 completed 3-seed mean이 `0.271835`까지 내려간다. 다만 reported branch는 세 seed 모두 frozen `dlinear` peer라서, 이 값은 student가 teacher를 넘은 결과라기보다 deployment-time peer selection 결과로 보는 편이 맞다 |
| Classification | CIFAR-100 | `resnet34_gelu x resnet34_gelu` | `0.536567` | `strict128 independent = 0.528533`, `strict128 DML = 0.545067`, matched `scaled_fair_bs3072 independent = 0.552333`, matched `scaled_fair_bs3072 DML = 0.554167` | 현재 공식 row는 계속 `strict128`만 쓴다. best는 여전히 `augfilter_seeded_v1 = 0.536567`이고, `teacher_ft_seeded_v4`, `disagreement_memory`, `scheduled/complement-lite`는 전부 이 strict best를 못 넘었다. latest `strict128_aggressive_v1` / `scaled_fair_aggressive_v1`의 저점들은 이후 확인된 `LR/weight_decay` wiring bug 영향으로 undertrained diagnostic으로 보는 편이 맞다 |

## Detail Tables

### Operator

| Item | Value |
| --- | --- |
| Best setting | `Burgers / FNO x DeepONet / polish_aggressive_v4 / cos_relay_full_l0012_s20_70_40_sample_lr4e4` |
| SSML (3-seed mean best) | `0.0000009653` |
| Independent, same-campaign control | `0.0000009846` |
| DML, latest completed reference | `0.0000093153` |
| Ordering | `SSML < independent < DML` |
| Note | same-campaign `ctrl_cos`와 차이는 작다 (`best mean` 기준 약 `1.96%`). 다만 prior best `fair_v3 = 0.0000029749`보다 크게 낮아졌고, matched `DML` rerun for `v4`는 아직 없다 |
| SSML reference | [summary.json](/home/namkyeong/ssmo/results/operator_burgers_polish_aggressive_v4/cos_relay_full_l0012_s20_70_40_sample_lr4e4/operator/burgers/fno__deeponet_ssml_mse_seed0/summary.json) |
| Independent reference | [summary.json](/home/namkyeong/ssmo/results/operator_burgers_polish_aggressive_v4/ctrl_cos_lr4e4_w10_min02_clip1/operator/burgers/fno_independent_mse_seed0/summary.json) |
| DML reference | [summary.json](/home/namkyeong/ssmo/results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed0/summary.json) |

### Time-Series

| Dataset | Pair | SSML | Independent | DML | Extra Baseline | Ordering | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Weather | `transformer x dlinear` | `0.261344` | `0.272783` | `0.281672` | - | `SSML < independent < DML` | `instruction_matrix_v1` 3-seed mean 기준 |
| ETTh1 | `transformer x dlinear` | `0.271835` | `0.321354` | `0.327667` | `dlinear independent = 0.272032`, matched `dlinear (seed 0/1/2) = 0.271835`, `transformer rerun = 0.323224`, `teacher_ft_v1 student/guided = 0.283504`, `teacher_ft_v1 next = 0.291310`, `snapshot_handoff_v1 best = 0.315550`, `corrective v1 exploratory best = 0.329268`, `complementary v4 best = 0.346186` | `SSML deployed ~= dlinear independent < transformer independent < DML` | official ETTh1 row는 이제 `teacher_ft_pairdeploy_reeval_20260405_v1`의 deployed `best_branch` 기준이다. same trained checkpoints를 pair output으로 다시 읽으면 mean best `0.271835`가 나오지만, reported branch가 세 seed 모두 frozen `dlinear` peer라서 student-side rescue `0.283504`와는 구분해서 해석해야 한다 |
| Electricity | `transformer x dlinear` | `0.100142` | `0.165290` | `0.164034` | `dlinear independent = 0.152387` | `SSML < dlinear independent < DML < transformer independent` | SSML은 `corrective_v1 / corr_gate64_l15_sp5e4` 3-seed mean best_val_mse 기준이다. 다만 curve 자체는 `epoch 11~13` 부근에서 best를 찍은 뒤 후반에 다시 상승하는 late-drift 패턴이 있어, final epoch보다 early best / best checkpoint 기준으로 읽는 편이 맞다. 후속 `corrective_v3 e12`는 early-stop을 걸면 mean final `0.101655`까지 안정화됐지만, same handoff를 full `60 epoch`로 다시 돌리면 seed0이 `epoch 31+` decay 구간에서 다시 폭주해 아직 공식 대체안으로는 못 쓴다 |

| Dataset | SSML reference | Independent reference | DML reference |
| --- | --- | --- | --- |
| Weather | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed0/summary.json) |
| ETTh1 | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json) | best transformer: [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed0/summary.json), rerun transformer: [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/transformer_independent_huber_seed0/summary.json), rerun dlinear: [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_independent_rerun_20260405_v1/time_series/etth1/dlinear_independent_huber_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed0/summary.json) |
| Electricity | [summary.json](/home/namkyeong/ssmo/results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer_independent_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed0/summary.json) |

### Classification

| Dataset | Pair | Run | SSML | Independent | DML | Ordering | Note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| CIFAR-100 strict128 | `resnet34_gelu x resnet34_gelu` | `augfilter_seeded_v1 best = 0.536567`, `alt_focus_v2 best = 0.532567`, `visual_complement_v3 full best = 0.528567` | `0.536567` | `0.528533` | `0.545067` | `independent < SSML < DML` | 공식 paper row는 계속 `strict128`만 사용한다. current best는 `pcu_pb20_thr38_gap20_augmin72_augmax90_agap03 = 0.536567`이고 matched `DML 0.545067`는 아직 못 넘음 |
| CIFAR-100 strict128 aggressive_v1 | `resnet34_gelu x resnet34_gelu` | `bestckpt_pool_v2 + SGD/cosine + strong aug + EMA` | `0.467467` | `0.465067` | `0.467200` | `independent < DML < SSML` | completed 3-seed 기준 best family는 `uh_sched_mem`. 다만 이후 확인 결과 `run_core_classification.sh`가 `--lr`, `--weight-decay`를 누락해 이 family 전체가 default `lr=1e-3`, `weight_decay=1e-4`로 undertrained 상태였다. 현재 수치는 diagnostic으로만 본다 |
| CIFAR-100 strict128 followup_v1 | `resnet34_gelu x resnet34_gelu` | `bestckpt_pool_v3 + corrected strict128 rerun (partial 2-seed)` | `0.668900` | `0.669200` | `0.670400` | `SSML < independent < DML` | corrected `pool_v3` 위에서 현재 best completed SSML은 `uh_sched_mem_v2`다. 이전 aggressive_v1의 LR/WD wiring bug는 벗어났고 수치도 크게 회복됐지만, 아직 `2/3 seed`만 완료됐고 현재 partial mean에서는 `DML`과 `independent`를 못 넘는다 |
| CIFAR-100 scaled_fair aggressive_v1 | `resnet34_gelu x resnet34_gelu` | `matched 3072/1536 controls + SSML rerun` | `0.458533` | `0.458200` | `0.458633` | `independent < SSML < DML` | completed family 기준 best SSML은 `oxtra38_trainer_v2` (1536 batch). 다만 strict128 aggressive_v1과 동일한 `LR/weight_decay` wiring bug가 있었으므로 이 값들도 diagnostic으로만 유지한다 |
| CIFAR-100 scaled_fair | `resnet34_gelu x resnet34_gelu` | `overbatch exploratory spikes = 0.5477~0.5558 (single-seed)` | pending | pending | pending | pending | `batch_size=3072/1536` mixed exploratory에서 `0.55+` 신호가 반복해서 보였지만, 아직 matched `independent / DML / SSML` 3-seed protocol이 없어 공식 승패 판단에는 쓰지 않는다. 새 `scaled_fair` track에서 같은 batch/epoch로 다시 맞춘다 |
| CIFAR-10 | `resnet18 x resnet18` | `SSML/independent = instruction_matrix_v1`, `DML = homo_dml_long_v1` | `0.864233` | `0.809233` | `0.849667` | `independent < DML < SSML` | 중간에 끊기던 DML curve를 `100 epoch`로 다시 돌려도 SSML이 앞섬 |

| Dataset | SSML reference | Independent reference | DML reference |
| --- | --- | --- | --- |
| CIFAR-100 strict128 | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed0/summary.json) |
| CIFAR-100 strict128 aggressive_v1 | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_strict128_aggressive_v1/uh_sched_mem/classification/cifar100/resnet34_gelu_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_strict128_aggressive_v1/strict128_independent_v2/classification/cifar100/resnet34_gelu_independent_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_strict128_aggressive_v1/strict128_dml_v2/classification/cifar100/resnet34_gelu_dml_kl_seed0/summary.json) |
| CIFAR-100 strict128 followup_v1 (partial 2-seed) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_strict128_followup_v1/uh_sched_mem_v2/classification/cifar100/resnet34_gelu_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_strict128_followup_v1/strict128_independent_v3/classification/cifar100/resnet34_gelu_independent_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_strict128_followup_v1/strict128_dml_v3/classification/cifar100/resnet34_gelu_dml_kl_seed0/summary.json) |
| CIFAR-100 scaled_fair aggressive_v1 | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_scaled_fair_aggressive_v1/oxtra38_trainer_v2/classification/cifar100/resnet34_gelu_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_scaled_fair_aggressive_v1/scaled1536_independent_v2/classification/cifar100/resnet34_gelu_independent_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_scaled_fair_aggressive_v1/scaled1536_dml_v2/classification/cifar100/resnet34_gelu_dml_kl_seed0/summary.json) |
| CIFAR-100 scaled_fair | pending | pending | pending |
| CIFAR-10 | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed0/summary.json) |

## Carry Forward

| Task | Setting | Why keep it |
| --- | --- | --- |
| Operator | `Burgers / FNO x DeepONet / polish_aggressive_v4 / cos_relay_full_l0012_s20_70_40_sample_lr4e4` | 현재 strongest Burgers 결과. same-campaign `ctrl_cos = 0.0000009846`보다도 소폭 낮고, prior best `fair_v3 = 0.0000029749`를 크게 갱신했다 |
| Operator | `Darcy / FNO x DeepONet / SSML` | 가장 깔끔한 초기 operator 승리 |
| Time-Series | `Weather / transformer x dlinear / instruction_matrix_v1` | 실제로 `SSML < independent < DML` |
| Time-Series | `Electricity / transformer x dlinear / corrective_v1 corr_gate64_l15_sp5e4` | strongest single baseline인 `dlinear independent`까지 넘긴 가장 강한 신호는 여전히 이 설정이다. 다만 official reading은 final이 아니라 best checkpoint 기준으로 붙인다 |
| Classification | `CIFAR-10 / resnet18 x resnet18 / instruction_matrix_v1 classification_homo` | clean 3-seed mean에서 `DML`까지 넘는 승리 |

## Drop / Ignore

| Setting | Reason |
| --- | --- |
| `ETTh1 / transformer x dlinear / rescue_v3 a0p5 carry-forward` | early-stop `0~5` seed 재검증에서 `independent`가 best라 carry-forward claim 유지 불가 |
| `ETTh1 / corrective_v1`을 곧바로 paper win으로 승격 | best 3-seed는 `0.329268`까지 왔지만 `12/12` run이 모두 `best_epoch=1`이라 구조적 cold-start 의심이 남음 |
| `ETTh1 / complementary_v4 full sweep` | cold-start는 줄어서 `best_epoch`가 `8~10`으로 이동했지만 best full 3-seed가 `0.346186`이라 matched baseline보다 한참 아래 |
| `ETTh1 / peer_advantage_seeded_v1` | full 3-seed best가 `advq70_tail30_rg0_min5e3_f8 = 0.369487`이고 나머지도 `0.374753~0.383826`으로 더 나빴다. `active_imitation_ratio`가 `1e-5~1e-3` 수준이라 selection이 너무 희박했고 전부 early stop으로 끝남 |
| `ETTh1 / dense_regime_horizon_seeded_v3` | full 3-seed best가 `dhr_full_reg55_ad42_ws1_exp1_raw = 0.374205`이고 나머지도 `0.374504~0.386370`이라 기존 `complementary_v4` best `0.346186`보다 더 나쁘다. dense / regime / horizon focus를 넓혀도 아직 baseline과 거리가 큼 |
| `ETTh1 / dense_regime_horizon_seeded_v4` | full 3-seed best가 `dhr_dense_full_reg40_ad95_ws11_exp13_raw = 0.365420`로 `v3`보단 나아졌지만, 여전히 `complementary_v4` best `0.346186`와 matched baseline `0.321354`보다 많이 높다 |
| `Electricity / corrective_v1`을 final epoch 공식 claim으로 유지 | best checkpoint win 자체는 맞았지만 mean final이 `0.157514`까지 다시 올라가 late-drift가 컸다 |
| `Electricity / corrective_v3 e12`를 early-stop final 그대로 공식 claim으로 승격 | `patience=8`을 둔 exploratory run에선 mean final `0.101655`로 안정화됐지만, same handoff를 early-stop 없이 full `60 epoch`로 다시 돌리면 seed0가 `epoch 31+`에서 급격히 폭주했다. fixed-budget 공식 대체안으로는 아직 이르다 |
| `weather rescue v17~v18` | 성능 실패 |
| `CIFAR-100 / clean homogeneous current paper claim` | matched `DML` reference `0.545067`가 현재 SSML `0.536567`보다 높음 |
| `CIFAR-100 / visual_complement_v3 full sweep` | best full 3-seed가 `0.528567`로 prior SSML best `0.532567`보다 낮음. `overbatch` single-seed spike는 `scaled_fair`로 따로 재검증하고, strict128 공식 row와는 섞지 않는다 |
| `classification worker3 backup` | baseline 이하 |
| `classification clean hetero ResNet x ViT` | clean IID에서는 아직 약함 |
| `classification v17 main` | baseline 근처이거나 아래 |

## Latest Checked Runs

| Run | Status | Key result | Interpretation |
| --- | --- | --- | --- |
| `instruction_matrix_v1 / worker2 time_series` | 완료 | `36 summaries`, Weather mean `0.261344`, ETTh1 mean `0.298151`, Electricity mean `0.168784` | Weather/Electricity 값은 현재 표에 반영된 3-seed mean과 일치. ETTh1는 기존 carry-forward best는 아님 |
| `time_series_electricity_followup_v1 / best_known` | 완료 | `12 summaries`, transformer independent mean `0.165290`, DML mean `0.164034`, SSML mean `0.165709`, dlinear independent mean `0.152387` | transformer/DML/dlinear baseline reference로 유지. SSML 자체는 이제 corrective run으로 대체 |
| `time_series_electricity_corrective_v1` | 완료 | best `corr_gate64_l15_sp5e4` mean best_val_mse `0.100142`, next `corr_gate64_do10_l20_sp10e4` = `0.100196` | 가장 강한 best checkpoint 신호는 여전히 여기서 나왔다. 다만 final 기준으로는 late-drift가 커서 official reading은 계속 best checkpoint로 제한해야 한다 |
| `time_series_electricity_corrective_v1 / curve audit` | 완료 | seed0 기준 `best_val_mse = 0.099632 @ epoch 13`, final `epoch 60 = 0.161586`; seed1/2도 동일하게 후반 재상승 | 플롯에서 SSML이 다시 올라가는 건 버그가 아니라 실제 late-drift다. `first_active_epoch=1`, `active_imitation_ratio ~ 0.90 -> 0.99`, `mean_imitation_weight ~ 0.58 -> 0.92`로 초반부터 guidance가 너무 넓게 켜져 후반 과교정/overfit이 생긴 쪽으로 해석하는 게 맞음 |
| `time_series_electricity_corrective_v3` | 완료 | `corr_handoff_e12_l15_sp5e4`: mean best `0.100127`, mean final `0.101655`; next `corr_handoff_e14_l15_sp5e4`: mean final `0.102281` | `epoch 12` 이후 backbone freeze + correction gate만 업데이트하는 late handoff는 early-stop을 걸면 curve를 안정화할 수 있었다. 다만 이건 exploratory stability check로 보고, fixed-budget 공식 row로는 아직 승격하지 않는다 |
| `time_series_electricity_corrective_v3_full60` | 부분 완료 | `corr_handoff_e12_l15_sp5e4_full60` seed0: `best_val_mse = 0.099273 @ epoch 11`, `epoch 30 = 0.101029`, 이후 `epoch 31 = 0.100805 -> epoch 35 = 0.163942 -> epoch 40 = 0.552666 -> epoch 50 = 2.470134` | same handoff를 early-stop 없이 full `60 epoch`로 밀면 decay가 시작되는 `epoch 31+`에서 다시 크게 무너진다. 따라서 early-stop run을 official final replacement로 쓰면 안 된다 |
| `time_series_electricity_corrective_v2` | 완료 | best `late_b15_q75_m15_do10` mean best_val_mse `0.161111`, next `late_b10_q80_m20 = 0.165278`, `late_b08_q85_m25_sp15e4 = 0.167274` | `later guidance + peer-advantage focus + budget cap + early stop`로 drift는 크게 줄었다. 예를 들어 best case는 mean final `0.165618`로 old v1 final `0.157514` 수준의 late blow-up은 막았지만, 정작 best mean 자체가 baseline 근처로 올라가 기존 Electricity win `0.100142`를 재현하지 못했다. 현재 설정은 너무 보수적이라 `active_imitation_ratio`가 대체로 `0.003~0.014` 수준에 머물렀다 |
| `time_series_electricity_handoff_router_seeded_v1` | 완료 | best full 3-seed `elec_handoff_q70_b20 = 0.160030`, next `elec_handoff_q75_b18 = 0.161869`, `elec_handoff_q80_b15 = 0.164009` | `handoff + router + trend-only`로 final drift는 꽤 안정적이지만, 수치 자체는 기존 `corrective_v1 win 0.100142`보다 훨씬 높고 strongest baseline `dlinear independent 0.152387`도 못 넘었다 |
| `instruction_matrix_v1 / worker1 classification_homo` | 완료 | `CIFAR-10 / resnet18 mean: independent 0.809233, DML 0.836867, SSML 0.864233` | `Needs Fix`에 있던 CIFAR-10 clean row는 이제 제거 가능 |
| `time_series_etth1_early_stop_v1 / seeds 0 1 2` | 완료 | mean best `independent 0.322104`, `DML 0.325724`, `SSML 0.325958` | curve tail 제거는 성공했지만 `independent`가 여전히 best |
| `time_series_etth1_early_stop_v1_rerun3_seeds345` | 완료 | mean best `independent 0.320604`, `DML 0.333756`, `SSML 0.334418` | 추가 3-seed에서도 같은 결론 재현. ETTh1는 paper win 후보에서 제외하는 쪽이 맞음 |
| `time_series_etth1_corrective_v1` | 완료 | best `corr_lr1e4_l15_sp5e4_h64` mean best_val_mse `0.329268`; `12/12` runs 모두 `best_epoch=1` | raw corrective 방향 자체는 약간의 개선 신호가 있지만, gate cold-start 없이 그대로 carry-forward 하긴 어려움 |
| `time_series_etth1_complementary_v4` | 완료 | best full 3-seed `trqreg_biasm25_k17_lr6e5_l12_sp4e4_h96` mean `0.346186`, next `trcomp_gate8_biasm3_k13_lr1e4_l15_sp5e4_h64` = `0.346378` | `best_epoch`는 전부 `8~10`으로 옮겨가 cold-start는 완화됐지만, 수치 자체는 기존 ETTh1 best보다 더 나쁨 |
| `time_series_etth1_peer_advantage_seeded_v1` | 완료 | best full 3-seed `advq70_tail30_rg0_min5e3_f8` mean `0.369487`, next `advq75_tail30_rg0_min0_f8` = `0.374753` | `active_imitation_ratio`가 `1e-5~1e-3`로 너무 작아 correction selection이 거의 안 열렸다. ETTh1 후속은 sparse peer-advantage보다 `reweight_only`나 denser complementary 쪽이 더 유망 |
| `time_series_etth1_dense_regime_horizon_seeded_v3` | 완료 | best full 3-seed `dhr_full_reg55_ad42_ws1_exp1_raw` mean `0.374205`, next `dhr_full_reg50_ad35_ws3_exp9_raw` = `0.374504` | dense corrective / regime-aware / horizon-aware로 selection을 넓혀도 ETTh1 값은 더 악화됐다. 현재 기준으로는 `complementary_followup_seeded_v2`보다도 아래 |
| `time_series_etth1_dense_regime_horizon_seeded_v4` | 완료 | best full 3-seed `dhr_dense_full_reg40_ad95_ws11_exp13_raw` mean `0.365420`, next `dhr_relax_tail20_reg35_ad85_ws9_exp11_raw` = `0.372873` | `v3`보다는 내려왔지만 여전히 `complementary_followup_seeded_v2 = 0.346186`과 matched baseline `0.321354`보다 한참 높다. dense corrective만으로는 ETTh1를 못 살림 |
| `classification_cifar100_alt_focus_v2 / worker3_queue` | 완료 | `15 summaries`, best `conf_pb26_thr33_aw4e4` mean `0.532567` | `v17` best `0.531233`보다 소폭 개선. 다만 matched DML reference 완료 후에는 paper lead가 아님 |
| `classification_cifar100_augfilter_seeded_v1` | 완료 | best full 3-seed `pcu_pb20_thr38_gap20_augmin72_augmax90_agap03` mean `0.536567`, next `pcu_pb20_thr40_gap20_augmin75_augmax90_agap04` = `0.535433` | prior SSML best `0.532567`를 넘기면서 가장 나은 clean homogeneous SSML로 갱신됐다. 다만 matched DML `0.545067`는 아직 못 넘음 |
| `classification_cifar100_augfilter_seeded_v2` | 부분 완료 | completed 3-seed best `uhconf_pb24_thr34_gap18_augmin74_augmax92_agap02 = 0.531000`, next `pcu_pb20_thr39_gap18_augmin74_augmax90_agap03 = 0.529700` | 3-seed 완료된 세팅들 기준으로는 전부 `seeded_v1 best 0.536567` 아래다. 한 세팅은 아직 `2 seeds`만 완료 |
| `classification_cifar100_augfilter_seeded_v3` | 완료 | best full 3-seed `uhconf_pb18_thr37_gap24_augmin78_augmax93_agap03 = 0.536100`, next `pcu_pb16_thr42_gap22_augmin76_augmax92_agap04 = 0.534733` | node0 follow-up은 전부 `3 seeds` 완료됐다. closest case가 나오긴 했지만 현재 best `seeded_v1 0.536567`는 못 넘음 |
| `classification_cifar100_visual_complement_v3` | 완료 | best full 3-seed `viscomp40_thr45_gap1_spmax90_aw3e4` mean `0.528567`, next `viscomp25_thr60_gap4_spmax70_aw2e4` = `0.527933` | aug-consistency / visual complement full sweep는 prior SSML best `0.532567`를 못 넘음 |
| `classification_cifar100_visual_complement_v3 / overbatch exploratory` | 부분 완료 | best single-seed `visfill_gpu0a_bs3072_s0 = 0.551200`, nearby spikes `0.5490~0.5510` | single-seed로는 `DML 0.545067` 위 신호가 있지만 batch-size / seed confound가 커서 아직 carry-forward 불가 |
| `classification_cifar100_homo_dml_reference_v1` | 완료 | best `dml_l4e2_t6` mean `0.545067`, next `dml_l4e2_t4` = `0.544967` | matched clean homogeneous DML reference가 현재 SSML `0.536567`보다 확실히 높음 |
| `classification_cifar100_strict128_aggressive_v1` | 부분 완료 | completed 3-seed controls는 `independent = 0.465067`, `DML = 0.467200`; completed SSML best는 `uh_sched_mem = 0.467467`, next `pcu_ramp_wide = 0.466867`, `pcu_dual55 = 0.466700` | 이후 확인 결과 `run_core_classification.sh`가 `--lr`, `--weight-decay`를 실제 runner에 넘기지 않아 전부 default `lr=1e-3`, `weight_decay=1e-4`로 돌았다. 따라서 이 family는 stack regression이라기보다 wiring bug로 인한 undertraining으로 해석하는 것이 맞다 |
| `classification_cifar100_strict128_followup_v1` | 진행 중 | partial `2/3 seed`: `pool_v3 independent = 0.652850`, strict matched `independent = 0.669200`, `DML = 0.670400`, best SSML `uh_sched_mem_v2 = 0.668900`; exploratory `uh_sched_mem_aug72`는 `epoch 73` 기준 seed0 `0.6501`, seed1 `0.6479` | corrected pool과 strict rerun은 undertrained aggressive_v1보다 크게 회복됐다. 다만 현재 completed `2-seed` 기준으로는 아직 `SSML`이 `DML`과 `independent`를 못 넘고 있고, `aug72` 변형도 초반 신호는 `uh_sched_mem_v2`보다 약하다. seed2와 남은 variants를 더 봐야 한다 |
| `classification_cifar100_scaled_fair_aggressive_v1` | 부분 완료 | `3072`: `independent = 0.458300`, `DML = 0.458533`, `oxtra42 = 0.458433`; `1536`: `independent = 0.458200`, `DML = 0.458633`, `oxtra38 = 0.458533` | strict128 aggressive_v1과 동일한 `LR/weight_decay` wiring bug 영향이다. completed 숫자는 남겨두되, 정상 matched rerun 결과로 쓰면 안 된다 |
| `classification_cifar100_scaled_fair_cifarstem_v1` | 진행 중 | completed는 아직 `independent seed2 = 0.583800`만 있다 | `resnet34_cifar_gelu` 쪽 ceiling은 높아 보이지만 3-seed family가 안 끝나서 아직 판단 보류다 |
| `time_series_etth1_all_methods_long_v3` | 완료 | `6 seeds / 80 epochs / no early-stop`: `dlinear independent = 0.272032`, `transformer independent = 0.321354`, `SSML = 0.325609`, `DML = 0.327667` | ETTh1가 짧게 끝나던 문제를 길게 다시 돌려도 결론은 안 바뀌었다. SSML/DML 모두 transformer baseline도 못 넘고, strongest single baseline인 dlinear와는 격차가 큼 |
| `time_series_etth1_independent_rerun_20260405_v1` | 완료 | `6 seeds / 80 epochs / independent-only`: `dlinear independent = 0.272032`, `transformer independent = 0.323224` | strong `dlinear` baseline은 사실상 그대로 재현됐다. `transformer` rerun은 prior `0.321354`보다 약간 높아졌지만, ETTh1의 strongest single baseline이 계속 `dlinear`라는 결론은 변하지 않는다 |
| `time_series_etth1_teacher_ft_v1` | 완료 | best full 3-seed `tft_tail10_reg15_l010_lr2e4 = 0.283504`, next `tft_tail20_reg20_l015_lr3e4 = 0.291310` | 이번 follow-up 중 가장 의미 있는 개선이다. frozen `dlinear` teacher + transformer checkpoint fine-tuning으로 SSML이 transformer baseline `0.321354`와 DML `0.327667`는 확실히 넘었다. 다만 strongest `dlinear independent 0.272032`는 아직 못 넘음 |
| `time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1` | 완료 | deployed `best_branch` full 3-seed `tft_tail10_reg15_l010_lr2e4 = 0.271835` | same trained checkpoints를 pair-deployed output으로 다시 읽은 결과다. completed 3-seed mean이 matched `dlinear independent seed 0/1/2 = 0.271835`와 정확히 같고, reported branch도 세 seed 모두 `peer`라서 ETTh1에서는 현재 “좋은 놈으로 업데이트”가 사실상 frozen `dlinear` peer selection으로 구현됐다고 보는 편이 정확하다 |
| `time_series_etth1_teacher_ft_v2` | 완료 | best completed 3-seed `trs13_res04_lr2e4 = 0.296472`, next `trs14_res02_lr2e4 = 0.302700`, `trs18_res01_lr3e4 = 0.308239` | `trend/residual scale` 분리는 방향성은 있었지만, 여전히 correction density가 너무 높았다. best epoch는 대체로 `20~26`으로 이동했지만 `active_imitation_ratio`가 `0.47~0.96` 수준으로 넓게 켜져 late drift가 다시 강하게 남았다 |
| `time_series_etth1_teacher_ft_v3` | 완료 | best completed 3-seed `b12_q80_m20_t18_r00_lr3e4 = 0.316102`, next `b10_q80_m20_t14_r00_lr2e4 = 0.318657` | `budget cap + stronger teacher-advantage focus`는 drift는 줄였다. best epoch가 `22~24`로 유지되고 last epoch에선 gate가 거의 꺼지지만, 너무 보수적으로 바뀌면서 값이 transformer independent 근처까지 다시 올라가 rescue 효과 자체를 잃었다 |
| `time_series_etth1_handoff_router_seeded_v1` | 완료 | best full 3-seed `hr_long_q85_b08_h34 = 0.319122`, next `hr_bal_q80_b12_h28 = 0.319461`, `hr_mid_q82_b10_h30 = 0.319697` | `handoff + horizon routing + trend-only`는 late drift를 줄이고 final을 `0.322104` 근처로 안정화했다. 다만 기존 best rescue `teacher_ft_v1 = 0.283504`보다 훨씬 나쁘고 strongest `dlinear independent 0.272032`도 못 넘었다 |
| `classification_cifar10_homo_dml_long_v1` | 완료 | `resnet18:resnet18 DML / 3 seeds / 100 epochs` mean `0.849667` | top-level classification plot에서 중간에 끊겨 보이던 CIFAR-10 DML curve를 길게 다시 채웠다. 기존 `0.836867`보다 올라왔지만, SSML `0.864233`는 여전히 못 넘음 |
| `operator_burgers_followup_v1` | 부분 완료 | first completed case `burgers_l005_m0_w12_d60_120_ow1`: `FNO independent = 4.2316e-06`, `SSML = 4.4312e-06`, `DML = 9.3153e-06` | low-lambda late-guidance로 gap을 줄이긴 했지만, strongest `FNO independent`는 아직 못 넘었다. 즉 follow-up 방향은 더 봐야 하지만 현 시점 결과는 non-win |
| `operator_burgers_polish_aggressive_v4` | 완료 | best full 3-seed `cos_relay_full_l0012_s20_70_40_sample_lr4e4 = 9.6532e-07`, next `cos_relay_coarse_l0008_s20_70_50_element_lr4e4 = 9.7304e-07`, `cos_full_l0010_w20_d90_150_sample_lr4e4 = 9.7025e-07`, same-campaign `ctrl_cos_lr4e4_w10_min02_clip1 = 9.8461e-07` | new strongest Burgers result다. 다만 `relay_full`이 이기긴 해도 same-campaign control이 이미 거의 같은 수준이라, 이번 점프의 대부분은 새 cosine polish regime에서 나왔고 relay 자체의 추가 이득은 작다. matched `DML` rerun for this regime은 아직 없다 |
| `classification_cifar100_teacher_ft_seeded_v4` | 완료 | best full 3-seed `uh_late_pb16_thr38_gap22_aug78_94_ag03 = 0.532800`, next `pcu_late_pb14_thr42_gap26_aug80_92_ag04 = 0.531833` | checkpoint warm-start late teacher fine-tuning은 안전하긴 했지만, 기존 `seeded_v1 best 0.536567`와 matched `DML 0.545067`는 못 넘었다 |
| `classification_cifar100_disagreement_memory_seeded_v1` | 진행 중 | completed full 3-seed best `pcu_mem_df35_m90_x10 = 0.532100`, next `uh_mem_df35_m90_x10 = 0.530767`; `pcu_mem_df45_m95_x15`, `uh_mem_df50_m95_x20`는 아직 실행 중 | 현재는 `mixed-batch exploratory`로만 본다. `disagreement preserve + class-deficit memory`는 strict official best `0.536567`보다 낮고, 공식 row 교체 후보는 아니다 |
| `classification_cifar100_augfilter_complement_lite_seeded_v1` | 부분 완료 | completed full 3-seed best `pcu_lite_df10_m80_x05 = 0.531800`, next `uh_lite_df15_m85_x08 = 0.531200`; `pcu_lite_df15_m85_x08`는 아직 `seed0 best = 0.537700`만 존재 | 이것도 지금은 `mixed-batch exploratory`로만 유지한다. 기존 augfilter를 최대한 보존했지만 completed 3-seed 기준으로는 아직 strict best `0.536567`를 못 넘었다 |
| `classification_cifar100_scheduled_complement_v1` | 부분 완료 | completed full 3-seed best `uh_sched_df10_x05_r30_60 = 0.535133`, next `pcu_sched_df10_x05_r30_60 = 0.534633`; `pcu_sched_df15_x08_r35_65`는 진행 중 | disagreement floor와 complement ramp를 epoch schedule로 다시 걸어도 현재 completed 결과는 prior SSML best `0.536567`와 matched `DML 0.545067`를 못 넘는다. strict 신규 루프는 이제 `dual_peer_consensus_strict128_v2`로 좁힌다 |
| `classification_cifar100_scaled_fair_v2` | 완료 | matched `scaled_fair_bs3072` 기준 `SSML best = 0.550067` (`oxtra42_thr42_gap1_pc18_aug125`), next `0.549400`, `0.548067`; same-protocol `independent = 0.552333`, `DML = 0.554167` | overbatch warm-start 자체는 강했다. 하지만 matched `independent / DML / SSML` 3-seed를 같은 `batch_size=3072`, `100 epoch`, `best_ckpt_pool` init으로 다시 맞춰보니 SSML이 둘 다 못 넘었다. 즉 `0.55+` single-seed spike는 재현됐지만 clean homogeneous CIFAR-100의 새로운 SSML official win으로 해석하긴 어렵다 |
| `operator_burgers_student_lift_v2` | 완료 | control `ctrl_ft_lr4e4 = 0.005031`, best SSML `elem_ft_huber_l002_w30_d80_140_lr3e4 = 0.004658`, original `DeepONet independent = 0.012436` | weak `DeepONet`는 크게 개선됐다. 다만 이 실험은 strong `FNO`를 넘기는 그림이 아니라 student rescue에 가깝고, strongest `FNO independent 4.2316e-06`와는 여전히 큰 차이가 난다 |
| `operator_burgers_relay_v1` | 완료 | `140 epoch` relay best는 `burgers_relay_hotspot = 0.004683`, next `burgers_relay_coarse = 0.004759` | relay 자체는 `student_lift_v2 best 0.004658`보다도 살짝 약했다. 그리고 이 `140 epoch` 결과를 `180 epoch`의 `FNO independent 4.2316e-06`와 직접 비교하면 안 된다. Burgers의 공식 비교는 계속 matched `180 epoch`인 `followup_v1` row를 기준으로 본다 |
| `operator_burgers_fno_polish_fair_v3` | 완료 | best full 3-seed `fno_polish_ultra_full_l0012_w30_d110_180 = 2.9749e-06`, next `fno_polish_ultra_hotspot_l0008_w48_d130_180 = 3.0979e-06`, `fno_polish_ultra_coarse_l0010_w36_d120_180 = 3.4136e-06` | 당시에는 strongest Burgers win이었지만, 이제는 `polish_aggressive_v4` best `9.6532e-07`에 의해 명확히 superseded 됐다 |
| `time_series_etth1_teacher_ft_snapshot_handoff_v1` | 완료 | best full 3-seed `tft_h18_t10_l008_lr15e4 = 0.315550`, next `tft_h22_t12_l010_lr2e4 = 0.316346`, `tft_h26_t14_l012_lr2e4 = 0.317144` | snapshot anchor + handoff 조합은 실행 완료됐지만 `active_imitation_ratio = 0.0`으로 사실상 correction이 안 켜진 채 transformer checkpoint fine-tune에 가까웠다. 결과도 prior rescue `teacher_ft_v1 = 0.283504`와 strongest `dlinear independent 0.272032`를 모두 못 넘었다 |
| `operator_darcy_student_lift_v1` | 완료 | control `ctrl_ft_lr3e4 = 0.013669`, best SSML `sample_ft_l010_w5_d20_70_lr3e4 = 0.013768`, next `elem_ft_l010_w5_d20_70_lr3e4 = 0.013860`, original `DeepONet independent = 0.018561` | weak `DeepONet` 자체는 꽤 올라왔다. 다만 peer-guided SSML이 같은 init checkpoint에서의 no-peer fine-tuning control을 넘지는 못했다. 즉 현재 Darcy의 주된 승리는 여전히 `strong FNO polishing` 쪽이고, `weak-model lift` 메시지는 추가 개선이 필요하다 |
| `operator_darcy_student_lift_v2` | 완료 | control `ctrl_ft_lr2e4 = 0.013792`, best SSML `elem_ft_l002_w20_d50_95_lr2e4 = 0.013763`, next `sample_ft_l003_w15_d45_90_lr2e4 = 0.013821`, original `DeepONet independent = 0.018561` | `v2`는 same-campaign control은 소폭 넘겼다. 그래도 best no-peer control across runs `0.013669`는 아직 못 넘어서, weak-model lift를 논문 메시지로 쓰기엔 조금 더 필요하다 |
| `operator_darcy_relay_v1` | 완료 | best full 3-seed `darcy_relay_coarse = 0.013723`, next `darcy_relay_hotspot = 0.014294` | `relay`가 현재 weak-model lift 쪽 최고치는 만들었다. 그래도 best control `0.013669`는 못 넘었고, 깔끔한 win까지는 아직 한 끗 모자란다 |
| `time_series_etth1_teacher_win_reweight_fair_v1` | 완료 | best full 3-seed `twr_q80_top20_a35_h26_r70_lr25e4 = 0.311170`, next `twr_q85_top18_a40_h22_r75_lr2e4 = 0.313425`, `twr_q90_top15_a45_h18_r80_lr15e4 = 0.314175` | output imitation을 버리고 teacher-win horizon에만 supervised weight를 더 싣는 방향은 끝까지 strongest `dlinear independent 0.272032`를 못 따라갔다. best가 prior rescue `teacher_ft_v1 0.283504`도 넘지 못해서, 이 family는 현재 ETTh1에선 명확한 win이 아니다 |
| `time_series_etth1_teacher_win_reweight_fair_rerun_20260405_v1` | 완료 | best full 3-seed `twr_q80_top20_a35_h26_r70_lr25e4 = 0.313311`, next `twr_q90_top15_a45_h18_r80_lr15e4 = 0.314986`, `twr_q85_top18_a40_h22_r75_lr2e4 = 0.315204` | rerun도 결론을 바꾸지 못했다. 완료된 3개 setting이 전부 original fair_v1 best `0.311170`, prior rescue `teacher_ft_v1 0.283504`, strongest `dlinear independent 0.272032`보다 높아서, `reweight_only + handoff + horizon router` 조합은 robust win이 아니다 |
| `paper_gap_v1 / classification_homo_noise` | 완료 | `independent mean = 0.462900`, `DML mean = 0.491633`, `SSML mean = 0.405367` | `noise=0.2` reference는 확보됐지만 clean homogeneous CIFAR-100 비교 기준으로 직접 쓰면 안 됨 |
| `paper_gap_v1 / classification_hetero_noise` | 진행 중 | summary `0개`, `independent seed0` epoch log만 존재 | 미시작은 아님. 아직 첫 clean baseline run도 summary까지는 못 감 |

## Plot Preview

### Classification

![Classification validation error](./test_error_classification.png)

Current CIFAR-100 preview combo defaults to the `strict128_aggressive_v1` diagnostic curves (`Independent = strict128_independent_v2`, `DML = strict128_dml_v2`, `SSML = uh_sched_mem`) until the corrected `strict128_followup_v1` track produces a completed `3-seed` preview bundle. Once `results/logs/classification_cifar100_strict128_followup_v1/node0/narrow_exploit_report.json` reports `preview_mode = corrected_followup_3seed`, the top-level classification preview auto-switches to the corrected follow-up control/SSML set. Official paper row is still the historical `strict128` reference table above.

Supplemental CIFAR-100 follow-up preview (`strict128_followup_v1`, corrected `pool_v3`) uses the current follow-up preview case (`uh_sched_mem_v2` by default, promoted case if one clears the narrow-exploit seed2 rule) and automatically renders either the available `2-seed` partial view or the completed `3-seed` view:

![CIFAR-100 follow-up partial validation error](./test_error_classification_cifar100_followup_partial.png)

### Time-Series

![Time-series validation error](./test_error_time_series.png)

### Operator Learning

![Operator validation error](./test_error_operator.png)

<!-- CIFAR100_CIFARSTEM_FOLLOWUP_V1_START -->

## CIFAR-100 cifarstem_followup_v1 Appendix

### Why backbone pivot

SSML 자체가 전반적으로 망가진 것은 아니다. 다른 domain과 일부 classification setting에서는 이미 개선 신호와 승리가 있었고, 현재 CIFAR-100 clean homogeneous는 `resnet34_gelu` strict track의 backbone/stem 병목이 더 크게 보인다. 그래서 이번 pivot은 방법론 포기가 아니라, capacity와 inductive bias를 바꿔 같은 SSML logic를 다시 검증하는 CIFAR-100 병목 분리 실험이다.

### Latest matched result

| Track | Backbone | Protocol | Independent | DML | SSML | Current verdict | Notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CIFAR-100 cifarstem_followup_v1 | `resnet34_cifar_gelu x resnet34_cifar_gelu` | `matched 3-seed` | `0.547167` | `0.550300` | `0.550733` | `SSML > independent and DML` | preview SSML case = `oxtra42_cifarstem_v1` (oxtra); current preview mode = `matched_3seed` |

Promoted backfill targets: `oxtra42_cifarstem_v1`, `pcu_cifarstem_dense_v1`

![CIFAR-100 only validation error](./test_error_cifar100_only.png)
<!-- CIFAR100_CIFARSTEM_FOLLOWUP_V1_END -->
