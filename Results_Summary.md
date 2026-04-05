# Results Summary

현재 워크스페이스 기준에서 **paper candidate win**과 **latest control / non-win**을 다시 정리한 요약이다.
공식 비교는 **같은 epoch budget으로 맞춘 completed 결과**를 우선 사용하고, `partial / exploratory / 다른 epoch` 결과는 참고용으로만 적는다.

## Confirmed SSML Wins

| Task | Dataset | Pair | Best SSML | Independent | DML | Ordering | Note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| Operator | Burgers | `FNO x DeepONet` | `0.0000031432` | `0.0000042316` | `0.0000093153` | `SSML < independent < DML` | `operator_burgers_fno_polish_fair_v2 / fno_polish_coarse_l002_w24_d90_170` 3-seed mean best 기준, 같은 `180 epoch` budget에서 strongest `FNO independent`를 넘어섬 |
| Operator | Darcy | `FNO x DeepONet` | `0.003148` | `0.003200` | `0.004758` | `SSML < independent < DML` | 현재 전체 실험 중 가장 깔끔한 승리 |
| Time-Series | Weather | `transformer x dlinear` | `0.261344` | `0.272783` | `0.281672` | `SSML < independent < DML` | `instruction_matrix_v1` 3-seed mean 기준, activation 이후 실제 개선 |
| Time-Series | Electricity | `transformer x dlinear` | `0.100142` | `0.152387` | `0.164034` | `SSML < independent < DML` | `corrective_v1 / corr_gate64_l15_sp5e4` 3-seed mean best 기준, strongest independent는 `dlinear` |
| Classification | CIFAR-10 | `resnet18 x resnet18` | `0.864233` | `0.809233` | `0.849667` | `independent < DML < SSML` | SSML은 `instruction_matrix_v1`, DML은 `classification_cifar10_homo_dml_long_v1` 3-seed mean 기준. long rerun 후에도 SSML이 여전히 앞섬 |

## Needs Fix

| Task | Dataset | Pair | SSML | Best baseline | Problem |
| --- | --- | --- | ---: | ---: | --- |
| Time-Series | ETTh1 | `transformer x dlinear` | `0.283504` | `dlinear independent = 0.272032`, `transformer independent = 0.321354`, `DML = 0.327667` | 새 `teacher_ft_v1`가 checkpoint warm-start + frozen `dlinear` teacher로 SSML을 `0.283504`까지 끌어내려 transformer baseline과 DML은 확실히 넘겼다. 그래도 strongest single baseline인 `dlinear independent 0.272032`는 아직 못 넘어서 paper win으로는 여전히 부족하다 |
| Classification | CIFAR-100 | `resnet34_gelu x resnet34_gelu` | `0.536567` | `independent = 0.528533`, `DML = 0.545067` | 기존 best는 여전히 `augfilter_seeded_v1 = 0.536567`이다. 새 `teacher_ft_seeded_v4`는 best completed 3-seed가 `0.532800`이라 prior SSML best도, matched clean homogeneous `DML 0.545067`도 못 넘었다 |
| Operator | Darcy | `DeepONet <- frozen FNO teacher` | `best SSML = 0.013723` | `best control fine-tune = 0.013669`, `original DeepONet independent = 0.018561` | 최신 `relay_v1`의 `darcy_relay_coarse = 0.013723`가 이전 `student_lift_v2 best 0.013763`보단 소폭 낫다. 그래도 best no-peer control `0.013669`는 아직 못 넘어서 아직 non-win이다 |

## Detail Tables

### Operator

| Item | Value |
| --- | --- |
| Best setting | `Burgers / FNO x DeepONet / fair_v2` |
| SSML | `0.0000031432` |
| FNO independent | `0.0000042316` |
| DML | `0.0000093153` |
| Ordering | `SSML < independent < DML` |
| SSML reference | [summary.json](/home/namkyeong/ssmo/results/operator_burgers_fno_polish_fair_v2/worker3_gpu0/fno_polish_coarse_l002_w24_d90_170/operator/burgers/fno__deeponet_ssml_mse_seed0/summary.json) |
| Independent reference | [summary.json](/home/namkyeong/ssmo/results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno_independent_mse_seed0/summary.json) |
| DML reference | [summary.json](/home/namkyeong/ssmo/results/operator_burgers_followup_v1/burgers_l005_m0_w12_d60_120_ow1/operator/burgers/fno__deeponet_dml_mse_seed0/summary.json) |

### Time-Series

| Dataset | Pair | SSML | Independent | DML | Extra Baseline | Ordering | Note |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| Weather | `transformer x dlinear` | `0.261344` | `0.272783` | `0.281672` | - | `SSML < independent < DML` | `instruction_matrix_v1` 3-seed mean 기준 |
| ETTh1 | `transformer x dlinear` | `0.283504` | `0.321354` | `0.327667` | `dlinear independent = 0.272032`, `teacher_ft_v1 next = 0.291310`, `corrective v1 exploratory best = 0.329268`, `complementary v4 best = 0.346186` | `dlinear independent < SSML < transformer independent < DML` | `time_series_etth1_teacher_ft_v1` best full 3-seed `tft_tail10_reg15_l010_lr2e4 = 0.283504` 기준. frozen `dlinear` teacher + transformer checkpoint fine-tuning으로 transformer baseline은 확실히 넘겼지만, strongest single baseline인 dlinear와는 아직 차이가 남는다 |
| Electricity | `transformer x dlinear` | `0.100142` | `0.165290` | `0.164034` | `dlinear independent = 0.152387` | `SSML < dlinear independent < DML < transformer independent` | SSML은 `corrective_v1 / corr_gate64_l15_sp5e4` 3-seed mean best_val_mse 기준. 다만 curve 자체는 `epoch 11~13` 부근에서 best를 찍은 뒤 후반에 다시 상승하는 late-drift 패턴이 있어, final epoch보다 early best / best checkpoint 기준으로 읽는 편이 맞다. 후속 `corrective_v2`는 late drift는 줄였지만 best mean이 `0.161111~0.167274`로 내려가 기존 win을 재현하지 못했다 |

| Dataset | SSML reference | Independent reference | DML reference |
| --- | --- | --- | --- |
| Weather | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_ssml_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer_independent_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/time_series/time_series/weather/transformer__dlinear_dml_mse_seed0/summary.json) |
| ETTh1 | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_teacher_ft_v1/tft_tail10_reg15_l010_lr2e4/time_series/etth1/transformer__dlinear_ssml_huber_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer_independent_huber_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_etth1_all_methods_long_v3/time_series/etth1/transformer__dlinear_dml_huber_seed0/summary.json) |
| Electricity | [summary.json](/home/namkyeong/ssmo/results/time_series_electricity_corrective_v1/corr_gate64_l15_sp5e4/time_series/electricity/transformer__dlinear_ssml_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer_independent_mse_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/time_series_electricity_followup_v1/best_known/time_series/electricity/transformer__dlinear_dml_mse_seed0/summary.json) |

### Classification

| Dataset | Pair | Run | SSML | Independent | DML | Ordering | Note |
| --- | --- | --- | ---: | ---: | ---: | --- | --- |
| CIFAR-100 | `resnet34_gelu x resnet34_gelu` | `augfilter_seeded_v1 best = 0.536567`, `alt_focus_v2 best = 0.532567`, `visual_complement_v3 full best = 0.528567` | `0.536567` | `0.528533` | `0.545067` | `independent < SSML < DML` | `augfilter_seeded_v1` full 3-seed best는 `pcu_pb20_thr38_gap20_augmin72_augmax90_agap03 = 0.536567`로 prior SSML best를 갱신했다. 하지만 matched DML reference `0.545067`에는 아직 못 미침 |
| CIFAR-10 | `resnet18 x resnet18` | `SSML/independent = instruction_matrix_v1`, `DML = homo_dml_long_v1` | `0.864233` | `0.809233` | `0.849667` | `independent < DML < SSML` | 중간에 끊기던 DML curve를 `100 epoch`로 다시 돌려도 SSML이 앞섬 |

| Dataset | SSML reference | Independent reference | DML reference |
| --- | --- | --- | --- |
| CIFAR-100 | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_augfilter_seeded_v1/node0_gpu1/pcu_pb20_thr38_gap20_augmin72_augmax90_agap03/classification/cifar100/resnet34_gelu_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_neural_ode_cifar100_v11_main/baseline/classification/cifar100/resnet34_gelu_independent_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar100_homo_dml_reference_v1/dml_l4e2_t6/classification/cifar100/resnet34_gelu_dml_kl_seed0/summary.json) |
| CIFAR-10 | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_ssml_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/instruction_matrix_v1/classification_homo/classification/cifar10/resnet18_independent_kl_seed0/summary.json) | [summary.json](/home/namkyeong/ssmo/results/classification_cifar10_homo_dml_long_v1/classification/cifar10/resnet18_dml_kl_seed0/summary.json) |

## Carry Forward

| Task | Setting | Why keep it |
| --- | --- | --- |
| Operator | `Burgers / FNO x DeepONet / fair_v2` | 같은 `180 epoch` budget의 strongest `FNO independent`를 실제로 넘긴 최신 승리 |
| Operator | `Darcy / FNO x DeepONet / SSML` | 가장 깔끔한 초기 operator 승리 |
| Time-Series | `Weather / transformer x dlinear / instruction_matrix_v1` | 실제로 `SSML < independent < DML` |
| Time-Series | `Electricity / transformer x dlinear / corrective_v1 corr_gate64_l15_sp5e4` | strongest single baseline인 `dlinear independent`까지 넘긴 새 승리 |
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
| `weather rescue v17~v18` | 성능 실패 |
| `CIFAR-100 / clean homogeneous current paper claim` | matched `DML` reference `0.545067`가 현재 SSML `0.536567`보다 높음 |
| `CIFAR-100 / visual_complement_v3 full sweep` | best full 3-seed가 `0.528567`로 prior SSML best `0.532567`보다 낮음. overbatch single-seed spike는 아직 incomplete라 보류 |
| `classification worker3 backup` | baseline 이하 |
| `classification clean hetero ResNet x ViT` | clean IID에서는 아직 약함 |
| `classification v17 main` | baseline 근처이거나 아래 |

## Latest Checked Runs

| Run | Status | Key result | Interpretation |
| --- | --- | --- | --- |
| `instruction_matrix_v1 / worker2 time_series` | 완료 | `36 summaries`, Weather mean `0.261344`, ETTh1 mean `0.298151`, Electricity mean `0.168784` | Weather/Electricity 값은 현재 표에 반영된 3-seed mean과 일치. ETTh1는 기존 carry-forward best는 아님 |
| `time_series_electricity_followup_v1 / best_known` | 완료 | `12 summaries`, transformer independent mean `0.165290`, DML mean `0.164034`, SSML mean `0.165709`, dlinear independent mean `0.152387` | transformer/DML/dlinear baseline reference로 유지. SSML 자체는 이제 corrective run으로 대체 |
| `time_series_electricity_corrective_v1` | 완료 | best `corr_gate64_l15_sp5e4` mean best_val_mse `0.100142`, next `corr_gate64_do10_l20_sp10e4` = `0.100196` | Electricity `Needs Fix`는 해소. corrective SSML이 strongest baseline까지 넘김 |
| `time_series_electricity_corrective_v1 / curve audit` | 완료 | seed0 기준 `best_val_mse = 0.099632 @ epoch 13`, final `epoch 60 = 0.161586`; seed1/2도 동일하게 후반 재상승 | 플롯에서 SSML이 다시 올라가는 건 버그가 아니라 실제 late-drift다. `first_active_epoch=1`, `active_imitation_ratio ~ 0.90 -> 0.99`, `mean_imitation_weight ~ 0.58 -> 0.92`로 초반부터 guidance가 너무 넓게 켜져 후반 과교정/overfit이 생긴 쪽으로 해석하는 게 맞음 |
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
| `time_series_etth1_all_methods_long_v3` | 완료 | `6 seeds / 80 epochs / no early-stop`: `dlinear independent = 0.272032`, `transformer independent = 0.321354`, `SSML = 0.325609`, `DML = 0.327667` | ETTh1가 짧게 끝나던 문제를 길게 다시 돌려도 결론은 안 바뀌었다. SSML/DML 모두 transformer baseline도 못 넘고, strongest single baseline인 dlinear와는 격차가 큼 |
| `time_series_etth1_teacher_ft_v1` | 완료 | best full 3-seed `tft_tail10_reg15_l010_lr2e4 = 0.283504`, next `tft_tail20_reg20_l015_lr3e4 = 0.291310` | 이번 follow-up 중 가장 의미 있는 개선이다. frozen `dlinear` teacher + transformer checkpoint fine-tuning으로 SSML이 transformer baseline `0.321354`와 DML `0.327667`는 확실히 넘었다. 다만 strongest `dlinear independent 0.272032`는 아직 못 넘음 |
| `time_series_etth1_teacher_ft_v2` | 완료 | best completed 3-seed `trs13_res04_lr2e4 = 0.296472`, next `trs14_res02_lr2e4 = 0.302700`, `trs18_res01_lr3e4 = 0.308239` | `trend/residual scale` 분리는 방향성은 있었지만, 여전히 correction density가 너무 높았다. best epoch는 대체로 `20~26`으로 이동했지만 `active_imitation_ratio`가 `0.47~0.96` 수준으로 넓게 켜져 late drift가 다시 강하게 남았다 |
| `time_series_etth1_teacher_ft_v3` | 완료 | best completed 3-seed `b12_q80_m20_t18_r00_lr3e4 = 0.316102`, next `b10_q80_m20_t14_r00_lr2e4 = 0.318657` | `budget cap + stronger teacher-advantage focus`는 drift는 줄였다. best epoch가 `22~24`로 유지되고 last epoch에선 gate가 거의 꺼지지만, 너무 보수적으로 바뀌면서 값이 transformer independent 근처까지 다시 올라가 rescue 효과 자체를 잃었다 |
| `time_series_etth1_handoff_router_seeded_v1` | 완료 | best full 3-seed `hr_long_q85_b08_h34 = 0.319122`, next `hr_bal_q80_b12_h28 = 0.319461`, `hr_mid_q82_b10_h30 = 0.319697` | `handoff + horizon routing + trend-only`는 late drift를 줄이고 final을 `0.322104` 근처로 안정화했다. 다만 기존 best rescue `teacher_ft_v1 = 0.283504`보다 훨씬 나쁘고 strongest `dlinear independent 0.272032`도 못 넘었다 |
| `classification_cifar10_homo_dml_long_v1` | 완료 | `resnet18:resnet18 DML / 3 seeds / 100 epochs` mean `0.849667` | top-level classification plot에서 중간에 끊겨 보이던 CIFAR-10 DML curve를 길게 다시 채웠다. 기존 `0.836867`보다 올라왔지만, SSML `0.864233`는 여전히 못 넘음 |
| `operator_burgers_followup_v1` | 부분 완료 | first completed case `burgers_l005_m0_w12_d60_120_ow1`: `FNO independent = 4.2316e-06`, `SSML = 4.4312e-06`, `DML = 9.3153e-06` | low-lambda late-guidance로 gap을 줄이긴 했지만, strongest `FNO independent`는 아직 못 넘었다. 즉 follow-up 방향은 더 봐야 하지만 현 시점 결과는 non-win |
| `classification_cifar100_teacher_ft_seeded_v4` | 완료 | best full 3-seed `uh_late_pb16_thr38_gap22_aug78_94_ag03 = 0.532800`, next `pcu_late_pb14_thr42_gap26_aug80_92_ag04 = 0.531833` | checkpoint warm-start late teacher fine-tuning은 안전하긴 했지만, 기존 `seeded_v1 best 0.536567`와 matched `DML 0.545067`는 못 넘었다 |
| `classification_cifar100_disagreement_memory_seeded_v1` | 진행 중 | completed full 3-seed best `pcu_mem_df35_m90_x10 = 0.532100`, next `uh_mem_df35_m90_x10 = 0.530767`; `pcu_mem_df45_m95_x15`, `uh_mem_df50_m95_x20`는 아직 실행 중 | `disagreement preserve + class-deficit memory`는 현재까지는 기존 `seeded_v1 best 0.536567`보다 낮다. 진행 중 두 케이스의 현재 single-run peak는 각각 `0.5356`, `0.5351`이지만 아직 completed 3-seed 결과가 아니라 carry-forward는 보류한다 |
| `classification_cifar100_augfilter_complement_lite_seeded_v1` | 실행 중 | `4 settings / 100 epoch / 3 seeds`, best `augfilter_seeded_v1` 주변에 `light disagreement floor + light deficit memory`만 얹은 fair rerun | 강한 `disagreement_memory`는 val_acc를 같이 눌렀다. 이번엔 기존 best augfilter selection은 유지하고 complement preservation만 약하게 넣어서 matched `DML 0.545067`를 다시 추격한다 |
| `operator_burgers_student_lift_v2` | 완료 | control `ctrl_ft_lr4e4 = 0.005031`, best SSML `elem_ft_huber_l002_w30_d80_140_lr3e4 = 0.004658`, original `DeepONet independent = 0.012436` | weak `DeepONet`는 크게 개선됐다. 다만 이 실험은 strong `FNO`를 넘기는 그림이 아니라 student rescue에 가깝고, strongest `FNO independent 4.2316e-06`와는 여전히 큰 차이가 난다 |
| `operator_burgers_relay_v1` | 완료 | `140 epoch` relay best는 `burgers_relay_hotspot = 0.004683`, next `burgers_relay_coarse = 0.004759` | relay 자체는 `student_lift_v2 best 0.004658`보다도 살짝 약했다. 그리고 이 `140 epoch` 결과를 `180 epoch`의 `FNO independent 4.2316e-06`와 직접 비교하면 안 된다. Burgers의 공식 비교는 계속 matched `180 epoch`인 `followup_v1` row를 기준으로 본다 |
| `operator_burgers_fno_polish_fair_v2` | 완료 | best full 3-seed `fno_polish_coarse_l002_w24_d90_170 = 3.1432e-06`, next `fno_polish_full_l0015_w30_d100_180 = 3.1552e-06`, `fno_polish_hotspot_l003_w20_d80_160 = 3.3594e-06` | `FNO <- DeepONet` student-only + frozen peer strong-model polish가 통했다. 완료된 세 세팅이 전부 prior strongest `FNO independent 4.2316e-06`보다 낮았고, 그중 best가 새로운 Burgers matched win이 됐다 |
| `operator_darcy_student_lift_v1` | 완료 | control `ctrl_ft_lr3e4 = 0.013669`, best SSML `sample_ft_l010_w5_d20_70_lr3e4 = 0.013768`, next `elem_ft_l010_w5_d20_70_lr3e4 = 0.013860`, original `DeepONet independent = 0.018561` | weak `DeepONet` 자체는 꽤 올라왔다. 다만 peer-guided SSML이 같은 init checkpoint에서의 no-peer fine-tuning control을 넘지는 못했다. 즉 현재 Darcy의 주된 승리는 여전히 `strong FNO polishing` 쪽이고, `weak-model lift` 메시지는 추가 개선이 필요하다 |
| `operator_darcy_student_lift_v2` | 완료 | control `ctrl_ft_lr2e4 = 0.013792`, best SSML `elem_ft_l002_w20_d50_95_lr2e4 = 0.013763`, next `sample_ft_l003_w15_d45_90_lr2e4 = 0.013821`, original `DeepONet independent = 0.018561` | `v2`는 same-campaign control은 소폭 넘겼다. 그래도 best no-peer control across runs `0.013669`는 아직 못 넘어서, weak-model lift를 논문 메시지로 쓰기엔 조금 더 필요하다 |
| `operator_darcy_relay_v1` | 완료 | best full 3-seed `darcy_relay_coarse = 0.013723`, next `darcy_relay_hotspot = 0.014294` | `relay`가 현재 weak-model lift 쪽 최고치는 만들었다. 그래도 best control `0.013669`는 못 넘었고, 깔끔한 win까지는 아직 한 끗 모자란다 |
| `time_series_etth1_teacher_win_reweight_fair_v1` | 실행 중 | `3 settings / 80 epoch / 3 seeds`, checkpoint warm-start + frozen `dlinear` teacher + `reweight_only + handoff + horizon router` | `teacher_ft_v1`는 strongest `dlinear`까지는 못 갔다. 이번엔 output imitation을 버리고 teacher-win horizon에만 student supervised를 더 싣는 구조로, 같은 `80 epoch` budget에서 `dlinear independent 0.272032`를 다시 추격한다 |
| `paper_gap_v1 / classification_homo_noise` | 완료 | `independent mean = 0.462900`, `DML mean = 0.491633`, `SSML mean = 0.405367` | `noise=0.2` reference는 확보됐지만 clean homogeneous CIFAR-100 비교 기준으로 직접 쓰면 안 됨 |
| `paper_gap_v1 / classification_hetero_noise` | 진행 중 | summary `0개`, `independent seed0` epoch log만 존재 | 미시작은 아님. 아직 첫 clean baseline run도 summary까지는 못 감 |

## Plot Preview

### Classification

![Classification validation error](./test_error_classification.png)

### Time-Series

![Time-series validation error](./test_error_time_series.png)

### Operator

![Operator validation error](./test_error_operator.png)
