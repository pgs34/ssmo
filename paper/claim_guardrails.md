# Paper Claim Guardrails

This file is the safety rail for the manuscript.
Use it when turning `paper_draft.md` into a polished paper, poster, or abstract.

## Safe Main Claim

SSML is a selective peer-teaching rule that improves over matched independent training and the repository's soft-gated DML baseline in multiple finalized settings across classification, time-series forecasting, and operator learning.

## Safe Headline Wins

| Task | Dataset | Pair | SSML | Independent | DML | Main-paper status |
| --- | --- | --- | ---: | ---: | ---: | --- |
| Classification | CIFAR-10 | `resnet18 x resnet18` | `0.864233` | `0.809233` | `0.849667` | use |
| Time-series | Weather | `transformer x dlinear` | `0.261344` | `0.272783` | `0.281672` | use |
| Time-series | Electricity | `transformer x dlinear` | `0.100142` | `0.152387` | `0.164034` | use, but note best-checkpoint reporting |
| Operator | Burgers | `FNO x DeepONet` | `0.0000009653` | `0.0000009846` | `0.0000093153` | use, but note DML reference is older completed family |
| Operator | Darcy | `FNO x DeepONet` | `0.003148` | `0.003200` | `0.004758` | use |

## Safe Negative / Boundary Claims

| Task | Dataset | Current reading | How to write it |
| --- | --- | --- | --- |
| Time-series | ETTh1 | deployed best branch equals frozen peer on all three seeds | not a clean student-improvement win; use as failure case |
| Classification | CIFAR-100 strict128 | SSML beats independent but not DML | unresolved or partial success, not a headline win |

## Do Not Claim

1. Do not claim ETTh1 as a true SSML win.
2. Do not claim original strict CIFAR-100 as a win over DML.
3. Do not use `strict128_aggressive_v1` or `scaled_fair_aggressive_v1` as official evidence.
4. Do not describe the repository `dml` row as if it were guaranteed to be identical to every canonical DML implementation in the literature.
5. Do not say SSML is universally better across all tasks in this repo.

## Diagnostic-Only Results

These are useful for analysis, not for headline claims.

1. `classification_cifar100_strict128_aggressive_v1`
   Reason: `lr / weight_decay` wiring bug caused undertraining.
2. `classification_cifar100_scaled_fair_aggressive_v1`
   Reason: same wiring bug.
3. `time_series_etth1_teacher_ft_pairdeploy_reeval_20260405_v1`
   Reason: best deployed branch is effectively peer selection, not clear student lift.

## Appendix Candidate

This one looks promising, but it changes the classification backbone protocol, so keep it separate unless we intentionally promote it.

| Track | Independent | DML | SSML | Suggested use |
| --- | ---: | ---: | ---: | --- |
| `CIFAR-100 cifarstem_followup_v1` | `0.547167` | `0.550300` | `0.550733` | appendix or follow-up diagnosis |

## Source of Truth

Primary numeric source:

1. `/home/namkyeong/ssmo/Results_Summary.md`

Main plot files already available in the repository:

1. `/home/namkyeong/ssmo/test_error_classification.png`
2. `/home/namkyeong/ssmo/test_error_classification_cifar100_followup_partial.png`
3. `/home/namkyeong/ssmo/test_error_cifar100_only.png`
4. `/home/namkyeong/ssmo/test_error_time_series.png`
5. `/home/namkyeong/ssmo/test_error_operator.png`

## Recommended Paper Structure

1. Main text:
   five confirmed wins plus two honest failure cases
2. Appendix:
   CIFAR-100 backbone-pivot follow-up, extra curve plots, and track-by-track notes
3. Future work:
   broader robustness shifts, more exact DML baselines, and stronger homogeneous CIFAR-100 protocol
