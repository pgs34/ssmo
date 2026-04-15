# Selective-Supervised Mutual Learning: Peer Teaching Only Where the Peer Is Better

## Draft Status

Working draft based on the repository state on 2026-04-07.
All headline numbers below are copied from `Results_Summary.md` and are intentionally conservative.
This draft is written to be easy to port into LaTeX later; citations should be added in a second pass.

## Abstract

Mutual learning can improve a model by exposing it to a peer's predictions, but broad imitation also transfers the peer's mistakes. We study Selective-Supervised Mutual Learning (SSML), a simple rule that preserves the ordinary supervised objective everywhere and adds imitation only on samples, elements, or horizons where the peer currently has lower supervised error. The same principle can be instantiated across classification, operator learning, and time-series forecasting by changing only the granularity of the selection mask. Across matched three-seed comparisons in this repository, SSML yields five confirmed wins over both independent training and the repository's soft-gated DML baseline: CIFAR-10 classification, Burgers and Darcy operator learning, and Weather and Electricity forecasting. The gains range from 1.6% relative error reduction on Darcy to 34.3% on Electricity, plus a 1.46-point accuracy gain over DML on CIFAR-10. At the same time, the method is not uniformly successful: ETTh1 currently collapses to deployed peer selection rather than genuine student improvement, and the original CIFAR-100 strict protocol still trails the best DML baseline. These results suggest that selective imitation is a useful cross-domain principle, but that stability and protocol design remain critical in hard homogeneous settings.

## 1. Introduction

Peer-based training methods are appealing because different models often make different mistakes. If one model is better on some portion of the data, it seems natural to let the other model learn from it. The difficulty is that unconditional or overly broad imitation can become harmful: a peer that is locally helpful may still be globally weaker, unstable, or badly calibrated on other regions.

This repository explores a simple alternative. Instead of asking a model to mimic its peer everywhere, SSML asks it to imitate the peer only where the peer is currently better according to the supervised loss. The supervised target remains active on all training signals, so the peer acts as a selective auxiliary teacher rather than a replacement objective.

The working hypothesis is straightforward: when the two models have complementary errors, selective imitation should preserve the benefit of knowledge transfer while reducing negative transfer. This idea is especially attractive when the pair is heterogeneous, such as `Transformer + DLinear` or `FNO + DeepONet`, because each model family tends to excel on different structures.

The current workspace already gives us a paper-worthy empirical story, although not a clean universal win. We now have:

1. Confirmed wins on five finalized settings across three domains.
2. A clear account of where the method fails or becomes ambiguous.
3. Enough implementation detail to describe a single core algorithm with task-specific instantiations.

### Main contributions

1. We formulate SSML as supervised learning plus peer imitation masked by peer-superiority.
2. We show that the same rule can be applied at multiple granularities: sample-level for classification, field or element-level for operator learning, and horizon-level for forecasting.
3. We provide a transparent empirical study with both positive results and explicit non-wins, rather than filtering the paper down to successful cases only.

## 2. Method

### 2.1 Core idea

Consider two models, `f_i` and `f_j`, trained on the same example or structured output location `z`. Let `ell_sup_i(z)` be model `i`'s supervised loss on `z`. SSML defines a directional binary gate

`w_{i<-j}(z) = 1[ell_sup_j(z) + m < ell_sup_i(z)]`,

where `m >= 0` is an optional margin. Model `i` then optimizes

`L_i = E_z[ell_sup_i(z)] + lambda * E_z[w_{i<-j}(z) * ell_imit(f_i(z), stopgrad(f_j(z)))]`.

The supervised term is always present. The imitation term is present only where the peer is better.

This is the key difference from broad mutual learning: the peer does not define the full target everywhere, and a weak peer cannot freely pull the student in regions where it is already worse.

### 2.2 Relation to the repository baselines

The repository compares three methods:

1. `independent`: no peer interaction.
2. `dml`: supervised learning plus a soft peer-better imitation gate.
3. `ssml`: supervised learning plus a hard peer-better imitation gate.

So the main head-to-head in this codebase is not "hard selection versus classic always-on DML", but rather "hard selective imitation versus soft selective imitation." That distinction should be stated clearly in the paper to avoid overselling the baseline setup.

### 2.3 Task-specific instantiations

The same rule is used at different granularities.

#### Classification

For image classification, the selection unit is primarily the sample. The peer-teaching score is derived from quantities such as the student's supervised error, the peer's true-class probability, or the peer-student probability gap. In the stronger CIFAR-100 follow-ups, SSML additionally uses top-k filtering, disagreement floors, and class-balancing heuristics to prevent the mask from becoming too dense or too class-skewed.

#### Time-series forecasting

For forecasting, the selection unit can be a timestep or horizon element. This is important because a peer may be helpful only on specific forecast regions, such as long-horizon trend segments. The stronger runs in this repository add sparse top-k selection, handoff schedules, and corrective gates so that imitation is concentrated where the peer advantage is large enough to matter.

#### Operator learning

For operator learning, the selection unit may be a sample-level score or an elementwise field score. The strongest runs use relay or polish variants that let a strong peer provide structured hints only on regions where it has lower reconstruction error.

### 2.4 Practical principle

Across all three domains, the main practical lesson is the same: sparse and directional guidance is usually safer than dense peer imitation. When the imitation mask becomes too active for too long, the training often drifts toward the peer rather than borrowing only the peer's strengths.

## 3. Experimental Protocol

### 3.1 Benchmarks and model pairs

We focus on the settings that are currently most mature in the repository.

1. Classification:
   `CIFAR-10`, pair `resnet18 x resnet18`
2. Time-series forecasting:
   `Weather`, `Electricity`, and `ETTh1`, pair `transformer x dlinear`
3. Operator learning:
   `Burgers` and `Darcy`, pair `FNO x DeepONet`

### 3.2 Baselines

Each official row compares the same dataset, seed count, and epoch budget while changing only the training method:

1. `independent`
2. `dml`
3. `ssml`

All headline numbers in this draft use completed three-seed summaries whenever possible. Partial, exploratory, or mismatched-budget runs are excluded from the main claims.

### 3.3 Metrics and reporting rules

1. Classification reports top-1 accuracy, where higher is better.
2. Time-series and operator learning report MSE, where lower is better.
3. Official comparisons use matched completed runs first.
4. Electricity is reported with best-checkpoint selection because late drift makes final-epoch comparison misleading in the current stable run family.
5. ETTh1 is not counted as a clean win because the best deployed branch is effectively frozen-peer selection.

## 4. Main Results

### 4.1 Confirmed wins

| Domain | Dataset | Pair | SSML | Best baseline | Gain |
| --- | --- | --- | ---: | ---: | --- |
| Classification | CIFAR-10 | `resnet18 x resnet18` | `0.864233` acc | `0.849667` acc | `+1.4566` accuracy points over DML |
| Time-series | Weather | `transformer x dlinear` | `0.261344` MSE | `0.272783` MSE | `4.19%` relative reduction |
| Time-series | Electricity | `transformer x dlinear` | `0.100142` MSE | `0.152387` MSE | `34.28%` relative reduction |
| Operator | Burgers | `FNO x DeepONet` | `0.0000009653` MSE | `0.0000009846` MSE | `1.96%` relative reduction |
| Operator | Darcy | `FNO x DeepONet` | `0.003148` MSE | `0.003200` MSE | `1.63%` relative reduction |

These five rows are the cleanest current evidence that SSML can outperform both independent training and the repository's soft-gated DML variant.

### 4.2 What the wins suggest

The results support three working claims.

1. Selective imitation helps most when the pair is complementary.
   The strongest cross-domain wins appear in heterogeneous pairs such as `Transformer + DLinear` and `FNO + DeepONet`.
2. Sparse guidance matters.
   The good forecasting and operator runs are not "imitate the teacher more"; they are "imitate only the teacher's locally strong regions."
3. The method is useful but not magic.
   Even where SSML wins, the gain size depends heavily on optimization stability and how the selective mask is constructed.

## 5. Failure Cases and Honest Boundaries

### 5.1 ETTh1 is not a clean SSML win

The best deployed ETTh1 value in the current summary is `0.271835`, which matches the rerun `dlinear` independent mean. However, the deployed `best_branch` resolves to the frozen `dlinear` peer on all three seeds. That means the strongest ETTh1 number should be interpreted as pair-time branch selection, not as evidence that the student model itself improved past the peer.

For the paper, ETTh1 should therefore be framed as a diagnostic failure case:

1. SSML can rescue a weaker student partway.
2. The final stable deployment may still collapse to choosing the peer.
3. Selective teaching and selective branch selection are not the same claim.

### 5.2 CIFAR-100 is still unresolved in the original strict protocol

The historical strict homogeneous row remains

`SSML = 0.536567`, `Independent = 0.528533`, `DML = 0.545067`.

So SSML beats independent but not DML. That is promising, but it is not a full win under the original official protocol.

We also have two reasons to stay conservative:

1. Some later aggressive tracks were invalidated by an `lr / weight_decay` wiring bug and should be treated as diagnostic only.
2. The corrected strict follow-up has not yet delivered a completed three-seed replacement that surpasses the original DML reference.

### 5.3 CIFAR-100 backbone-pivot follow-up is promising but should stay appendix-only for now

The repository now contains a matched `cifarstem_followup_v1` appendix block with

`Independent = 0.547167`, `DML = 0.550300`, `SSML = 0.550733`.

This is encouraging because it suggests the earlier bottleneck may have been tied to the original backbone or stem choice rather than the SSML principle itself. Still, because this changes the backbone family for the benchmark, the safest paper strategy is to keep it as an appendix or a "follow-up diagnosis" unless we explicitly decide to promote the backbone pivot into the main protocol.

## 6. Discussion

### 6.1 Why selective imitation appears to help

The simplest explanation is negative-transfer control. If the peer is only used where it is actually better, the student can borrow complementary structure without inheriting the peer's full error profile. This is especially visible in heterogeneous pairs where inductive biases differ substantially.

### 6.2 Why the hard settings stay hard

The non-wins are also informative.

1. Homogeneous CIFAR-100 is sensitive to optimization, augmentation, and mask design.
2. Forecasting can suffer late drift when imitation stays active too densely or too late in training.
3. A deployment rule that is allowed to choose between branches can hide whether the student actually improved.

These issues do not invalidate SSML, but they do tell us that the paper should emphasize stability and diagnosis instead of claiming universal superiority.

## 7. Limitations

This work, at its current repository stage, has several clear limitations.

1. The strongest evidence comes from a small set of finalized benchmark pairs rather than a very broad architecture sweep.
2. Some domains required substantial task-specific engineering to make selective imitation stable.
3. The current DML comparison is the repository implementation, which uses a soft peer-better gate rather than an exact reproduction of every historical DML variant.
4. The latest strongest Burgers regime does not yet have a fully matched rerun of DML under the exact same polish setting.
5. CIFAR-100 and ETTh1 show that the method can still fail, collapse to peer selection, or depend strongly on protocol details.

## 8. Conclusion

SSML is a simple idea: keep supervised learning everywhere, and let the peer teach only where the peer is currently better. In the current workspace, that rule is already enough to produce five confirmed wins across classification, forecasting, and operator learning. The same results also show the boundaries of the idea. Selective imitation is not automatically robust, and the hardest settings still require careful mask design, optimization, and reporting discipline.

The right paper message is therefore not "SSML wins everywhere." The right message is stronger and more credible: selective peer teaching is a reusable learning principle that works across several domains, and its failure modes are concrete enough to study rather than hand-wave away.

## 9. Paper TODOs

Before converting this draft into a submission-ready manuscript, we should add:

1. Citations for mutual learning, distillation, operator learning, and forecasting baselines.
2. One conceptual figure showing the selective gate versus broad imitation.
3. One main results table plus one honest failure-case table.
4. A compact appendix describing the CIFAR-100 backbone-pivot follow-up.
5. Standard deviations or confidence intervals for the final main-table rows.
