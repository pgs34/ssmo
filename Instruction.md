0) 전체 실험 설계의 목표(논문 주장과 1:1 매칭)

SSML이 주장하는 포인트를 실험으로 “증명”하려면 아래 4개를 각각 때려야 합니다.
	1.	Generality: 확률 출력(분류) + 결정론 출력(회귀/필드) 모두 커버
	2.	Heterogeneity: 서로 다른 아키텍처끼리 붙였을 때도 안정적으로 협업
	3.	Negative transfer 억제: 약한 peer가 있을 때 naive mutual은 강한 모델을 망치고, SSML은 덜 망치거나 보호
	4.	Train dynamics: conflict(gradient interaction)가 실제로 줄어드는지 수치로 보여줌

이 4개가 각 태스크 설계에 그대로 들어가야 합니다.

⸻

1) Image classification 실험 설계(구체)

1.1 데이터셋
	•	IID: CIFAR-10, CIFAR-100
	•	Robustness(OOD): CIFAR-10-C, CIFAR-100-C (corruption 평균 성능)
	•	Label noise(negative transfer를 가장 잘 드러냄):
	•	symmetric noise rate: {0.2, 0.4, 0.6}
	•	asymmetric noise: 표준 class-dependent mapping(코드에 흔히 있는 방식)

1.2 모델/페어(heterogeneity를 “보이게” 하는 핵심)
	•	Homogeneous pair
	•	ResNet-18 × ResNet-18 (또는 ResNet-32×2; CIFAR용 표준)
	•	Heterogeneous pair(필수)
	•	ResNet-18 (CNN) × ViT-Tiny/DeiT-Tiny(Transformer)
	•	(옵션) WideResNet × ViT, 또는 ResNet × ConvNeXt-tiny

최소는 “CNN vs Transformer” 1쌍이면 됩니다. 이게 SSML의 차별점(heterogeneity 안정화)을 가장 빠르게 보여줍니다.

1.3 비교할 방법(알고리즘)
	•	Independent: 각자 단독 supervised 학습
	•	Naive mutual: imitation 항상 ON
	•	분류에서는 imitation = KL(softmax)
	•	즉, DML 스타일(양방향 KL)
	•	SSML-hard: per-sample CE(또는 NLL) 비교로 mask, imitation은 KL
	•	SSML-soft(옵션): sigmoid gating(α)로 soft mask
	•	Controls(선택이지만 넣으면 리뷰어가 조용해짐)
	•	EMA self-distillation(Mean-Teacher류)
	•	Ensemble 성능(단, “2모델 추론”은 single-model 결과와 분리해서 표기)

1.4 OOD/노이즈 설정(꼭 이대로 가면 좋음)
	•	CIFAR-C: severity 1~5 평균
	•	label noise: {0, 0.2, 0.4, 0.6} sweep
	•	weak peer stress test(분류에서 매우 강함):
	•	peer2를 약하게 만드는 방법:
	1.	파라미터 수를 줄인 작은 모델로 교체(예: ResNet-18 ↔ ResNet-8급)
	2.	peer2는 train data 25%만 사용
	3.	peer2에만 label noise 주입

1.5 기대 결과(무슨 그림/표가 나와야 “승리”인지)
	•	IID: SSML이 보통 소폭 향상 또는 동급 (여기서 큰 차이 안 나도 괜찮음)
	•	OOD(CIFAR-C): SSML이 유의미하게 더 좋을 가능성이 큼
	•	label noise / weak peer:
	•	Naive mutual은 강한 모델 성능을 깎는 케이스가 반드시 나올 확률이 높음(negative transfer)
	•	SSML은 그 degradation을 거의 막거나 크게 줄이는 형태가 나와야 함
	•	dynamics:
	•	SSML의 conflict rate가 naive mutual보다 낮아야 함(특히 hetero pair에서)

⸻

2) Operator learning 실험 설계(구체)

2.1 데이터셋(최소 3종)
	•	Burgers (1D)
	•	Darcy flow (2D)
	•	Navier–Stokes (2D/time)

(이 3개는 “neural operator 협업” 주장을 직접 받쳐줌)

2.2 모델/페어(여기가 논문 메인)
	•	Homogeneous
	•	FNO × FNO
	•	DeepONet × DeepONet
	•	Heterogeneous(필수)
	•	FNO × DeepONet (최소 1쌍)
	•	(옵션) FNO × GNOT, DeepONet × GNOT

2.3 비교할 방법
	•	Independent
	•	Naive mutual(항상 imitation ON): operator에서는 imitation = L2 everywhere
	•	SSML/SSMO-hard (grid-point wise mask)
	•	SSML/SSMO-soft (옵션: sigmoid mask)

2.4 OOD 설정(“물리 파라미터 shift + resolution shift”로 고정)
	•	Burgers
	•	viscosity(ν) train 범위와 test 범위를 분리(shift)
	•	추가로 initial condition 분포 shift(가능하면)
	•	Darcy
	•	permeability field 분포 shift(예: correlation length/roughness 바꾸기)
	•	forcing shift(가능하면)
	•	Navier–Stokes
	•	forcing/Reynolds 관련 파라미터 shift(가능하면)
	•	resolution generalization: train 저해상도 → test 고해상도(가능하면)

2.5 기대 결과(“SSML이 필요한 이유”가 드러나는 포인트)
	•	IID에서도 improvement가 나올 수 있지만, 핵심은:
	1.	heterogeneous pair에서 naive mutual이 불안정/성능 저하가 나오기 쉬움
	2.	SSML은 그걸 완화하면서 성능을 끌어올리는 방향
	•	OOD/Resolution:
	•	SSML이 더 robust 해야 논문이 강해짐
	•	시각화(오퍼레이터는 이게 먹힘)
	•	error map(공간별 오차)
	•	mask map(어디서 누구를 teacher로 쓰는지)
	•	“error 큰 곳에서 teacher가 바뀌는 패턴”이 보이면 설득력 급상승

2.6 weak peer stress test(오퍼레이터 버전)
	•	peer2 약화 방법:
	•	FNO에서 modes/width 줄이기
	•	DeepONet trunk/branch 폭 줄이기
	•	peer2만 observation noise(출력 u에 노이즈) 또는 데이터 수 줄이기
	•	기대:
	•	naive mutual은 strong model이 끌려 내려감
	•	SSML은 degradation 방지(최소 독립학습 수준 유지)

⸻

3) Time-series forecasting 실험 설계(구체)

Time-series는 classification/operator/segmentation과 동등 중요도로 수행합니다.
실행 효율을 위해 데이터셋/모델 수는 2~3개 데이터셋 + 2개 모델 구성을 기본 단위로 가져갑니다.

3.1 데이터셋(추천 최소 구성)
	•	ETT (ETTh1 또는 ETTm1) 1개
	•	Electricity 1개
	•	Weather 1개
(총 2~3개면 충분)

3.2 모델/페어(heterogeneity가 잘 드러나는 조합)
	•	Transformer forecaster(Informer/Autoformer/PatchTST 중 1개) ×
	•	간단하지만 강한 baseline(DLinear 또는 TCN)
→ “유도편향 다른 모델” 조합이 핵심

3.3 방법
	•	Independent / Naive mutual / SSML-hard / (옵션) SSML-soft

3.4 OOD 설정(시계열에서 가장 싸고 강한 것)
	•	temporal shift: train 초반 기간, test 후반 기간(고정)
	•	seasonal/subperiod test: 특정 구간만 test(예: 여름만)
	•	missingness shift: test에 missing 비율을 인위적으로 올림(10%, 30% 등)

3.5 기대 결과
	•	IID에서는 소폭
	•	long horizon, missingness/temporal shift에서 SSML이 더 유리하게 나올 가능성이 큼
	•	naive mutual은 한 모델이 특정 horizon에서 약하면 그 약점이 전파되어 역효과가 나올 수 있음 → SSML의 차별점

⸻

4) Semantic segmentation 실험 설계(구체)

4.1 데이터셋(최소)
	•	Pascal VOC 2012 (기본)
	•	(옵션) Cityscapes 1개 추가하면 도메인 쉬프트 설득력이 커짐

4.2 모델/페어
	•	U-Net 계열 × DeepLabv3+ 계열 (가장 직관적인 hetero)
	•	(옵션) SegFormer × DeepLab

4.3 SSML 적용 방식(세그멘테이션 핵심 포인트)
	•	supervision: per-pixel CE
	•	imitation: per-pixel KL(softmax)
	•	mask: per-pixel loss 비교로 location-wise gating
→ 이건 SSML의 “where(공간별)” 장점을 가장 직접적으로 보여줌

4.4 OOD/robustness
	•	corruption(blur/noise 등) 적용
	•	resolution shift(입력 스케일 변화)
	•	(옵션) domain shift: VOC에서 학습 → Cityscapes에서 테스트(혹은 반대)

4.5 기대 결과
	•	clean(mIoU)에서 소폭
	•	corruption/resolution/domain shift에서 SSML이 더 robust
	•	naive mutual은 약한 peer의 boundary/texture 오류가 퍼져서 성능 저하가 나올 수 있음 → SSML로 억제

⸻

5) 4.4 Additional experiments를 “실행 단위”로 쪼갠 체크리스트

5.1 Ablation(SSML이 왜 먹히는지 분해)
	•	hard vs soft mask
	•	margin τ 유무/값 sweep
	•	warm-up epoch 수 sweep
	•	λ schedule(상수 vs ramp-up)
	•	selection granularity(sample-wise vs location-wise)
	•	imitation divergence 변경(회귀: L2 vs Huber, 분류: KL vs JS)

기대:
	•	warm-up/τ/soft mask가 mis-selection을 줄여서 안정화
	•	hetero + OOD 환경에서 효과가 더 커짐

5.2 Negative-transfer stress test(반드시 메인 결과로)

태스크별로 “weak peer 만드는 방식” 1~2개 고정해서 표로 보여주면 됩니다.
	•	capacity down / data down / noise injection

기대:
	•	naive mutual은 strong 모델을 깎음
	•	SSML은 깎임을 방지하거나 최소화

5.3 Training dynamics(여기서 논문이 한 단계 올라감)

측정할 로그(모든 태스크 공통으로 가능):
	•	conflict rate:  ⟨g_sup, g_im⟩ < 0 비율
	•	cosine similarity
	•	||g_im|| / ||g_sup||
	•	mask activation ratio(teacher selection fraction)

기대:
	•	SSML이 naive mutual보다 conflict rate가 낮고(특히 hetero), 곡선이 더 안정적

⸻

6) “반드시 돌려야 하는 동등 중요도 실험 세트”

4개 태스크를 동일 우선순위로 수행합니다.
	1.	Classification: CIFAR-10/100 + (hetero pair) + CIFAR-C + weak peer stress test
	2.	Operator learning: Burgers/Darcy/NS + (FNO×DeepONet hetero) + OOD shift + (가능하면 resolution)
	3.	Time-series forecasting: ETT/Electricity/Weather + hetero pair + temporal/missingness shift
	4.	Semantic segmentation: VOC(+Cityscapes 옵션) + hetero pair + corruption/resolution shift
	5.	Dynamics 로그(conflict/cosine/||g_im||/||g_sup||/mask ratio)는 1)~4) 모두에서 공통으로 수집

⸻

원하시면 다음으로는, 위 설계를 바탕으로 **각 태스크별 “실행 계획표”**를 만들겠습니다. 예를 들면:
	•	(태스크) CIFAR-100
	•	(pair) ResNet18×ViT-Tiny
	•	(methods) Indep / Naive / SSML-hard / SSML-soft
	•	(settings) clean + CIFAR-100-C + label-noise 0.4
	•	(expected) naive에서 strong drop, SSML에서 drop 억제
	•	(plots) Conf(t), mask ratio, accuracy curves

이런 형식으로 “실행 가능한 run matrix”까지 내려드릴 수 있습니다.


