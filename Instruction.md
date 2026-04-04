컨셉: SSML은 서로 다른 오류를 내는 두 모델이, 상대가 더 잘하는 부분만 선택적으로 가르쳐서 자기 약점을 보완한다.

목표: SSML이 성능으로 여러 task에서 DML/Independt를 이기는것

각 task 마다 컨셉에 맞춰서 다른 실험 및 feature engineering 가능

-------------------------------------------------------
Our computing resourcs are:
node0  : 4090x2
worker1: 2080tiX2
worker2: 3090tiX1
worker3: 3090tiX1

make sure we use all the vram as much as possible
-------------------------------------------------------

-> 결과 요약 방법:
최신 결과 중 잘나온 부분은 Results_Summary.md에 요약해서 정리하며
test_error_{task}.png 에 그림을 업데이트 한다
또한 끊임 없이 자동으로 개선 방안을 제시하며 수정사항 및 실햄 스크립트를 제시한다


-------------------------------------------------------

현재  Etth1은 실행 에폭이 너무 작으니 모든 모델에서 더 길게 해서 (60 이상) 또한 그림상 CIFAR 10 DML의 경우 epoch 이 중간에 멈춰서 다시 길게 실행하거라