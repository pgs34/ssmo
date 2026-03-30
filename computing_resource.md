# Computing Resources

- Date: 2026-03-30
- Workspace: `/home/namkyeong/ssmo`

## Workers

- `worker1`
  - GPUs: `2080Ti x2`
  - local GPU ids: `0`, `1`
  - recommended role: classification

- `worker2`
  - GPUs: `3090Ti x1`
  - local GPU id: `0`
  - recommended role: time-series

- `worker3`
  - GPUs: `3090Ti x1`
  - local GPU id: `0`
  - recommended role: operator or backup classification

## Rules

- GPU numbering is local to each worker.
- On `worker2` and `worker3`, always use `GPU=0`.
- Conservative scheduling:
  - one training process per physical GPU
  - use all 4 GPUs before packing multiple jobs onto one GPU
- Default priority:
  1. classification on `worker1`
  2. time-series on `worker2`
  3. operator on `worker3`

## Current Experiment Convention

- `operator` is stable enough to treat as mostly frozen.
- Active iteration focus:
  - `classification`
  - `time-series`
- Results should go under explicit directories instead of relying on a moving default tag.
