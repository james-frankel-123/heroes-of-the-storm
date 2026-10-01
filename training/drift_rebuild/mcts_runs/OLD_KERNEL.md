# Old-kernel MCTS agents (do not reuse after the X2 kernel fix)

Every agent in this directory (r6_*) and every drift2026 agent
(`training/drift2026/mcts_runs/w4_*, w7_*, w8_*`) was trained with
`training/cuda_mcts/mcts_kernel.cu` before the X2 fix: the tree expands
nodes only at our own turn, so search never reaches past the current
own-pick block. Max ruled this a must-fix (2026-10-01); these agents must
be retrained on the fixed kernel.

Stopped 2026-10-01 ~15:2x at the coordinator's request:
- r6_uoof_gc_s4, r6_uoof_gc_s5 (killed mid-training; checkpoints kept,
  incomplete)
- r6_mcut_rg_s0-2 never started.
Completed before the stop (old kernel): r6_uoof_gc_s0-s3.
