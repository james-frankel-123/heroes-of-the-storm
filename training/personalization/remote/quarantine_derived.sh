#!/bin/bash
# Run on a worker (in ~/hots/repo/training): move every derived P3 cache the
# October 2026 reruns rebuild into cache/_legacy_pre_oct26/, so no stage can
# silently reuse a table built with the old counts, kernels or keys.
set -euo pipefail
cd ~/hots/repo/training/personalization/cache
mkdir -p _legacy_pre_oct26
for f in hs_slots.npz hs_kernels.npz fix_hs_kernels_l1.npz hs_similarity.npz sd_similarity.npz sd_params.json \
         sd_pred_static_lag1.npz fix_sd_pred_static_lag1.npz x_slots_ext.npz x_predall.npz x_side_slots.npz \
         x_adopt_events.npz x_smurf_flags.npz x_ban_flags.npz x_ban_slotfeat.npz dr_lobbies.npz \
         dr_personal_post.npz dr_imitation.npz dr_runs.pkl.gz x_draft_sim.pkl.gz ds_prior.pt pgd_personal.pt \
         pgd_pref.npz hero_level_causal_snap.npz hero_level_causal_ext.npz; do
  [ -e "$f" ] && mv -f "$f" _legacy_pre_oct26/ || true
done
for f in mcts_*.pkl.gz ds_targets_*.pkl.gz dsmcts_*.pkl.gz dsgreedy_*.pkl.gz pgdmcts_*.pkl.gz; do
  [ -e "$f" ] && mv -f "$f" _legacy_pre_oct26/ || true
done
ls _legacy_pre_oct26 | wc -l
