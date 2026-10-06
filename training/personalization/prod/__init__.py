"""Production plumbing for the P3 personal layer (review sections 5.1, 5.4, 5.5).

  wp_chain       versioned causal WP scoring chain (vintage registry, rescoring,
                 residual noise-ratio invariant)
  gates          regression battery and deployment gates on a frozen golden set
  nightly_state  player_skill_state builder (as-of visibility, incremental and
                 full rebuild paths) and the serve-time collision-weighted lookup

Nothing here is wired into cron, refresh.py or the site. numpy only.
"""
