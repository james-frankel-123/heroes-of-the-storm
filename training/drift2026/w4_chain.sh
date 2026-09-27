#!/bin/bash
# W4 auto-launch chain (2026-07-07): wait for the phase_w2 GPU pool to drain,
# run the wave-2 summaries, then launch the W4 MCTS driver (best vs worst
# regime VFs — d2c_cumprev vs d2b_allhist — 5 seeds x 200 sims x 300K eps,
# future-GD benchmark). Runs detached via:
#   nohup bash drift2026/w4_chain.sh > drift2026/logs/phase_w4.log 2>&1 &
# Monitor:
#   tail -f drift2026/logs/phase_w4.log drift2026/logs/w4_mcts_*.log
cd /home/max/heroes-of-the-storm/training || exit 1
set -a; source ../.env; set +a

echo "[w4_chain] waiting for phase_w2 pool to finish..."
until grep -q "Pool finished" drift2026/logs/phase_w2_pool.log 2>/dev/null; do
    sleep 30
done
while pgrep -f "drift2026/phase_w2.py" > /dev/null; do sleep 10; done
echo "[w4_chain] pool finished: $(grep 'Pool finished' drift2026/logs/phase_w2_pool.log)"

echo "[w4_chain] running summarize_w2.py (full, incl. W2a depth analysis)..."
python3 -u drift2026/summarize_w2.py > drift2026/logs/summarize_w2.log 2>&1 \
    || echo "[w4_chain] WARNING: summarize_w2 failed (see logs/summarize_w2.log); continuing to W4"

echo "[w4_chain] launching W4 MCTS driver..."
exec python3 -u drift2026/phase_w4_mcts.py
