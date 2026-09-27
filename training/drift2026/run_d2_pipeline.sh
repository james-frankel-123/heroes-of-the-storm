#!/bin/bash
# Full D2 pipeline: 4 feature-extraction passes (CPU, ~2-3h) then the 30-job
# training grid on the GPU pool (~2-4h). Idempotent: existing outputs are
# skipped, so it can be relaunched after a failure.
set -e
cd "$(dirname "$0")/.."
python3 -u drift2026/build_drift_features.py
python3 -u drift2026/phase_d2.py
