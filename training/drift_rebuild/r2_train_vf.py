"""
R2 — train one drift value function with drift2026/train_drift_wp.py,
unchanged, but reading leak-free feature passes (features_oof_<build>.npz from
r1_oof_features.py) and writing models/results under drift_rebuild/.

Any train_drift_wp.py flag passes through. --features oof_<build> reads
drift_rebuild/feature_cache; any other feature pass reads drift2026's cache.
Deployment statistics for the sanity suite: oof_<build> deploys with the
unmodified cumulative stats at <build> (the same stats its deployment-period
rows carry).

Usage:
  CUDA_VISIBLE_DEVICES=1 python3 drift_rebuild/r2_train_vf.py --name r2_allhist_oof_s42 \
      --features oof_2.55.14.95918 --regime all --seed 42
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rb_common as rb  # noqa: E402

from drift2026 import common  # noqa: E402

feat = sys.argv[sys.argv.index("--features") + 1]
if feat.startswith(("oof_", "tierfix_")):
    common.CACHE_DIR = rb.CACHE_DIR
common.MODELS_DIR = rb.MODELS_DIR
common.RESULTS_DIR = rb.RESULTS_DIR

from drift2026 import train_drift_wp as T  # noqa: E402

_orig_deploy = T.deploy_stats


def deploy_stats(features):
    if features.startswith("oof_"):
        return common.load_patch_stats("cumulative", features[len("oof_"):])
    return _orig_deploy(features)


T.deploy_stats = deploy_stats

if __name__ == "__main__":
    T.main()
