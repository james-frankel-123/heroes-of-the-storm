# Environment for rich-CF runs on the full_1017 panel, on any host with the
# repo at $1 (default: the current repo). Source it: . rcf_env.sh [repo training dir] [results subdir]
_T=${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
_R=${2:-rcf_shrink}
export P3_CACHE=$_T/personalization/cache_full P3_RESULTS=$_T/personalization/results/$_R \
  P3_RCF_ROWS=$_T/personalization/cache_full/merged_rows_full_1017.npz \
  P3_RCF_KERNELS=$_T/personalization/cache_full/hs_kernels.npz
