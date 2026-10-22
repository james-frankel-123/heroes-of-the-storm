# Environment of the P3 definitive build <D> on any host (source with explicit args):
#   . personalization/remote/def_env.sh <training dir> <D>
_T=${1:?training dir}; _D=${2:?build tag}
export DEF_TAG=$_D DEF_CACHE=$_T/personalization/cache_def_$_D DEF_RESULTS=$_T/personalization/results/def_$_D
export P3_CACHE=$DEF_CACHE P3_RESULTS=$DEF_RESULTS P3_RCF_KERNELS=$DEF_CACHE/hs_kernels.npz
export P3_RCF_ROWS=$(ls $DEF_CACHE/merged_rows_${_D}_*.npz 2>/dev/null | head -1)
export DEF_MANIFEST=$DEF_RESULTS/manifest.json
