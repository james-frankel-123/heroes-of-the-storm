"""Ladder arms on the full-history panel, identical to A2S except the kernel:
offset (no skill state), nopool (cell shrunk alone, t I)."""
import os, sys, argparse
sys.path.insert(0, os.path.expanduser("~/hots/repo/training/personalization"))
import numpy as np
import p3_x_common as X
import p3_c_tests as T
kind = sys.argv[1]
orig = X.kernels
def patched():
    Ks, table = orig()
    H = Ks[T.BEST].shape[0]
    if kind == "offset":
        Ks[T.BEST] = 1e-12 * np.eye(H)
    elif kind == "nopool":
        Ks[T.BEST] = (0.03881045348451971 ** 2) * np.eye(H)
    return Ks, table
X.kernels = patched
B = os.path.expanduser("~/hots/repo/training/personalization")
a = argparse.Namespace(arm=f"L_{kind}", history="merged", rows=f"{B}/cache_full/merged_rows_full_1017.npz",
                       rule="game", proxy="hero_era", refit=False, ex_weight=1.0, exp_counts="slots",
                       half_life=None, since=None)
T.predict(a)
