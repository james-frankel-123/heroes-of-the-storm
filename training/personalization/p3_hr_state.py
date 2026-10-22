"""Main-model state as of 2026-09-28 (end of the full_1017 store) for every
pulled player, for the B2 replay study's pre-2.57 baseline. Writes
cache/hr/state_0928.npz (gitignored, player keys): region, blizz_id,
mean_all / var_all (full_ms posterior, residual units), n_all (games in the
full history), offset_all (experience offset at slot-row counts), hero_names."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np  # noqa: E402

import p3_hr_common as HC  # noqa: E402
import p3_hs_core as C  # noqa: E402
from p3_keys import key_blizz, key_region  # noqa: E402

a = HC.load()
t = C.day_of("2026-09-29")
players = np.sort(a["pulled_pid"]).astype(np.int64)
st = HC.state_at(a, t, players)
m, v = HC.post_all(HC.params("full_ms"), st)
sl = (a["slot"] >= 0) & (a["day"] < t)
pos = np.full(int(a["n_players"]), -1, np.int64)
pos[players] = np.arange(len(players))
ok = sl & (pos[a["pid"]] >= 0)
cell = pos[a["pid"][ok]] * 90 + a["hero"][ok].astype(np.int64)
nph = np.bincount(cell, None, len(players) * 90).reshape(len(players), 90)
npl = nph.sum(1, keepdims=True) * np.ones((1, 90))
off = a["table"][C.exp_bins(npl.astype(np.int64).ravel(), nph.astype(np.int64).ravel())].reshape(len(players), 90)
keys = a["player_keys"][players]
np.savez(os.path.join(HC.HR_CACHE, "state_0928.npz"), region=key_region(keys), blizz_id=key_blizz(keys).astype(np.int64),
         mean_all=m, var_all=v, n_all=st["N"].astype(np.float32), offset_all=off.astype(np.float32),
         hero_names=a["hero_names"], as_of_day=np.int64(t))
print("wrote state", len(players))
