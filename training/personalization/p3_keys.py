"""Player identity for the P3 research code: (region, blizz_id).

A blizz_id is unique only within a region, so every player key carries the
region. A NULL player region is replaced by the game's region when the
caller has it (a Storm League game is played within one region); any row
still without a region is kept under the explicit UNKNOWN_REGION code, never
under a negative code that would split one account into several players.
"""
import numpy as np

UNKNOWN_REGION = 0  # HP regions are 1 (US), 2 (EU), 3 (KR), 5 (CN)
REGION_SHIFT = 40


def player_keys(region, blizz_ids, game_region=None, label=""):
    """int64 keys region << 40 | blizz_id; prints how many rows needed the
    game region or the explicit unknown region."""
    r = np.asarray(region).astype(np.int64)
    b = np.asarray(blizz_ids).astype(np.int64)
    assert (b >= 0).all() and (b < (1 << REGION_SHIFT)).all(), "blizz_id outside [0, 2^40)"
    miss = r <= 0
    n_game = 0
    if game_region is not None and miss.any():
        g = np.asarray(game_region).astype(np.int64)
        fill = miss & (g > 0)
        n_game = int(fill.sum())
        r = np.where(fill, g, r)
        miss = r <= 0
    n_unknown = int(miss.sum())
    r = np.where(miss, UNKNOWN_REGION, r)
    if n_game or n_unknown:
        print(f"player keys{(' ' + label) if label else ''}: {n_game:,} rows took the game's region, "
              f"{n_unknown:,} rows kept under the unknown region")
    return (r << REGION_SHIFT) | b


def key_region(keys):
    return np.asarray(keys) >> REGION_SHIFT


def key_blizz(keys):
    return np.asarray(keys) & ((1 << REGION_SHIFT) - 1)
