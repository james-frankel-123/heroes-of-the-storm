"""
P3 — hero similarity by play preference ("players who pick X also pick Y").

From the April 2024 - May 2026 player rows (personalization/cache/
players_2024q2.npz), for players with >= MIN_GAMES games:
  1. Player x hero matrix of log(1 + games), each row centered on the
     player's mean (removes volume) -> hero x hero Pearson correlation
     across players ("preference correlation").
  2. Hero embeddings: truncated SVD (rank K) of that centered matrix; hero
     vector = V * S. Cosine similarity between hero vectors.
  3. Stability: the same correlation on two random halves of the players;
     reports the split-half correlation of all hero-pair entries.
Heroes are ordered by average-linkage clustering on 1 - cosine.

Outputs (personalization/results/): P3_PREF_SIMILARITY.md,
p3_pref_similarity.json, fig_hero_pref_corr.png
Usage (from training/): nice -n 19 taskset -c 48-63 python3 personalization/p3_pref_similarity.py
"""
import os
import sys
import json

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TRAINING)

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PLAYERS = os.path.join(HERE, "cache", "players_2024q2.npz")
OUT = os.path.join(HERE, "results")
MIN_GAMES = 100
K = 12


def corr_matrix(X):
    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    return (Z.T @ Z) / len(Z)


def main():
    from shared import HERO_ROLE_FINE
    p = np.load(PLAYERS)
    names = [str(h) for h in p["hero_names"]]
    pid, pinv = np.unique(p["blizz_ids"], return_inverse=True)
    H = len(names)
    counts = np.zeros((len(pid), H), np.float32)
    np.add.at(counts, (pinv, p["hero"].astype(np.int64)), 1.0)
    keep = counts.sum(1) >= MIN_GAMES
    C = counts[keep]
    print(f"{keep.sum():,} players with >= {MIN_GAMES} games; {C.sum():,.0f} player-games")
    X = np.log1p(C)
    X -= X.mean(1, keepdims=True)
    R = corr_matrix(X)

    U, S, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
    emb = (Vt[:K].T * S[:K])
    embn = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    cos = embn @ embn.T
    var_k = float((S[:K] ** 2).sum() / (S ** 2).sum())

    rng = np.random.RandomState(0)
    half = rng.rand(len(X)) < 0.5
    iu = np.triu_indices(H, 1)
    r_half = float(np.corrcoef(corr_matrix(X[half])[iu], corr_matrix(X[~half])[iu])[0, 1])

    from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
    from scipy.spatial.distance import squareform
    D = np.clip(1 - cos, 0, 2)
    np.fill_diagonal(D, 0)
    Lk = linkage(squareform(D, checks=False), "average")
    order = leaves_list(Lk)
    clusters = fcluster(Lk, t=8, criterion="maxclust")

    role = [HERO_ROLE_FINE.get(n, "?") for n in names]
    pairs = sorted(((R[i, j], names[i], names[j], role[i] != role[j]) for i, j in zip(*iu)),
                   reverse=True)
    top = pairs[:25]
    cross = [x for x in pairs if x[3]][:15]
    anti = pairs[-10:]
    nn = {names[i]: [names[j] for j in np.argsort(-R[i]) if j != i][:3] for i in range(H)}

    os.makedirs(OUT, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(15, 13))
    Ro = R[np.ix_(order, order)].copy()
    np.fill_diagonal(Ro, np.nan)
    lim = np.nanpercentile(np.abs(Ro), 99)
    im = ax.imshow(Ro, cmap="RdBu_r", vmin=-lim, vmax=lim)
    ax.set_xticks(range(H))
    ax.set_yticks(range(H))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=6.5)
    ax.set_yticklabels([names[i] for i in order], fontsize=6.5)
    fig.colorbar(im, ax=ax, shrink=0.7, label="correlation of play preference across players")
    ax.set_title(f"Heroes played by the same players ({keep.sum():,} players with {MIN_GAMES}+ "
                 f"games; split-half stability r = {r_half:.2f})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig_hero_pref_corr.png"), dpi=130)

    res = {"players": int(keep.sum()), "split_half_r": r_half, "svd_k": K,
           "svd_var_explained": var_k,
           "clusters": {int(c): [names[i] for i in order if clusters[i] == c]
                        for c in sorted(set(clusters))},
           "top_pairs": [(a, b, round(float(r), 3)) for r, a, b, _ in top],
           "top_cross_role_pairs": [(a, b, round(float(r), 3)) for r, a, b, _ in cross],
           "most_negative": [(a, b, round(float(r), 3)) for r, a, b, _ in anti],
           "nearest": nn}
    json.dump(res, open(os.path.join(OUT, "p3_pref_similarity.json"), "w"), indent=1)
    lines = ["# Hero similarity by play preference", "",
             f"{keep.sum():,} players with {MIN_GAMES}+ games (April 2024 to May 2026). "
             f"Correlation across players of volume-centered log(1 + games) per hero. "
             f"Split-half stability of the whole matrix: r = {r_half:.2f}. "
             f"A rank-{K} embedding explains {100 * var_k:.0f}% of the variance.", "",
             "## Clusters (average linkage on embedding cosine)", ""]
    for c, hs in res["clusters"].items():
        lines.append(f"- {', '.join(hs)}")
    lines += ["", "## Strongest pairs", "", "| hero | hero | r |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {r:+.2f} |" for a, b, r in res["top_pairs"]]
    lines += ["", "## Strongest cross-role pairs", "", "| hero | hero | r |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {r:+.2f} |" for a, b, r in res["top_cross_role_pairs"]]
    lines += ["", "## Most negative pairs (rarely the same players)", "", "| hero | hero | r |",
              "|---|---|---|"]
    lines += [f"| {a} | {b} | {r:+.2f} |" for a, b, r in res["most_negative"]]
    open(os.path.join(OUT, "P3_PREF_SIMILARITY.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
