"""
P3 distilled personalized prior: shared pieces.

Prior for the acting slot's player p at a pick state x (hero index in the
shared.HEROES order, kernel team labels):

  logit(h) = alpha * log p_BC(h | x) + g(f(p, h))       over valid heroes

  p_BC     the outcome-free BC policy (overfit2026/models/bc_prior.pt), the
           kernel's own prior, softmax over the valid mask (free heroes in
           p's pool, as in the kernel)
  f(p, h)  per-hero player context (all causal, from the personal tables and
           the imitation recency features):
             b2 * s(p, h)        personal skill term on the logit scale
             off-role(p, h)
             log(1 + games on h), never played h (window counts)
             EWMA pick share of h (half-life 20 and 100 games)
             log(1 + days since p last played h), never played (recency)
             log share of p's games in h's fine role
  g        MLP 9 -> 32 -> 32 -> 1 shared across heroes
g depends only on the player, so for a lobby the term is a per-slot vector
(10 x 90) that the prior kernel adds to its PUCT priors; alpha is a scalar.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
import torch.nn as nn

import p3_hs_core as C
import p3_mcts_core as M

MODEL = os.path.join(C.CACHE, "ds_prior.pt")
FEATS = ["b2_s", "off_role", "log1p_n", "never_n", "ewma20", "ewma100", "log1p_days_since", "never_played",
         "log_fine_share"]


class DistillPrior(nn.Module):
    def __init__(self, nf=len(FEATS), h=32):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.g = nn.Sequential(nn.Linear(nf, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, 1))
        self.register_buffer("mu", torch.zeros(nf))
        self.register_buffer("sd", torch.ones(nf))

    def bias(self, F):
        """F: (..., 90, nf) raw features -> (..., 90) prior term."""
        return self.g((F - self.mu) / self.sd).squeeze(-1)

    def forward(self, bc_logp, F, mask):
        u = self.alpha * bc_logp + self.bias(F)
        return u.masked_fill(~mask, -1e9)


def player_features(S, rows):
    """(len(rows), 90, nf) in MY hero order, float32."""
    I = S.I
    rows = np.asarray(rows)
    rec = I.recency_features(S.d, rows)
    fake = {"row": rows, "recpos": np.arange(len(rows)), "lp": np.zeros((len(rows), 90), np.float32)}
    Xp = I.feature_tensor(fake, S.T, rec, S.meta)  # gd_logp, log1p_n, never_n, e20, e100, log1p_days, never_pl, log_fine
    pos = np.array([S.T["pos"][int(r)] for r in rows])
    b2 = float(S.coefs[2])
    F = np.concatenate([(b2 * S.T["s"][pos])[..., None], S.T["off"][pos][..., None].astype(np.float32), Xp[:, :, 1:]],
                       -1)
    return F.astype(np.float32)


def state_and_mask(lob, k, pool_row_my, to_sh):
    """Kernel-format 290-d state before pick step k (real prefix) and the
    kernel valid mask (free and in the acting player's pool), shared order."""
    acts = lob["acts_real"]
    x = np.zeros(290, np.float32)
    taken = np.zeros(90, bool)
    for j in range(k):
        h = int(acts[j])
        taken[h] = True
        if M.IS_PICK[j]:
            x[(0 if M.DRAFT_TEAM[j] == 0 else 90) + h] = 1
        else:
            x[180 + h] = 1
    x[270 + lob["map"]] = 1
    x[284 + lob["tier"]] = 1
    x[287] = k / 15.0
    x[288] = 1.0
    x[289] = float(M.DRAFT_TEAM[k])
    pool = np.zeros(90, bool)
    pool[to_sh] = pool_row_my
    m = ~taken & pool
    if not m.any():
        m = ~taken
    return x, m


def bc_logp(net, X, mask, device="cpu", bs=8192):
    out = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs], device=device)
            mb = torch.tensor(mask[i:i + bs].astype(np.float32), device=device)
            lg = net(xb, mb)
            lg = lg[0] if isinstance(lg, tuple) else lg
            out.append(torch.log_softmax(lg, -1).cpu().numpy())
    return np.concatenate(out)


def load_model():
    ck = torch.load(MODEL, weights_only=False, map_location="cpu")
    m = DistillPrior()
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m, ck


def slot_bias(model, S, lob_rows, to_sh):
    """(10, 90) prior term in SHARED order for a lobby's slot rows."""
    F = player_features(S, lob_rows)
    with torch.no_grad():
        b = model.bias(torch.tensor(F)).numpy()
    out = np.zeros((len(lob_rows), 90), np.float32)
    out[:, to_sh] = b
    return out
