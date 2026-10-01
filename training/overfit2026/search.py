"""
Draft generation harness over the CUDA MCTS kernel with a swappable proxy.

Uses overfit2026/cuda_ofit (a copy of training/cuda_mcts with two additions:
a tree-capacity guard plus overflow counter, and an optional two-output
"LCB" value net). With guard=0 and a standard 1-output WP net the copy runs
the original code path (verified bit-identical in check_kernel()).

Priors
  ckpt:<run>    a paper MCTS checkpoint (rerun2026/mcts_runs/<run>)
  uniform       AlphaZeroDraftNet with a zeroed policy head -> uniform prior
  bc            outcome-free behavioral-cloning prior (models/bc_prior.pt,
                distilled from the GD opponent model; see train_bc_prior)
Opponent and rollout policy: the paper's 5 GD models, cycled per batch.

Search (X2 fix, 2026-10-01): search_mode 1 = opponent chance nodes (default),
0 = the pre-fix kernel (bit-identical to every result produced before the
fix), 2 = open-loop roll-forward. The default can be overridden with the
OFIT_SEARCH_MODE environment variable to rerun old scripts unchanged.
"""
import os
import sys
import json
import importlib.util

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)
sys.path.insert(0, os.path.join(TRAINING_DIR, "cuda_mcts"))

import numpy as np
import torch

RR = os.path.join(TRAINING_DIR, "rerun2026")


def load_kernel(which="ofit"):
    if which == "ofit":
        d, pref = os.path.join(HERE, "cuda_ofit"), "ofit_kernel"
    else:
        d, pref = os.path.join(TRAINING_DIR, "cuda_mcts"), "cuda_mcts_kernel"
    so = [f for f in os.listdir(d) if f.startswith(pref) and f.endswith(".so")][0]
    spec = importlib.util.spec_from_file_location(pref, os.path.join(d, so))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_GD = None


def gd_dir():
    """The GD opponent pool: the site-tier retrain (rerun2026 namespace
    p1site) under P1_TIERS=site, else the paper's pool."""
    from overfit2026 import data as _d
    if _d.TIER_SCHEME == "site":
        return os.path.join(RR, "ns", "p1site", "models")
    return os.path.join(RR, "models")


def gd_flats():
    global _GD
    if _GD is None:
        from train_generic_draft import GenericDraftModel
        from extract_weights import extract_gd_weights
        _GD = []
        for i in range(5):
            gd = GenericDraftModel()
            gd.load_state_dict(torch.load(os.path.join(gd_dir(), f"generic_draft_{i}.pt"),
                                          weights_only=True, map_location="cpu"))
            gd.eval()
            _GD.append(extract_gd_weights(gd))
    return _GD


def policy_net(kind):
    from train_draft_policy import AlphaZeroDraftNet
    net = AlphaZeroDraftNet(size="base", policy_head_type="linear")
    if kind.startswith("ckpt:") or kind.startswith("path:"):
        path = (os.path.join(RR, "mcts_runs", kind[5:], "draft_policy.pt")
                if kind.startswith("ckpt:") else kind[5:])
        sd = torch.load(path, weights_only=True, map_location="cpu")
        if any(k.startswith("res_block1.") for k in sd):
            sd = {k.replace("res_block1.", "res_blocks.0.").replace("res_block2.", "res_blocks.1.")
                  .replace("res_block3.", "res_blocks.2."): v for k, v in sd.items()}
        net.load_state_dict(sd)
    elif kind == "uniform":
        with torch.no_grad():
            for p in net.policy_head.parameters():
                p.zero_()
    elif kind == "bc":
        net.load_state_dict(torch.load(os.path.join(HERE, "models", "bc_prior.pt"),
                                       weights_only=True, map_location="cpu"))
    else:
        raise ValueError(kind)
    net.eval()
    return net


def policy_flat(kind):
    from extract_weights import extract_policy_weights
    return extract_policy_weights(policy_net(kind))


def wp_config(models, stats, lcb_lambda=0.0):
    """models: list of WinProbEnrichedModel (283-d, 2 hidden layers each).
    One model -> the standard net. Several -> an exact ReLU network whose two
    outputs are the members' mean logit and mean absolute deviation of the
    member logits (used as a pessimism penalty on our side only)."""
    from extract_weights import (extract_wp_weights, build_wp_net_offsets,
                                 extract_lookup_tables)
    lut = extract_lookup_tables(stats)
    if len(models) == 1 and lcb_lambda == 0.0:
        wf, wn = extract_wp_weights(models[0])
        wo = build_wp_net_offsets(models[0], wn, 283)
        return wf, wo, lut
    K = len(models)

    class _L:   # BatchNorm-folded linear layer (eval mode)
        def __init__(self, W, b):
            self.weight, self.bias = W, b
            self.out_features, self.in_features = W.shape

    def folded(mod):
        mods, out, i = list(mod.net), [], 0
        while i < len(mods):
            m = mods[i]
            if isinstance(m, torch.nn.Linear):
                W, b = m.weight.detach().cpu().clone(), m.bias.detach().cpu().clone()
                if i + 1 < len(mods) and isinstance(mods[i + 1], torch.nn.BatchNorm1d):
                    bn = mods[i + 1]
                    g = bn.weight.detach().cpu() / torch.sqrt(bn.running_var.cpu() + bn.eps)
                    W = W * g[:, None]
                    b = (b - bn.running_mean.cpu()) * g + bn.bias.detach().cpu()
                out.append(_L(W, b))
            i += 1
        return out
    lin = [folded(mod) for mod in models]
    assert all(len(l) == 3 for l in lin), "members must be 283->h1->h2->1"
    h1 = [l[0].out_features for l in lin]
    h2 = [l[1].out_features for l in lin]
    assert sum(h1) <= 1024 and sum(h2) <= 1024 and 2 * K + 2 <= 1024
    W1 = torch.cat([l[0].weight for l in lin], 0)
    b1 = torch.cat([l[0].bias for l in lin], 0)
    W2 = torch.zeros(sum(h2), sum(h1))
    r = c = 0
    for l in lin:
        W2[r:r + l[1].out_features, c:c + l[1].in_features] = l[1].weight
        r += l[1].out_features
        c += l[1].in_features
    b2 = torch.cat([l[1].bias for l in lin], 0)
    W3 = torch.zeros(K, sum(h2))
    c = 0
    for k, l in enumerate(lin):
        W3[k, c:c + l[2].in_features] = l[2].weight[0]
        c += l[2].in_features
    b3 = torch.stack([l[2].bias[0] for l in lin])
    # layer 4 (ReLU): [relu(z_k - m)]_k, [relu(m - z_k)]_k, relu(m), relu(-m)
    A = torch.eye(K) - torch.full((K, K), 1.0 / K)
    mrow = torch.full((1, K), 1.0 / K)
    W4 = torch.cat([A, -A, mrow, -mrow], 0)
    b4 = torch.zeros(2 * K + 2)
    # layer 5 (linear): mean logit, mean |z_k - m|
    W5 = torch.zeros(2, 2 * K + 2)
    W5[0, 2 * K] = 1.0
    W5[0, 2 * K + 1] = -1.0
    W5[1, :2 * K] = 1.0 / K
    b5 = torch.zeros(2)
    layers = [(W1, b1, 1), (W2, b2, 1), (W3, b3, 0), (W4, b4, 1), (W5, b5, 0)]
    flat, off = [], 0
    wo = {"num_layers": 5, "use_enriched": 1, "input_dim": 283, "use_sigmoid": 0,
          "n_out": 2, "lcb_lambda": float(lcb_lambda),
          "layer_in": [0] * 6, "layer_out": [0] * 6, "weight_off": [0] * 6,
          "bias_off": [0] * 6, "has_bn": [0] * 6, "bn_w_off": [0] * 6,
          "bn_b_off": [0] * 6, "bn_m_off": [0] * 6, "bn_v_off": [0] * 6,
          "use_relu": [0] * 6}
    for i, (W, b, relu) in enumerate(layers):
        W = W.detach().float().contiguous()
        b = b.detach().float().contiguous()
        wo["layer_in"][i], wo["layer_out"][i] = W.shape[1], W.shape[0]
        wo["weight_off"][i] = off
        flat.append(W.flatten())
        off += W.numel()
        wo["bias_off"][i] = off
        flat.append(b)
        off += b.numel()
        wo["use_relu"][i] = relu
    return torch.cat(flat).numpy().astype(np.float32), wo, lut


def lcb_reference(models, X):
    """Python reference of the combined net on (N,283) inputs -> (mean logit, MAD)."""
    with torch.no_grad():
        zs = []
        for m in models:
            p = m(torch.tensor(X)).view(-1).clamp(1e-7, 1 - 1e-7)
            zs.append(torch.log(p / (1 - p)))
        Z = torch.stack(zs)
        mu = Z.mean(0)
        return mu.numpy(), (Z - mu).abs().mean(0).numpy()


def draft_configs(n, seed, tier_idx=1):
    rng = np.random.RandomState(seed)
    from shared import MAPS
    return np.array([[rng.randint(len(MAPS)), tier_idx, rng.randint(2)] for _ in range(n)],
                    dtype=np.int32)


def run(kernel, pflat, wcfg, n=200, sims=200, seed=1234, root_temp=1.0, dir_eps=0.0,
        guard=1, device_id=0, batch=100, tier_idx=1, is_ofit=True,
        search_mode=None, pw_k=1.0, pw_alpha=0.5):
    from shared import HEROES, NUM_HEROES, MAPS, SKILL_TIERS
    pf, po = pflat
    wf, wo, lut = wcfg
    if search_mode is None:
        search_mode = int(os.environ.get("OFIT_SEARCH_MODE", "1"))
    cfgs = draft_configs(n, seed, tier_idx)
    out = []
    gds = gd_flats()
    for bi, bs in enumerate(range(0, n, batch)):
        ca = np.ascontiguousarray(cfgs[bs:bs + batch])
        gf, go = gds[bi % len(gds)]
        Eng = getattr(kernel, "OfitEngine", None) or kernel.MCTSKernelEngine
        eng = Eng(pf, gf, wf, po, go, wo, lut,
                                      max_concurrent=len(ca), device_id=device_id)
        if not hasattr(kernel, "SEARCH_CHANCE"):   # build predates the X2 fix
            if search_mode != 0:
                raise RuntimeError("kernel build predates the X2 fix; rebuild it or pass search_mode=0")
            extra = ()
        else:
            extra = (search_mode, pw_k, pw_alpha)
        if is_ofit:
            res = eng.run_episodes(ca, sims, 2.0, seed + bs, root_temp, 0.3, dir_eps, guard, *extra)
        else:
            res = eng.run_episodes(ca, sims, 2.0, seed + bs, root_temp, 0.3, dir_eps, *extra)
        del eng
        for k, r in enumerate(res):
            t = np.array(r[2])
            mi, ti, ou = (int(x) for x in ca[k])
            t0 = [HEROES[j] for j in range(NUM_HEROES) if t[j] > 0.5]
            t1 = [HEROES[j] for j in range(NUM_HEROES) if t[NUM_HEROES + j] > 0.5]
            d = {"our": t0 if ou == 0 else t1, "opp": t1 if ou == 0 else t0,
                 "map": MAPS[mi], "tier": SKILL_TIERS[ti], "side": ou,
                 "kernel_wp": float(r[0]), "search_mode": search_mode}
            if is_ofit:
                d["max_nodes"] = int(r[4])
                d["cap_hits"] = int(r[5])
            out.append(d)
    return out
