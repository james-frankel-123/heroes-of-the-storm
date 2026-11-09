"""
CQL and IQL in their published, bootstrapped form, on the same p1site naive
transitions, network and optimizer as the Monte Carlo baselines (CQL naive,
experiment_iql_draft), so the only change is the backup.

Transitions. The cache holds 32 rows per replay: 16 draft steps from each
team's seat (rows 16-31 mirror rows 0-15 with the team columns exchanged).
A row's successor is the next row of the same half with the same outcome: the
acting team's next decision, with the opponent's logged moves in between as
the environment. A team's last decision is terminal; the reward is the game
result there and 0 elsewhere, gamma = 1.

  CQL (Kumar et al., 2020; discrete CQL(H) on a DQN backbone):
      y = r                                    at a team's last decision
      y = sigmoid(max_a' Qbar(s', a'))         otherwise (valid a' only)
      loss = BCE(sigmoid(Q(s,a)), y) + alpha * (logsumexp_a Q(s,a) - Q(s,a_data))
  IQL (Kostrikov et al., 2022):
      V    <- expectile_tau( sigmoid(Qbar(s,a)) - V(s) )
      Q    <- BCE(sigmoid(Q(s,a)), r or V(s'))
      pi   <- AWR, weights exp(beta * (sigmoid(Qbar(s,a)) - V(s))), clipped at 100

Qbar is a Polyak copy (0.005 per step). Q lives on the sigmoid scale, as in the
Monte Carlo baselines, so BCE replaces the squared error; both have the same
minimizer. Data actions that are invalid under the state's mask (about 3e-5 of
rows, duplicate-hero records) get zero weight. Budget: Adam 3e-4, batch 2048,
a fixed 15 epochs, final weights (TD losses do not support early stopping on
held-out error); held-out diagnostics are logged per epoch. Resumable at every
epoch. Checkpoints are CQLDraftAgent state dicts, so make_cql_strategy loads them.

Usage (from training/, on a GPU box holding the packed caches):
  python3 paper1_revision/td_offline.py cql --alpha 1.0
  python3 paper1_revision/td_offline.py iql --tau 0.7 --beta 3.0
  python3 paper1_revision/td_offline.py check          # successor-index checks only
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING_DIR = os.path.dirname(HERE)
sys.path.insert(0, TRAINING_DIR)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NS = os.path.join(TRAINING_DIR, "rerun2026", "ns", "p1site")
CACHE = os.path.join(NS, "feature_cache")
OUT = os.path.join(NS, "models", "td")
STEP_COL = 287
EPOCHS = 15
BATCH = 2048
LR = 3e-4
POLYAK = 0.005


# ── data ──

_SHIFTS = None


def unpack(bits, step):
    global _SHIFTS
    if _SHIFTS is None or _SHIFTS.device != bits.device:
        _SHIFTS = torch.arange(7, -1, -1, device=bits.device, dtype=torch.uint8)
    b = ((bits.unsqueeze(-1) >> _SHIFTS) & 1).reshape(len(bits), 288).float()
    return torch.cat([b[:, :STEP_COL], (step.float() / 15.0).unsqueeze(1), b[:, STEP_COL:]], 1)


# side of the actor at each step: 0 = the team that bans first (it also picks first)
SIDE = [0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1]


def successors(outs, step):
    """Index of each row's next decision by the same side in its 16-row half, -1 if last.
    Sides follow the fixed draft order (SIDE); ban rows carry no reliable actor
    label in the cache, so the outcome is never used to find the actor. A side's
    last decision is a pick (steps 14 and 15), whose stored outcome is the actor's."""
    n = len(outs)
    assert n % 32 == 0, n
    st = step.reshape(-1, 16)
    assert (st == np.arange(16)).all(), "rows are not in 16-step halves"
    nxt_step = [next((k for k in range(j + 1, 16) if SIDE[k] == SIDE[j]), -1) for j in range(16)]
    base = np.arange(n // 16, dtype=np.int64)[:, None] * 16
    ns = np.array(nxt_step)
    nxt = np.where(ns >= 0, base + ns, -1)
    o = outs.reshape(-1, 16)
    # pick rows: the two sides' outcomes must differ (one winner)
    pick = [k for k in range(16) if k in (4, 5, 6, 7, 8, 11, 12, 13, 14, 15)]
    a_side = [k for k in pick if SIDE[k] == 0]
    b_side = [k for k in pick if SIDE[k] == 1]
    bad = int(((o[:, a_side].min(1) != o[:, a_side].max(1)) | (o[:, b_side].min(1) != o[:, b_side].max(1))
               | (o[:, a_side[0]] == o[:, b_side[0]])).sum())
    return nxt.reshape(-1), bad


class Data:
    def __init__(self, split, dev):
        d = os.path.join(CACHE, f"cql_naive_{split}", "packed")
        assert os.path.exists(os.path.join(CACHE, f"cql_naive_{split}", "swapfix.json"))
        L = lambda f: np.load(os.path.join(d, f), mmap_mode="r")[:].copy()
        bits, step, acts, outs = L("bits.npy"), L("step.npy"), L("actions.npy"), L("outcomes.npy")
        nxt, self.bad_halves = successors(outs, step)
        T = lambda x: torch.from_numpy(x).to(dev)
        self.bits, self.step = T(bits), T(step)
        self.acts, self.outs = T(acts).long(), T(outs).float()
        self.nxt = T(nxt)
        self.n = len(acts)
        self.dev = dev

    def batch(self, idx):
        s = unpack(self.bits[idx], self.step[idx])
        ni = self.nxt[idx]
        term = ni < 0
        nic = ni.clamp(min=0)
        s2 = unpack(self.bits[nic], self.step[nic])
        m = 1.0 - s[:, :270].reshape(-1, 3, 90).sum(1).clamp(0, 1)
        m2 = 1.0 - s2[:, :270].reshape(-1, 3, 90).sum(1).clamp(0, 1)
        a = self.acts[idx]
        ok = m.gather(1, a.unsqueeze(1)).squeeze(1) > 0.5
        return s, a, self.outs[idx], m, s2, m2, term, ok

    def batches(self, shuffle, limit=None):
        order = torch.randperm(self.n, device=self.dev) if shuffle else torch.arange(self.n, device=self.dev)
        if limit:
            order = order[:limit]
        for i in range(0, len(order), BATCH):
            yield self.batch(order[i:i + BATCH])


# ── networks ──

def qnet():
    from experiment_cql_draft import CQLDraftAgent
    return CQLDraftAgent()


class ValueNet(nn.Module):
    def __init__(self, input_dim=289):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, s):
        return torch.sigmoid(self.net(s)).squeeze(-1)


def polyak(net, tgt):
    with torch.no_grad():
        for p, tp in zip(net.parameters(), tgt.parameters()):
            tp.mul_(1 - POLYAK).add_(p, alpha=POLYAK)


def wmean(x, w):
    return (x * w).sum() / w.sum().clamp(min=1)


def expectile(u, tau, w):
    return wmean(torch.where(u > 0, tau, 1 - tau) * u.pow(2), w)


# ── algorithms ──

class CQL:
    def __init__(self, alpha, dev):
        self.alpha = alpha
        self.q, self.qt = qnet().to(dev), qnet().to(dev)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()
        self.opt = torch.optim.Adam(self.q.parameters(), lr=LR)
        self.nets = {"q": self.q, "qt": self.qt}
        self.opts = {"q": self.opt}

    def target(self, o, s2, m2, term):
        with torch.no_grad():
            boot = torch.sigmoid(self.qt(s2, m2).max(1).values)
        return torch.where(term, o, boot)

    def step(self, s, a, o, m, s2, m2, term, ok):
        w = ok.float()
        y = self.target(o, s2, m2, term)
        qr = self.q(s, m)
        qa = qr.gather(1, a.unsqueeze(1)).squeeze(1)
        bell = wmean(F.binary_cross_entropy_with_logits(qa, y, reduction="none"), w)
        pen = wmean(torch.logsumexp(qr, 1) - qa, w)
        loss = bell + self.alpha * pen
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        polyak(self.q, self.qt)
        return {"bellman": bell.item(), "cql": pen.item()}

    def policy_state(self):
        return self.q.state_dict()

    @torch.no_grad()
    def evaluate(self, s, a, o, m, s2, m2, term, ok):
        w = ok.float()
        qr = self.q(s, m)
        qa = qr.gather(1, a.unsqueeze(1)).squeeze(1)
        y = self.target(o, s2, m2, term)
        p = torch.sigmoid(qr) * m
        k = m.sum(1)
        mu = p.sum(1) / k
        sd = torch.sqrt(((p - mu[:, None]) ** 2 * m).sum(1) / k)
        return {"td_bce": wmean(F.binary_cross_entropy_with_logits(qa, y, reduction="none"), w).item(),
                "mc_bce": wmean(F.binary_cross_entropy_with_logits(qa, o, reduction="none"), w).item(),
                "match": wmean((qr.argmax(1) == a).float(), w).item(),
                "q_sd_over_actions": sd.mean().item(),
                "q_mean": wmean(torch.sigmoid(qa), w).item()}


class IQL:
    def __init__(self, tau, beta, dev):
        self.tau, self.beta = tau, beta
        self.q, self.qt, self.pi = qnet().to(dev), qnet().to(dev), qnet().to(dev)
        self.qt.load_state_dict(self.q.state_dict())
        self.qt.eval()
        self.v = ValueNet().to(dev)
        self.nets = {"q": self.q, "qt": self.qt, "pi": self.pi, "v": self.v}
        self.opts = {k: torch.optim.Adam(self.nets[k].parameters(), lr=LR) for k in ("q", "pi", "v")}

    def step(self, s, a, o, m, s2, m2, term, ok):
        w = ok.float()
        with torch.no_grad():
            qbar = torch.sigmoid(self.qt(s, m).gather(1, a.unsqueeze(1)).squeeze(1))
        v = self.v(s)
        v_loss = expectile(qbar - v, self.tau, w)
        self.opts["v"].zero_grad()
        v_loss.backward()
        self.opts["v"].step()

        with torch.no_grad():
            y = torch.where(term, o, self.v(s2))
            adv = qbar - self.v(s)
            aw = torch.exp(self.beta * adv).clamp(max=100.0) * w
        qa = self.q(s, m).gather(1, a.unsqueeze(1)).squeeze(1)
        q_loss = wmean(F.binary_cross_entropy_with_logits(qa, y, reduction="none"), w)
        self.opts["q"].zero_grad()
        q_loss.backward()
        self.opts["q"].step()
        polyak(self.q, self.qt)

        logits = self.pi(s, m)
        a_safe = torch.where(ok, a, logits.argmax(1))
        nll = F.cross_entropy(logits, a_safe, reduction="none")
        pi_loss = (aw * nll).sum() / w.sum().clamp(min=1)
        self.opts["pi"].zero_grad()
        pi_loss.backward()
        self.opts["pi"].step()
        return {"q": q_loss.item(), "v": v_loss.item(), "pi": pi_loss.item(),
                "w_mean": wmean(aw, w).item(), "adv_sd": adv[ok].std().item()}

    def policy_state(self):
        return self.pi.state_dict()

    @torch.no_grad()
    def evaluate(self, s, a, o, m, s2, m2, term, ok):
        w = ok.float()
        qa = self.q(s, m).gather(1, a.unsqueeze(1)).squeeze(1)
        y = torch.where(term, o, self.v(s2))
        logits = self.pi(s, m)
        return {"td_bce": wmean(F.binary_cross_entropy_with_logits(qa, y, reduction="none"), w).item(),
                "mc_bce": wmean(F.binary_cross_entropy_with_logits(qa, o, reduction="none"), w).item(),
                "pi_match": wmean((logits.argmax(1) == a).float(), w).item(),
                "q_match": wmean((self.q(s, m).argmax(1) == a).float(), w).item(),
                "v_mean": self.v(s).mean().item()}


# ── driver ──

def name_of(a):
    return f"cql_td_a{a.alpha}" if a.algo == "cql" else f"iql_td_t{a.tau}_b{a.beta}"


def train(a):
    dev = torch.device("cuda")
    torch.manual_seed(42)
    name = name_of(a)
    os.makedirs(OUT, exist_ok=True)
    final, state_path = os.path.join(OUT, f"{name}.pt"), os.path.join(OUT, f"{name}.state.pt")
    log_path = os.path.join(OUT, f"{name}.json")
    if os.path.exists(final):
        print(f"{name} done already")
        return
    t0 = time.time()
    tr, te = Data("train", dev), Data("test", dev)
    print(f"{name}: {tr.n:,} train / {te.n:,} test rows, halves with inconsistent pick outcomes: {tr.bad_halves}/{te.bad_halves}, "
          f"terminal share {float((tr.nxt < 0).float().mean()):.4f}, loaded {time.time() - t0:.0f}s", flush=True)
    alg = CQL(a.alpha, dev) if a.algo == "cql" else IQL(a.tau, a.beta, dev)
    log, start = [], 0
    if os.path.exists(state_path):
        ck = torch.load(state_path, map_location=dev)
        for k, net in alg.nets.items():
            net.load_state_dict(ck["nets"][k])
        for k, opt in alg.opts.items():
            opt.load_state_dict(ck["opts"][k])
        torch.set_rng_state(ck["rng"])
        torch.cuda.set_rng_state(ck["cuda_rng"])
        log, start = ck["log"], ck["epoch"]
        print(f"resumed at epoch {start}", flush=True)
    for ep in range(start, a.epochs):
        te0 = time.time()
        for net in alg.nets.values():
            net.train()
        alg.qt.eval()
        acc, nb = {}, 0
        for b in tr.batches(True, a.limit):
            r = alg.step(*b)
            for k, v in r.items():
                acc[k] = acc.get(k, 0.0) + v
            nb += 1
        for net in alg.nets.values():
            net.eval()
        ev, ne = {}, 0
        for b in te.batches(False, a.limit and a.limit // 4):
            r = alg.evaluate(*b)
            for k, v in r.items():
                ev[k] = ev.get(k, 0.0) + v
            ne += 1
        rec = {"epoch": ep + 1, "train": {k: v / nb for k, v in acc.items()},
               "test": {k: v / ne for k, v in ev.items()}, "minutes": (time.time() - te0) / 60}
        log.append(rec)
        print(json.dumps(rec), flush=True)
        torch.save({"nets": {k: n.state_dict() for k, n in alg.nets.items()},
                    "opts": {k: o.state_dict() for k, o in alg.opts.items()},
                    "rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                    "log": log, "epoch": ep + 1}, state_path + ".tmp")
        os.replace(state_path + ".tmp", state_path)
    if a.limit:
        print("smoke run: no final checkpoint written")
        return
    torch.save(alg.policy_state(), final + ".tmp")
    os.replace(final + ".tmp", final)
    json.dump({"name": name, "algo": a.algo, "alpha": a.alpha, "tau": a.tau, "beta": a.beta,
               "epochs": a.epochs, "batch": BATCH, "lr": LR, "polyak": POLYAK, "log": log,
               "minutes": (time.time() - t0) / 60}, open(log_path, "w"), indent=1)
    print(f"{name} done in {(time.time() - t0) / 60:.1f} min", flush=True)


def check():
    for split in ("test", "train"):
        d = os.path.join(CACHE, f"cql_naive_{split}", "packed")
        outs = np.load(os.path.join(d, "outcomes.npy"))
        step = np.load(os.path.join(d, "step.npy"))
        acts = np.load(os.path.join(d, "actions.npy")).astype(np.int64)
        nxt, bad = successors(outs, step)
        # successor is the same team: same outcome, later step, and the hero it picked is now taken
        has = nxt >= 0
        assert (step[nxt[has]] > step[has]).all()
        PICK = np.isin(step, [4, 5, 6, 7, 8, 11, 12, 13, 14, 15])
        pp = has & PICK & PICK[np.where(has, nxt, 0)]
        print(json.dumps({"split": split, "pick_to_pick_same_outcome": float((outs[nxt[pp]] == outs[pp]).mean())}))
        print(json.dumps({"split": split, "rows": len(outs), "halves_bad_pick_outcomes": bad,
                          "terminal_rows": int((~has).sum()), "terminal_per_half": float((~has).sum() / (len(outs) / 16)),
                          "successor_gap_hist": np.bincount(step[nxt[has]] - step[has]).tolist()}))
        bits = np.load(os.path.join(d, "bits.npy"), mmap_mode="r")
        rng = np.random.default_rng(0)
        idx = rng.choice(np.where(has)[0], 2000, replace=False)
        idx.sort()
        s = unpack(torch.from_numpy(np.asarray(bits[idx])), torch.from_numpy(step[idx])).numpy()
        s2 = unpack(torch.from_numpy(np.asarray(bits[nxt[idx]])), torch.from_numpy(step[nxt[idx]])).numpy()
        taken = s2[np.arange(len(idx)), acts[idx]] + s2[np.arange(len(idx)), 90 + acts[idx]] + s2[np.arange(len(idx)), 180 + acts[idx]]
        grows = (s2[:, :270] >= s[:, :270]).all(1)
        same_ctx = (s2[:, 270:287] == s[:, 270:287]).all(1)
        # the side's pick lands in the same team slot at every pick step of the half
        H = np.asarray(bits[:16 * 4000]).reshape(4000, 16, 36)
        Hs = np.asarray(step[:16 * 4000]).reshape(4000, 16)
        cnt = np.stack([unpack(torch.from_numpy(H[h]), torch.from_numpy(Hs[h])).numpy()[:, :180].reshape(16, 2, 90).sum(2)
                        for h in range(4000)])  # (halves, 16, 2)
        inc = np.diff(cnt, axis=1)  # increments after steps 0..14
        slot = inc.argmax(2)
        picks = [4, 5, 6, 7, 8, 11, 12, 13, 14]
        ok = np.ones(4000, bool)
        for k in picks:
            for k2 in picks:
                if SIDE[k] == SIDE[k2]:
                    ok &= slot[:, k] == slot[:, k2]
        print(json.dumps({"split": split, "pick_slots_consistent_with_order": float(ok.mean())}))
        print(json.dumps({"split": split, "action_present_in_successor": float((taken > 0).mean()),
                          "successor_superset": float(grows.mean()), "same_map_tier": float(same_ctx.mean())}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("algo", choices=["cql", "iql", "check"])
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=0.7)
    ap.add_argument("--beta", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--limit", type=int, default=None, help="rows per epoch (smoke test)")
    a = ap.parse_args()
    check() if a.algo == "check" else train(a)
