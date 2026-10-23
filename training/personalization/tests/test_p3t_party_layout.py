"""P3 party drafter: who holds the bans and who locks when (p3t_party.layout, plan, KnownState.place).
  P1 the ban holder is the team's top-ranked player: "ban" parties contain him, "noban" parties
     never do, random parties hold the bans exactly when he is a member
  P2 fixed timing: the party sits on the chosen team turns, the strangers on the others, the
     opponents keep the real order, every team turn is taken once
  P3 common random numbers: configurations sharing a crn key have the same members and the same
     stranger locking order, so a solo user at turn q and the lock plan see the same strangers
  P4 the grid has every turn set of every party size on both sides, with and without the top-ranked
     player (no "noban" for a party of five), and the lock plan is solo
  P5 KnownState.place moves a player to a turn and the displaced player to his old turn
Run (a host with the project env): python3 personalization/tests/test_p3t_party_layout.py"""
import os
import sys
import tempfile

TMP = tempfile.mkdtemp(prefix="p3t_party_")
os.environ["P3_CACHE"] = os.path.join(TMP, "cache")
os.environ["P3_RESULTS"] = os.path.join(TMP, "results")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import p3t_party as PT


class FakeLB:
    def __init__(self, n, seed=0):
        rng = np.random.RandomState(seed)
        self.n = n
        # real pick order: kernel slots 0-4 (team 0) and 5-9 (team 1) take their team's turns in a random order
        self.src = np.zeros((n, 10), np.int64)
        for i in range(n):
            for t in (0, 1):
                self.src[i, 5 * t:5 * t + 5] = np.array(PT.TEAM_TURNS[t])[rng.permutation(5)]


def check_team_turns(turn_of, t):
    for i in range(len(turn_of)):
        assert sorted(turn_of[i, 5 * t:5 * t + 5]) == PT.TEAM_TURNS[t]


def main():
    n = 200
    LB = FakeLB(n)
    rng = np.random.RandomState(1)
    top = np.stack([rng.randint(0, 5, n), 5 + rng.randint(0, 5, n)], 1)
    ar = np.arange(n)
    # P1
    for t in (0, 1):
        for m in (1, 2, 3, 4):
            for hold in ("ban", "noban", None):
                cfg = dict(name="x", t=t, m=m, turns=None if hold is None else list(range(m)), hold=hold,
                           crn=f"m{m}_t{t}_{hold}")
                turn_of, party, ban_ctl, sord = PT.layout(LB, cfg, top)
                assert (party[:, 5 * t:5 * t + 5].sum(1) == m).all() and not party[:, 5 * (1 - t):5 * (1 - t) + 5].any()
                assert (ban_ctl == party[ar, top[:, t]]).all()
                if hold == "ban":
                    assert ban_ctl.all()
                elif hold == "noban":
                    assert not ban_ctl.any()
                else:
                    assert 0.05 < ban_ctl.mean() < 0.95
    print("P1 ok")
    # P2
    for t in (0, 1):
        for turns in ([0], [4], [1, 3], [0, 2, 4]):
            cfg = dict(name="x", t=t, m=len(turns), turns=turns, hold="noban", crn="k")
            turn_of, party, _, _ = PT.layout(LB, cfg, top)
            check_team_turns(turn_of, t)
            assert (turn_of[:, 5 * (1 - t):5 * (1 - t) + 5] == LB.src[:, 5 * (1 - t):5 * (1 - t) + 5]).all()
            want = sorted(PT.TEAM_TURNS[t][q] for q in turns)
            for i in range(n):
                assert sorted(turn_of[i, party[i]]) == want
    print("P2 ok")
    # P3
    for t in (0, 1):
        for hold in ("ban", "noban"):
            lay = {q: PT.layout(LB, dict(name="x", t=t, m=1, turns=[q], hold=hold, crn=f"m1_t{t}_{hold}"), top)
                   for q in range(5)}
            lk = PT.layout(LB, dict(name="x", t=t, m=1, turns="lock", hold=hold, crn=f"m1_t{t}_{hold}"), top)
            for q in range(5):
                assert (lay[q][1] == lk[1]).all()                       # same user
                assert (lay[q][3][:, :4] == lk[3][:, :4]).all()         # same stranger order
                # at turn q the user locks; the k-th stranger takes the k-th other turn
                tt = PT.TEAM_TURNS[t]
                other = [tt[x] for x in range(5) if x != q]
                for i in range(0, n, 17):
                    for s_, j in zip(lk[3][i, :4], other):
                        assert lay[q][0][i, s_] == j
            assert (lk[3][:, 4] == np.argmax(lk[1], 1)).all()           # the user is last in sord
    print("P3 ok")
    # P4
    g = PT.plan("grid")
    names = {c["name"] for c in g}
    assert len(g) == 2 * (2 * 31 - 1) and len(names) == len(g)
    assert "grid_m5_01234_t0_ban" in names and "grid_m5_01234_t0_noban" not in names
    assert all(c["m"] == 1 for c in PT.plan("lock")) and len(PT.plan("lock")) == 4
    assert len(PT.plan("effect")) == 10
    print("P4 ok")
    # P5
    KS = PT.KnownState.__new__(PT.KnownState)
    KS.B = n
    KS.turn_of = LB.src.copy()
    KS.perm = np.argsort(KS.turn_of, 1)
    j = PT.TEAM_TURNS[0][0]
    slots = rng.randint(0, 5, n)
    old_turn = KS.turn_of[ar, slots].copy()
    occ = KS.perm[:, j].copy()
    KS.place(j, slots)
    assert (KS.perm[:, j] == slots).all() and (KS.turn_of[ar, slots] == j).all()
    assert (KS.turn_of[ar, occ] == np.where(occ == slots, j, old_turn)).all()
    assert (np.argsort(KS.turn_of, 1) == KS.perm).all()
    print("P5 ok")
    print("ALL OK")


if __name__ == "__main__":
    main()
