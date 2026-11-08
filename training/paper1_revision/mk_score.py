"""Scores the masked featureless MCTS (constrained_k) tournament row with search_score.score_tournament's
judges and conventions (v1 and composition-corrected v2, config-clustered SEs),
against the stored scores of the existing 110 pairs. Output:
site/results/masked_k/tournament_standings.json.
Usage (from training/, p1site env, where the judges live): python3 paper1_revision/mk_score.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core
from paper1_revision import search_eval as se
from paper1_revision import search_score as ss
from paper1_revision import tournament as T

D = os.path.join(core.RESULTS, "masked_k")
se.NEW = {"constrained_k": None}
se.PRIMARY = []
se.pairs = lambda: [("constrained_k", o) for o in T.STRATEGIES] + [(o, "constrained_k") for o in T.STRATEGIES]
se.pair_path = lambda a, b: os.path.join(D, "tournament", f"{a}__{b}.json")
ss.OUT = D
# the existing-pairs gN cache is shared with the search rows (same 110 pairs)
_orig_existing = ss._existing
def _existing():
    ss.OUT = se.OUT
    try:
        return _orig_existing()
    finally:
        ss.OUT = D
ss._existing = _existing

if __name__ == "__main__":
    ss.score_tournament()
