"""Scores a td_tour.py row (TD_NAME) with search_score.score_tournament's judges
and conventions (v1 and composition-corrected v2, config-clustered SEs), against
the stored scores of the existing 110 pairs. Output:
site/results/td/<name>/tournament_standings.json.
Usage (from training/, p1site env): TD_NAME=cql_td_a1.0 python3 paper1_revision/td_score.py
"""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from paper1_revision import core
from paper1_revision import search_eval as se
from paper1_revision import search_score as ss
from paper1_revision import tournament as T

NAME = os.environ["TD_NAME"]
D = os.path.join(core.RESULTS, "td", NAME)
se.NEW = {NAME: None}
se.PRIMARY = []
se.pairs = lambda: [(NAME, o) for o in T.STRATEGIES] + [(o, NAME) for o in T.STRATEGIES]
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
