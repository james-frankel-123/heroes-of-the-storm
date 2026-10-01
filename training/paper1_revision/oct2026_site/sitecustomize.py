"""
Import-time patches for the oct2026 expert-study refresh (no existing file is
edited). Active only when the env vars are set; put this directory first on
PYTHONPATH.

  P1R_COMP_PATH   StatsCache._load_compositions loads the own-corpus
                  role-composition table at this path (compositions.json
                  schema) instead of the external Heroes Profile table.
  P1R_MCTS_CKPT   rerun2026.constrained_search.load_mcts_policy defaults to
                  this checkpoint (its built-in default is the namespace's
                  J_800sim_s9, which the oct2026 run does not train).
  P1R_DROP_AUG=1  rerun2026.phase3b_roundrobin drops the synthetic-augmentation
                  strategy (enriched_aug) from STRATEGIES and the "augmented"
                  evaluator from WP_EVALUATORS (expert study v6 roster, Max
                  2026-10-01). constrained_search builds its roster from
                  phase3b's list at import, so it inherits the change.
Patches apply on import by module name, so the patched modules must be
imported (not run as __main__); oct2026_tournament.py does that.
"""
import os
import sys
import importlib.abc
import importlib.util
import importlib.machinery

# Chain to the interpreter's own sitecustomize (Homebrew Python adds its
# site-packages there), which this module shadows on PYTHONPATH.
_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.machinery.PathFinder.find_spec(
    "sitecustomize", [p for p in sys.path if os.path.abspath(p or ".") != _here])
if _spec is not None and _spec.origin and os.path.abspath(_spec.origin) != os.path.abspath(__file__):
    _code = compile(open(_spec.origin).read(), _spec.origin, "exec")
    exec(_code, {"__name__": "_chained_sitecustomize", "__file__": _spec.origin})

_COMP = os.environ.get("P1R_COMP_PATH")
_MCTS = os.environ.get("P1R_MCTS_CKPT")


def _patch_swp(mod):
    import json

    def _load_compositions(self):
        raw = json.load(open(_COMP))
        self.comp_data = {t: {",".join(sorted(c["roles"])): (c["winRate"], c["games"])
                              for c in cs} for t, cs in raw.items()}
    mod.StatsCache._load_compositions = _load_compositions


def _patch_cs(mod):
    mod.MCTS_CHECKPOINT = _MCTS
    mod.load_mcts_policy.__defaults__ = (_MCTS,)


def _patch_p3b(mod):
    mod.STRATEGIES[:] = [x for x in mod.STRATEGIES if x != "enriched_aug"]
    mod.WP_EVALUATORS.pop("augmented", None)


_TARGETS = {}
if os.environ.get("P1R_DROP_AUG") == "1":
    _TARGETS["rerun2026.phase3b_roundrobin"] = _patch_p3b
if _COMP:
    _TARGETS["sweep_enriched_wp"] = _patch_swp
if _MCTS:
    _TARGETS["rerun2026.constrained_search"] = _patch_cs


class _Hook(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name not in _TARGETS:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(name)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return None
        orig = spec.loader.exec_module
        patch = _TARGETS.pop(name)

        def exec_module(module, _orig=orig, _patch=patch):
            _orig(module)
            _patch(module)
        spec.loader.exec_module = exec_module
        return spec


if _TARGETS:
    sys.meta_path.insert(0, _Hook())
