"""Shared constants for the scientific-learning harness (paper 3).

Spec: paper/hypothesis_system/FRAMING.md. Everything a proposer may see goes through
the broker; the private store below is never mounted into a proposer sandbox.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CODE = REPO / "training" / "scilearn"
DOCS = REPO / "paper" / "hypothesis_system"

# Broker-owned private store: snapshots with labels, keys, raw answers, ledger.
PRIVATE = Path(os.environ.get("SCILEARN_PRIVATE", "/home/max/scilearn-private"))
SNAPSHOTS = PRIVATE / "snapshots"
LEDGER_DIR = PRIVATE / "ledger"
RAW_STORE = PRIVATE / "raw"
SEALED_RESULTS = PRIVATE / "sealed_results"
KEYS_FILE = PRIVATE / "keys.json"
FITS_DIR = PRIVATE / "fits"
ARCHIVE_BACKUP = Path("/archive-backup/scilearn")

# Per-session dirs (work dir, broker socket, proxy socket, transcripts).
SESSIONS = Path(os.environ.get("SCILEARN_SESSIONS", "/home/max/scilearn-sessions"))

REPLAYS_ROOT = Path("/home/max/hots-replays")
SEALED_IDS_FILE = REPLAYS_ROOT / "sealed" / "sealed_ids.txt"
SEALED_RULE_FILE = REPO / "training" / "replays" / "cohorts" / "sealed_rule.json"
REPLAY_MANIFEST = REPLAYS_ROOT / "manifest.jsonl"
TIMELINES = REPLAYS_ROOT / "timelines"

# Snapshot cut: FRAMING section 4.1 counts are "through 2026-10-12"; day 1 snapshot is as of Oct 13.
SNAPSHOT_AS_OF = "2026-10-13"

# Tranches by build (FRAMING 4.1). Builds are compared as integer tuples.
DPRE_EXTRA_BUILDS = {"2.55.16.96846", "2.55.16.96870", "2.55.16.96881"}
G_BUILDS = {"2.55.16.97039"}
C_END_DATE = "2026-10-27"          # C = 2.57 games played 09-28 .. 10-26
F_BLOCK_START = "2026-10-27"       # F_k = 4-week blocks from here
F_BLOCK_DAYS = 28
POOL_FRAC = 0.10                   # interesting pool, keyed hash of non-sealed 2.57
M_EXPLORE_FRAC = 0.30              # M split, keyed hash
AUDIT_FRAC = 0.10                  # same-era audit share of F arrivals

TRANCHES = ("S", "Dpre", "G", "Dpost", "M", "pool", "R", "C", "F", "audit")
QUARANTINED = {"R", "pool"}        # never in any view, cache, history or online state
NO_QUERY = {"C", "F", "R", "audit", "pool", "G"}  # no proposer query class at all

# Budgets per session per cycle (FRAMING 5.3).
BUDGET = {
    "sandbox_numbers": 5000,
    "dev_bits": 120.0,
    "group_bits": 8.0,
    "cards": 40,
    "canaries": 6,
    "mediator_bits": 60.0,
    "inspect": 300,
    "narrate": 100,
    "q4_per_cycle": 3,
    "q6_snapshots": 30,
    "q6_max_coefs": 20000,
}

# Query release rules (FRAMING 5.2).
Q0_MIN_CELL = 50
Q1_S_MIN_CELL, Q1_S_ROUND = 200, 0.001
Q1_DEV_MIN_CELL, Q1_DEV_ROUND, Q1_DEV_KAPPA = 500, 0.005, 2.0
Q1_MAX_DIMS, Q1_MAX_CELLS = 2, 20
Q2_BITS = 1.5
Q2_ROUND = 0.0001
Q4_BITS, Q4_BIN, Q4_MIN_GAIN = 3.7, 0.0005, 0.0005
Q3_MIN_CELL = 200

# Dev gate (FRAMING 5.4).
GATE_Z0 = 3.0
GATE_MARGIN = 0.59                 # half of Russo-Zou 1.18
GATE_NEAR_MISS = 2.0

# Model caps (FRAMING 6.1).
MAX_PARAMS = 20000
MAX_INTERACTIONS = 30
RULE_MAX_BETA = 0.15
RULE_MIN_S_GAMES = 5000

# Remote workers (WAVE0_BRIEF item 6). Never the 3080 (lent to tennis).
REMOTE_HOSTS = {
    "max-windows-3090": {"cpus": "24-35", "threads": 12, "mem_max": "56G"},
    "windows-5090-wsl": {"cpus": None, "threads": 6, "mem_max": "11G"},
}
REMOTE_ROOT = "/home/max/hots/scilearn"   # inside WSL, apart from ~/hots/repo data

SANDBOX_IMAGE = "scilearn-sandbox:1"
