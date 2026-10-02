"""
Phase 1: retrain every model the paper reports, from the phase0 caches.

GPU-pool queue in the run_all_experiments.py style: one subprocess per job,
one GPU each via CUDA_VISIBLE_DEVICES, refill on completion. NUM_GPUS env
(default 4); the last GPU (GPU 3, thermal throttling) only receives jobs
tagged "light" unless nothing else remains.

All checkpoints land in rerun2026/models/ — historical training/*.pt files
are never touched. Per-job logs in rerun2026/logs/.

Usage:
    python rerun2026/phase1_models.py --dry-run
    python rerun2026/phase1_models.py
    python rerun2026/phase1_models.py --only cql_naive --force
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rerun2026 import common
from rerun2026.common import Job, MODELS_DIR

RUNNER = os.path.join(common.RERUN_DIR, "train_jobs.py")

CQL_ALPHAS = [0.1, 0.5, 1.0, 2.0, 5.0]
CQL_HP_TAUS = [0.001, 0.005, 0.01, 0.05, 0.1]
CQL_HP_ARCHS = ["1024,512,256", "512,256,128", "256,128,64"]
MCQ_TAUS = [0.4, 0.5, 0.6, 0.7, 0.8]
BC_WEIGHTS = [0.1, 0.5, 1.0, 2.0]
WR_SWEEP = [0.0, 5.0, 10.0, 50.0]


def mpath(*p):
    return os.path.join(MODELS_DIR, *p)


def build_jobs():
    jobs = []

    def add(name, argv, outputs, weight="heavy"):
        jobs.append(Job(name, [RUNNER] + argv, outputs, weight=weight))

    # ── Full-draft WP variants (Table I, cross-eval, MCTS value functions) ──
    # naive/herostrength/enriched_256 use best-of-3 seeds (Table I protocol)
    add("wp_naive", ["wp", "--name", "wp_naive", "--groups", "naive",
                     "--arch", "256,128", "--seeds", "3"],
        [mpath("wp_naive.pt")], weight="light")
    add("wp_herostrength", ["wp", "--name", "wp_herostrength", "--groups",
                            "herostrength", "--arch", "256,128", "--seeds", "3"],
        [mpath("wp_herostrength.pt")], weight="light")
    add("wp_enriched_256", ["wp", "--name", "wp_enriched_256", "--groups",
                            "enriched", "--arch", "256,128", "--seeds", "3"],
        [mpath("wp_enriched_256.pt")], weight="light")
    add("wp_enriched_512", ["wp", "--name", "wp_enriched_512", "--groups",
                            "enriched", "--arch", "512,256,128"],
        [mpath("wp_enriched_512.pt")], weight="light")
    # true_base: identical features to naive, separate run (MCTS K_truebase input)
    add("wp_true_base", ["wp", "--name", "wp_true_base", "--groups", "true_base",
                         "--arch", "256,128"],
        [mpath("wp_true_base.pt")], weight="light")

    # ── Augmented WP: WR sweep (Table V / fig_wr_sweep), flat generator,
    #    unseen-only, volume 100, 512->256->128, train seed 42 (the paper's
    #    "5 seeds" are evaluation seeds — phase3 wr-sweep runs them) ──
    for wr in WR_SWEEP:
        name = f"wp_aug_wr{int(wr)}_vol100_512"
        add(name, ["wp_aug", "--name", name, "--wr", str(wr), "--volume", "100",
                   "--scope", "tier2_only", "--arch", "512,256,128",
                   "--generator", "flat"],
            [mpath(f"{name}.pt")], weight="light")

    # ── Augmented WP: v2 pairwise-adjusted generator (deployed models) ──
    # 512 arch == historical wp_enriched_winner.pt (Table VII "Enr.+aug")
    add("wp_aug_v2_512", ["wp_aug", "--name", "wp_aug_v2_512", "--wr", "10",
                          "--volume", "100", "--arch", "512,256,128",
                          "--generator", "v2"],
        [mpath("wp_aug_v2_512.pt")], weight="light")
    # 256 arch == historical wp_augmented.pt (MCTS H_augmented value function)
    add("wp_aug_v2_256", ["wp_aug", "--name", "wp_aug_v2_256", "--wr", "10",
                          "--volume", "100", "--arch", "256,128",
                          "--generator", "v2"],
        [mpath("wp_aug_v2_256.pt")], weight="light")

    # ── Scope comparison (Table IV): wr=10, vol=100, 256->128, flat generator.
    #    Baseline row = wp_enriched_256 (no augmentation). ──
    for scope in ["tier1_only", "tier2_only", "both"]:
        name = f"wp_scope_{scope}_256"
        add(name, ["wp_aug", "--name", name, "--wr", "10", "--volume", "100",
                   "--scope", scope, "--arch", "256,128", "--generator", "flat"],
            [mpath(f"{name}.pt")], weight="light")

    # ── Step-conditioned WP (replay-level split = leakage fix, Section VI-C) ──
    add("partial_wp", ["partial", "--name", "partial_wp", "--mode", "partial"],
        [mpath("partial_wp.pt")])
    add("base_partial_wp", ["partial", "--name", "base_partial_wp", "--mode", "base"],
        [mpath("base_partial_wp.pt")])
    # Leakage-bug reproductions (sample-level split) — regenerate the paper's
    # "inflated to 62.5%/62.7%" numbers for contrast
    add("partial_wp_leaky", ["partial", "--name", "partial_wp_leaky",
                             "--mode", "partial", "--sample-split"],
        [mpath("partial_wp_leaky.pt")])
    add("base_partial_wp_leaky", ["partial", "--name", "base_partial_wp_leaky",
                                  "--mode", "base", "--sample-split"],
        [mpath("base_partial_wp_leaky.pt")])

    # ── Generic Draft behavioral cloning pool (5 seed variants) ──
    for i in range(5):
        add(f"gd_{i}", ["gd", "--variant", str(i)],
            [mpath(f"generic_draft_{i}.pt")])

    # ── Gourdeau reproduction ──
    add("gourdeau_wp", ["gourdeau"], [mpath("gourdeau_wp.pt")], weight="light")
    add("wp_independent_siamese", ["siamese"],
        [mpath("wp_independent_siamese.pt")], weight="light")

    # ── CQL naive alpha sweep (Table tab:cql_basic / Table VII rows) ──
    for a in CQL_ALPHAS:
        add(f"cql_naive_a{a}", ["cql_naive", "--alpha", str(a)],
            [mpath("cql", f"_cql_temp_a{a}.pt")])

    # ── CQL enriched alpha sweep (Table VII "CQL enr" rows use 0.5 / 2.0) ──
    for a in CQL_ALPHAS:
        add(f"cql_enr_a{a}", ["cql_enr", "--alpha", str(a)],
            [mpath("cql", f"_cql_enriched_a{a}.pt")])

    # ── CQL hyperparameter grid (Table VIII + tau-sweep sentence) ──
    for arch in CQL_HP_ARCHS:
        for tau in CQL_HP_TAUS:
            arch_str = arch.replace(",", "x")
            add(f"cql_hp_t{tau}_{arch_str}",
                ["cql_hp", "--alpha", "1.0", "--tau", str(tau), "--arch", arch],
                [mpath("cql_hyperparams", f"_cql_temp_a1.0_t{tau}_{arch_str}.pt")])

    # ── MCQ / BC-CQL sweeps (Section VII) ──
    for t in MCQ_TAUS:
        add(f"mcq_t{t}", ["mcq", "--threshold", str(t)],
            [mpath("mcq", f"_mcq_temp_t{t}.pt")])
    for b in BC_WEIGHTS:
        add(f"bccql_b{b}", ["bccql", "--bc-weight", str(b)],
            [mpath("mcq", f"_bc_cql_temp_bc{b}.pt")])

    return jobs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", default=None, help="substring filter on job name")
    parser.add_argument("--num-gpus", type=int, default=None)
    args = parser.parse_args()

    common.setup()
    jobs = build_jobs()
    print(f"Phase 1: {len(jobs)} jobs defined")

    if not args.dry_run:
        missing = [p for p in [common.FULL_TRAIN_NPZ, common.FULL_TEST_NPZ]
                   + list(common.STEP_NPZ.values())
                   if not os.path.exists(p)]
        missing += [d for d in [common.CQL_NAIVE_TRAIN, common.CQL_ENR_TRAIN,
                                common.GD_TRAIN]
                    if not os.path.exists(os.path.join(d, "meta.json"))]
        if missing:
            print("Phase 0 caches missing — run phase0_features.py first:")
            for m in missing:
                print(f"  {m}")
            sys.exit(1)

    failed = common.run_pool(jobs, num_gpus=args.num_gpus, dry_run=args.dry_run,
                             force=args.force, only=args.only)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
