# Wave 0 brief: build the scientific-learning harness

You are the infrastructure session for **paper 3, "scientific learning"**: a new machine-learning method in which LLM agents propose theory-backed hypotheses and an enforced harness tests them under an information budget. The model's functional form is hill-climbed by hypothesis testing, not gradient descent. It is benchmarked against our deep-learning win-probability models (papers 1 and 2), but it is a separate project.

**The spec is `paper/hypothesis_system/FRAMING.md`. Read it fully first; it is binding.** Where this brief and the spec differ, this brief wins.

## Build (wave 0, §11 of the spec)
1. **Data layer and tranche manager.**
   - Use the spec's tranches (sandbox, pre-cutoff dev, gap, post-cutoff dev, confirmation C, prospective F, sealed replay holdout).
   - Materialize them as read-only snapshots from the local Postgres research role (`DATABASE_URL_RESEARCH` in `.env`; NEVER `NEON_DATABASE_URL`). Use replay timelines from `training/replays/` and `/home/max/hots-replays/` (the replay lane's output).
   - Quarantine sealed-range rows as the spec says.
2. **Query broker + information budget + append-only ledger**, per §5.
   - The broker is the ONLY path from a proposer to outcome data.
   - Ledger entries are hash-chained, so tampering shows.
3. **Enforcement.** Proposer sessions must not be able to bypass the broker. Passwordless sudo is NOT available, so no new Unix users. Use `bwrap` (`/home/linuxbrew/.linuxbrew/bin/bwrap`) or Docker (the user is in the `docker` group) to run proposer sessions with:
   - no `.env`;
   - no DB socket or credentials;
   - no label files;
   - no SSH keys;
   - a read-only view of the code and docs they need;
   - broker access only over a local socket or HTTP endpoint.

   Prove it with tests: a sandboxed process trying to read `.env`, connect to Postgres, read a label file or SSH to a worker must fail.
4. **Model language and fitting backends**, per §6.
   - The restricted term types.
   - Conventional MLE or Bayesian fitting with shrinkage.
   - The as-of contract and the self-inclusion ban in code, with planted-leak fixtures that must be caught.
5. **Evaluator and acceptance gate**, per §7 and §8. Out of the proposers' reach.
   - Controls: shuffled outcomes, hero-blinded, hidden canaries.
   - The prior-knowledge arms (a, b, c).
6. **Remote dispatch.** All fitting and scoring compute runs on the remote workers, not this box:
   - 3090 `max-windows-3090`: our cores are 24–35 only; the tennis lane has 0–23 until ~Oct 18. Stay under ~60 GB RAM there.
   - 5090 `windows-5090-wsl`: CPU threads only; paper 1 uses its GPU. Stay under ~12 GB RAM.
   - The 3080 is lent to tennis; don't use it.

   Use the job tooling in `training/remote_workers/` (hotsjob, `run_remote.sh`, `hosts.py`, memmon). This box does only light orchestration, the broker, the ledger and git.
7. **Baselines**, per the spec:
   - an LLM-free additive baseline (EBM/GA2M or a hierarchical additive logit) fitted on the proper tranches;
   - the score of the frozen neural WP (the paper-1 site leak-free enriched WP, and the drift-aware model if feasible) on the same dev and OOD tests.

   Write these numbers to `paper/hypothesis_system/results/baselines.md`.
8. **MCTS-leaf adapter.** A way to plug a fitted scientific-learning WP into our MCTS search as the leaf (the CUDA kernel takes a WP network; a Python leaf in the CPU reference search `cuda_mcts/ref_search.py` is fine to start), and the exploitability test from the spec.
9. **Kit-attribute table** for all 91 heroes, coded from hero-blind ability text, per the spec. Store it with provenance.
10. **Session launcher** for wave-1 proposer sessions:
    - claude2 account: `CLAUDE_CONFIG_DIR=$HOME/.claude-account-2`, `claude -p --model claude-opus-5-5`, run inside the sandbox;
    - one budget per session;
    - a per-session working directory.

    Do NOT launch wave 1 yourself; the coordinator will, after review.

## Acceptance before wave 1
Write `paper/hypothesis_system/WAVE0_REPORT.md` with:
- **Sandbox escape tests:** all must fail to escape.
- **Planted-leak tests:** all caught.
- **A dry-run hypothesis** going end to end through broker, fit, gate and ledger.
- **A shuffled-outcome dry run:** it must not accept the planted null.
- **The baseline numbers.**
- **Exact commands** to launch wave-1 sessions.

## Rules
- **No heavy compute on this box:** no GPU, nice 19, at most 4 cores for anything local.
- **Tennis machines:** never touch the tennis lane's processes or `~/tennis-*` dirs on the remotes. Never touch other lanes' jobs (paper 1, personalization, replay, production).
- **Public repo:** no secrets, battletags or player names in commits. Data and caches stay out of git. Code lives under `training/scilearn/` (or similar); docs under `paper/hypothesis_system/`.
- **Commits:** commit incrementally, your paths only, with explicit pathspecs. Don't push. End messages with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
- **Writing voice:** plain language, numbers first, no em-dashes.
- **Effort:** move fast. This is hours of agent work, not days.
