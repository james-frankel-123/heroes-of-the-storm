# Kit table (kit_2.57)

Hero kit attributes for the scientific-learning harness (paper 3). Spec: `paper/hypothesis_system/FRAMING.md` section 10.1, WAVE0_BRIEF item 9.

`kit_2.57.json` holds 12 mechanics-only axes for each of the 91 heroes in the 2.57 pool, coded by three independent LLM runs from hero-blind ability text. Values are fixed: they never move during model fits.

## Axes

hard_cc, soft_cc_peel, healing_throughput, self_sustain_shields, burst, sustained_damage, range, mobility, waveclear, siege, global_presence (all 0 to 4), early_vs_late (-2 early to +2 late). Definitions and anchors are in `RUBRIC.md`.

Team identities are fixed linear combinations (weights in `kitdefs.py` and in the kit file):

| Identity | Weights |
|---|---|
| poke | 0.50 range + 0.30 burst + 0.20 sustained_damage |
| dive | 0.35 mobility + 0.25 hard_cc + 0.25 burst + 0.15 self_sustain_shields |
| sustain | 0.50 healing_throughput + 0.30 self_sustain_shields + 0.20 soft_cc_peel |
| push | 0.40 siege + 0.35 waveclear + 0.25 global_presence |

A team's identity is the mean of its five heroes' scores.

## How it was built

1. `fetch_sources.py <download_dir>`: ability and talent text from heroespatchnotes/heroes-talents at the pinned commit `2dcbbf6` (same pin as `training/replays/build_reference.py`; data to Nov 2023), plus every live or hotfix patch-note change after that through build 2.57.0.98348, taken from the per-hero change histories on hotspatchnotes.com (PTR, bug fixes and datamined blocks dropped). Basic attack range, rate and damage come from HeroesToolChest/heroes-data (build 2.55.9.93640). Xal'atath is not in the pinned data: her kit is the client-extracted 2.57.0.98348 text, checked against the official live notes and the Oct 5 hotfix. Output: `sources_2.57.json`. Raw HTML and JSON downloads are kept outside the repo.
2. `blind.py`: replaces the hero's name, other heroes' names, ability and talent names (slot tokens like `[Q]`, `[R1]`, `[T4.2]`) and identifying summon names (`summon_names.json`) with neutral tokens. Numbers are kept. Writes `blind_text/<HID>.txt`. HIDs come from a random permutation stored in `hid_map_PRIVATE.json` (also copied into the kit file's provenance). `python3 -I blind.py --scan` lists leftover rare capitalised terms.
3. `run_coders.py`: three coders, each with its own seeded ordering of HIDs, in batches of 13. Each batch is a fresh `claude -p --model claude-opus-5-5` call (account 2) with no tools, no MCP, no session persistence and an empty working directory. JSON is validated and malformed answers retried. Prompts are in `codings/prompts/`, raw answers in `codings/coder<c>_batch<b>.json`.
4. `build_kit.py`: averages the three coders, records per-axis SD across coders, flags axes with SD > 1.0, computes identities, attaches roles (from `src/lib/data/hero-roles.ts`) and provenance, and hashes the canonical JSON. `python3 -I build_kit.py --check` rebuilds and confirms the stored hash.
5. `validate.py`: `within_role_permutation_check(kit, axis, score)` takes the harness's scoring callable (hero -> axis value mapping in, float out) and returns the share of 1000 within-role permutations the real values beat. `python3 -I validate.py --spotcheck` writes `SPOTCHECK.md` (15 heroes by fixed seeded hash) for Max.

No win rates, pick rates or database outcome data were used, and the database was not queried. The hotspatchnotes pages also show win rates and tiers; only their kit and change-history blocks are parsed.

## Known limits

- Ability and summon names were replaced, but mechanics are distinctive, so a coder could still recognise some heroes. FRAMING 13 item 3 covers this; the behavioral and permutation checks narrow it.
- Rework deltas are appended as change lists after the 2023 base text, not merged into it; coders are told later changes win.
- Basic attack data is from 2.55.9 (Feb 2025); later changes to it appear only through the patch-note deltas.
- Xal'atath has no basic attack range or rate in the sources (only "ranged"). Gall has no basic attack of his own.
