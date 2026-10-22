# P3: the party drafter (partial control), prototype on current data (2026-10-22)

Status: prototype on the current panel (store full_1017, CAP_TS 2026-09-28), the 300 tournament lobbies, kernels as of commit a2d5bad. It reruns on the definitive inputs after Oct 29 with the definitive player-state model. Model-judged numbers are labeled J (player-aware judge), J_cal (calibrated parts) and CONS (population consensus); the judge-free record uses later real games. Every interval is a 95% lobby-bootstrap CI over the 300 lobbies (paired by lobby between arms; both team seats pooled = 600 lobby-seats).

## The answer

- **A solo player using the drafter gains +2.8pp of judged win probability (J, CI 2.1 to 3.5)** over picking like a typical player, about **a fifth (19%, CI 14 to 25)** of what a full team of five gains (+14.5pp). A duo gets a third (+4.8pp), a trio about half (+6.9pp), a four-stack 78% (+11.3pp). The gain grows a little faster than the number of controlled seats once the party is four.
- Against the player-blind drafter controlling the same seats, the player-aware drafter wins by +3.5pp (solo) to +17.3pp (full team). The blind drafter is worse than typical picks under J (it raises the population win probability but puts players on heroes they are bad at), so a party should not use a player-blind tool.
- Against the party members simply playing what they played in the real game, the drafter adds +1.8pp (solo) to +10.2pp (five).
- **The best slot for a solo user is the team's first pick, because that player also bans** (+3.7pp). Holding the bans is worth about +2.3pp for a solo user and +2.6 to +4.0pp for a party. **Without the bans, later is better**: the first pick gains +1.4pp, the second +1.8, the third +2.6, the fourth and fifth +3.1. Picking late does not secure the player's comfort hero better (his best hero is still free 91% of the time at the team's first pick and 77 to 80% at the last two), but it lets the drafter choose knowing the opponents' and teammates' picks; the drafter still lands the player on a strong personal hero late (his pick's skill +0.8 to +1.2pp, against −1.0 to −1.8pp for typical picks at the same turns).
- First-pick and second-pick teams behave the same within the noise.
- The judge-free check is too noisy at this scale: the realized records of the party's team differ from the typical-pick baseline by −0.2 to +1.4pp per player with CIs spanning zero. The judged gains are mostly personal (the CONS part of the gain is about half of J's), so they inherit J's attenuation; the true gains are likely larger, as in the main tournament.

## Setup

- **Lobbies:** the 300 real tournament lobbies (2026-05-23 to 06-30). The party sits on one team (both teams in turn), drawn at random among the team's players.
- **Pick order:** the team's five players take the team's five pick turns in a random order (the effectiveness study) or the party is put on chosen turns (the position study); the opponents keep the real game's order.
- **Strangers (teammates outside the party) and opponents:** the personalized generic drafter, our rebuilt DraftRec, for that specific player: his own history, the draft so far and his turn, sampled at temperature 1 from the heroes his team can play. Opponents' bans: GD argmax.
- **Bans (assumption):** the player on the team's first pick turn bans for the team (Storm League gives the team's bans to the first player in its draft order). The party controls the bans exactly when it holds that turn; otherwise the team's bans are GD argmax. The rows marked "bans left to GD" put the party on the first turn without the bans, to separate the two effects.
- **The drafter for the party (aware):** the selected tournament design (`x-combo1-dr-chance`: distilled player-feature prior, denial bans, the DraftRec opponent at chance nodes, 400 simulations). In party mode:
  - each stranger's turn is a chance node in the search, with that stranger's own DraftRec distribution read at the decision state, exactly like an opponent's turn (kernel flag `cfg[40 + step] = -2`, `cuda_explore`);
  - because the pick order is known, each hero is valued for the player who picks it (slot mode), not the team's best assignment, and the assignment-aware root bias is off;
  - determinism and parity gates pass: reruns are byte-identical, flags on past steps are a no-op, flags on later steps change the search, and every earlier parity check is unchanged (`p3t_determinism.py`, `p3t_explore_verify.py`).
- **Baselines for the same seats:** blind = the population-only search (paper 1's F prior and leaf, 400 sims), strangers and opponents as GD chance nodes; real = the hero each party member played in the real game when still legal (else as generic), and the real team's bans when legal; generic = the party picks like strangers (DraftRec).
- **Common random numbers:** for a given configuration every arm shares the pick order, the party, the kernel seeds and the strangers' and opponents' random draws.
- **Scoring:** the tournament's judges on the final drafts, with the TRUE hero-to-player map (each hero belongs to whoever picked it). J_cal rescales the consensus part by 1.17 (the held-out judge-window calibration). The judge-free record is each party-team player's realized record (y − WP) on his hero in later judge-window games.

## Effectiveness by party size (party seats random, both team seats pooled)

| party size | aware − generic: J | J_cal | CONS | judge-free record (pp per player) | share of the full-team gain (J) | aware − blind: J | aware − real picks: J |
|---|---|---|---|---|---|---|---|
| 1 | +2.8 (2.1, 3.5) | +3.0 (2.3, 3.8) | +1.6 (1.1, 2.0) | −0.2 (−1.3, 1.0) | 19% (14, 25) | +3.5 (2.8, 4.3) | +1.8 (1.1, 2.5) |
| 2 | +4.8 (3.9, 5.7) | +5.2 (4.3, 6.1) | +2.4 (1.9, 2.9) | +0.6 (−0.8, 2.0) | 33% (27, 40) | +7.5 (6.6, 8.4) | +4.5 (3.5, 5.5) |
| 3 | +6.9 (5.8, 7.9) | +7.5 (6.4, 8.5) | +3.7 (3.2, 4.3) | +0.7 (−0.8, 2.2) | 48% (40, 55) | +10.2 (9.2, 11.3) | +5.7 (4.7, 6.7) |
| 4 | +11.3 (10.3, 12.4) | +12.2 (11.1, 13.3) | +5.8 (5.2, 6.4) | +1.4 (−0.3, 3.2) | 78% (69, 88) | +13.8 (12.8, 14.9) | +7.7 (6.6, 8.8) |
| 5 | +14.5 (13.4, 15.7) | +15.6 (14.4, 16.8) | +7.3 (6.7, 8.0) | −0.2 (−1.9, 1.6) | 100% (100, 100) | +17.3 (16.3, 18.4) | +10.2 (9.2, 11.3) |

Judged win rate of the party's team (J, %), seats averaged; the opponents are always five DraftRec strangers:

| party size | aware | blind | real picks | generic | aware, CONS | blind, CONS |
|---|---|---|---|---|---|---|
| 1 | 52.2 | 48.7 | 50.4 | 49.4 | 51.7 | 51.9 |
| 2 | 55.3 | 47.8 | 50.8 | 50.4 | 52.8 | 53.2 |
| 3 | 57.7 | 47.5 | 52.0 | 50.8 | 54.1 | 55.1 |
| 4 | 61.0 | 47.2 | 53.3 | 49.7 | 55.9 | 57.0 |
| 5 | 64.1 | 46.8 | 53.9 | 49.6 | 57.3 | 59.5 |

## Where the party should pick

Solo user, J gain of the aware drafter over a typical pick at that turn (both seats pooled, then each seat), with diagnostics from the same drafts:

| solo user's turn (team's picks) | draft step, first-pick team / second | J gain, both seats | first-pick team | second-pick team | CONS gain | his best hero still free at his turn (generic) | his pick's skill, aware / generic (pp) |
|---|---|---|---|---|---|---|---|
| 1st (bans for the team) | 4 / 5 | +3.7 (2.5, 4.8) | +4.3 (2.7, 5.8) | +3.1 (1.6, 4.7) | +1.5 (0.9, 2.1) | 91% | +1.1 / −0.6 |
| 2nd | 7 / 6 | +1.8 (1.0, 2.6) | +2.0 (0.8, 3.1) | +1.7 (0.5, 2.8) | +1.5 (1.0, 2.0) | 88% | +1.1 / −1.0 |
| 3rd | 8 / 11 | +2.6 (1.9, 3.3) | +2.9 (1.9, 3.9) | +2.2 (1.3, 3.2) | +1.3 (0.9, 1.7) | 86% | +1.1 / −1.0 |
| 4th | 13 / 12 | +3.1 (2.5, 3.7) | +3.1 (2.2, 3.9) | +3.1 (2.2, 3.9) | +1.7 (1.3, 2.0) | 77% | +1.2 / −1.5 |
| 5th | 14 / 15 | +3.1 (2.6, 3.5) | +3.0 (2.3, 3.7) | +3.1 (2.6, 3.7) | +1.5 (1.2, 1.7) | 80% | +0.8 / −1.8 |
| 1st, bans left to GD | 4 / 5 | +1.4 (0.5, 2.4) | +0.9 (−0.6, 2.4) | +1.9 (0.7, 3.3) | +0.7 (0.1, 1.2) | | |

Parties of 2 to 4 on the team's first turns (early), middle or last turns (late), J gain over typical picks, both seats:

| party | early (holds the bans) | early, bans left to GD | middle | late |
|---|---|---|---|---|
| 2 | +6.8 (5.7, 8.0) | +2.8 (1.9, 3.9) | +4.2 (3.3, 5.1) | +5.8 (5.1, 6.4) |
| 3 | +8.7 (7.5, 9.8) | +4.7 (3.6, 5.8) | +7.9 (7.0, 8.8) | +7.6 (6.8, 8.5) |
| 4 | +11.7 (10.6, 12.8) | +9.1 (8.1, 10.2) | – | +10.2 (9.3, 11.1) |

Reading:
- The first pick's value is the bans. A solo user on the first pick who does not ban gains the least of any slot (+1.4pp); with the bans it is the best slot (+3.7pp). For parties the same holds: early with the bans is best, early without them is worst.
- Among the picks themselves, later turns win: the drafter sees more of the draft (the opponents' and the strangers' picks) and fits the player's hero to it. The comfort risk of waiting is real but small (the player's best hero is gone by his turn in about a fifth of the late drafts) and the drafter compensates by picking another strong personal hero.
- Practical advice for a solo user: claim the team's first pick if your client gives that player the bans; otherwise ask for one of the last two picks.

## Definitive run: sizing and design (set up 2026-10-22)

The paper reports the party-size result and the pick-slot result, so the definitive run gets its own lobby set sized for slot-level CIs.

- **Lobby set (`P3T_LOBBYSET=party`, `p3t_party.py select`):** 1,800 real tournament-window lobbies (2026-05-23 to 06-30, never the sealed build, CAP_TS 2026-09-28), all ten players with 50+ earlier games. First every lobby whose players all have 10+ judge-window games (the tournament's rule), then lobbies with 5+ each in a fixed random order (seed 20261022). The tune lobbies are left out. On the current panel: 1,300 strict, 1,128 fill-eligible, 500 taken from the fill. The definitive store has more players with history, so the strict share should rise. The probe, player-prior and opponent-model fit lobbies are now drawn after this set and avoid it (`p3t_data.eval_lobbies`).
- **Plans:** the effectiveness plan (party sizes 1 to 5, random seats, arms aware, blind, generic, real) and the full slot grid: every set of team turns for every party size (31 sets), both team sides, each with the bans going to the team's first pick turn, plus the same set with no party bans whenever it includes the first turn (62 configurations per side pair, aware and generic).
- **Contrasts reported:** every grid cell per side and with both sides pooled; pooled over all slot sets of a party size and over all sizes; solo slot gaps (gain at slot a minus gain at slot b, paired by lobby) for every pair of slots including the no-ban first pick. All CIs are lobby-bootstrap.
- **Precision at 1,800 lobbies** (projected from the 300-lobby grid run by 1/sqrt(n); the 1,800-lobby smoke matched it: ±0.44 for the solo random-seat gain against ±1.02 at 300):

| solo slot (both sides pooled) | J gain 95% CI half-width | per side, worse side | lobbies for ±0.5 per side |
|---|---|---|---|
| 1st turn, with bans | ±0.46 | ±0.66 | 3,150 |
| 1st turn, no bans | ±0.42 | ±0.58 | 2,440 |
| 2nd | ±0.33 | ±0.48 | 1,670 |
| 3rd | ±0.29 | ±0.41 | 1,200 |
| 4th | ±0.24 | ±0.37 | 980 |
| 5th | ±0.17 | ±0.27 | 510 |

  Every solo slot meets ±0.5 with both sides pooled; the worst multi-player cell (both sides pooled) is ±0.50. Per side, the first-turn slots and the larger 3- and 4-player cells stay at ±0.6 to ±0.75. Slot gaps (paired) come to ±0.3 to ±0.6, so gaps of 1 to 2 points separate and gaps of about 0.3 (the last three solo slots) do not.
- **Judge-free check:** pooled over the slot sets of a party size, the realized-record difference gets ±0.23pp (solo and duo), ±0.28 (trio), ±0.36 (four), ±0.71 (five), and ±0.19 over all sizes; a 0.5pp true gain is detectable for parties of one to four. Per slot it stays at ±0.35 to ±0.62 and slot gaps at about ±0.7, which cannot separate slots. The per-player record noise (SD 10 to 24pp per lobby) means ±0.5 per slot needs about 2,700 lobbies and ±0.5 on the worst slot gap about 4,300 (both sides pooled), against 2,428 lobbies in the tournament window that meet the 5-game rule; and even ±0.5 would not resolve slot gaps in a record effect that is itself about 0.5pp. The paper can state the judge-free result by party size, not by slot.
- **Runtime (measured on the 5090):** the full grid takes 837 s on 300 lobbies; at 1,800 lobbies one configuration takes 4.2 to 5.8 times its 300-lobby time. Grid plus effect at 1,800 lobbies: about 1.5 h of search, plus about 10 min of scoring, per pass on the 5090 alone; about 1 h sharded across the 5090 and 3080 (`run_gpu_pair`). Two passes (new model headline, old full_ms model sensitivity): about 3.4 h on the 5090 alone or about 2 h sharded. Peak RSS 2.2 GB per process. Per-side ±0.5 on every solo slot (3,150 lobbies) would take about 2.8 h per pass on the 5090 alone and needs the 3-game fill rule.
- **Pipeline:** `remote/p3t_pipeline.sh` selects the party set right after the tune set and before the probe lobbies, builds the party DraftRec history with the opponent-model histories, and runs the effect and grid plans sharded over both GPU hosts, then scores on the 5090. It runs in both passes (record, then replay with the new model's answers). The kernel determinism and parity gates, including the party gate, run on both hosts before any stage.

## Limits

- One prototype pass on current data; the definitive rerun after Oct 29 uses the new player-state model and the definitive panel.
- The ban rule is an assumption; the "bans left to GD" rows bound its effect.
- Strangers and opponents are a model (DraftRec sampling), not real players; they do not react to the party, and there is no communication between party and strangers (in-game requests, swaps).
- The judge's attenuation applies here as in the tournament; the judge-free check is underpowered at 300 lobbies.
- Compute: 108 party drafts of 300 lobbies in 11 minutes on the 5090 (10 GPU-minutes of search), plus a few CPU-minutes of scoring.

## Files

`training/personalization/p3t_party.py` (drafts, scoring), `p3t_party_report.py` (capture ratios, position diagnostics), results `results/p3t/party.json`, `party_report.json`. Definitive: `P3T_LOBBYSET=party` with `select`, `run --plan effect`, `run --plan grid` (results `party_party.json`, `party_party_report.json`), wired into `remote/p3t_pipeline.sh`. Prototype rerun: `python personalization/p3t_party.py run --plan effect`, `run --plan position`, `score`, then `python personalization/p3t_party_report.py` (GPU host, run_rcf_full env).
