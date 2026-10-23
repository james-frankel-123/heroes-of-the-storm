# P3: the party drafter (partial control), prototype on current data (2026-10-22)

**Draft rules corrected 2026-10-23 (Max).** In ranked drafts the team's highest-ranked player bans, whatever the pick order, and on each team pick turn any teammate may lock (whoever clicks first takes the turn). The prototype below assumed the player on the team's first pick turn bans and treated pick slots as assigned. Its numbers stay here as the record of that setup; they are not the paper's numbers. The definitive run uses the corrected rules ("Definitive run" below).

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

## Definitive run: design (corrected rules, 2026-10-23) and sizing

**Rules in the simulation.**
- **Ban holder:** each team's top-ranked player by as-of MMR (Heroes Profile's causal rating, the latest stamp parsed before the game; `p3t_data.py ranks` writes `lobby_ranks.npz`). Causal MMR covers 98.1% of tournament-window slots; per-player league tiers are not stored (only the game's), so the other 1.9% are ranked at the lobby's average MMR. A party holds the bans iff a member is the top-ranked player.
- **Bans the party does not make** (a stranger holds them, and the opponents' bans) come from the holder's model, sampled: softmax(w0 log p_GD + a1 log r + a2 log(1 - q_holder)), with r = DraftRec's mixture for the other team's next pick (denial) and q_holder = the holder's own DraftRec pick at his team's next turn. Fit on the 3,000 opponent-model fit lobbies (`p3t_oppdr.py holderfit`, w0 fixed at the opponent model's 0.52): a1 0.36, a2 0.20. On 1,800 held-out bans the holder term adds nothing measurable (NLL 3.467 with and without it, GD alone 3.665), and a placebo holder (a random other teammate) fits as well. So the ban data neither confirm nor contradict the top-ranked rule; the rule stands on Max's account of the client. In the party's search a stranger holder's remaining ban is a chance node with this model's row; opponent bans are GD chance nodes.
- **Lock timing.** Fixed timing: the party always locks at the chosen team turns; the strangers lock in a uniformly random order on the other turns. The drafter does not know that order: every future stranger (and opponent) pick in its search is a chance node with DraftRec's mixture over the players still to pick, credited to a planning assignment drawn independently of the realized order.
- **Lock-or-wait (solo):** at each team pick turn the drafter values "lock now" (its search's value of the chosen hero) against "wait" (for each later team turn, the mean over 4 sampled futures of its search's value of the chosen hero at that turn; the best later turn counts). It locks iff now is at least as good; otherwise the next stranger in the random order locks. The rule and its settings (lookall, 4 sampled futures, chosen-hero value) are frozen as of 2026-10-23. Two simpler rules were tried first on the 300 tournament lobbies and dropped because they lock too early: a search rooted at the next stranger's chance node (kernel flag on the root step, still in `cuda_explore`, gated) and a one-step lookahead. On those lobbies the chosen rule beats always-first by +1.6pp J (0.8 to 2.4) and is level with always-last (-0.5, -1.2 to 0.2).
- **Real holding rates** (all tournament-window lobbies; party ids from Heroes Profile): a solo player holds the bans 19.0% of the time (362,397 solo players; 20% if rank were random), a duo 43.4%, a trio 66.1%, a four-stack 84.7%, a five-stack 100%. Gains are reported for holders, non-holders, and weighted by these rates.

**Plans (1,800 lobbies or every lobby meeting the strict rule).** Effect: party sizes 1 to 5, random members and random lock turns, arms aware, blind, generic, real; the party holds the bans when a member is the top-ranked player (rate m/5 by construction), and the gains are split by holding and reweighted to the real rates. Grid: every set of team turns for every party size (31), both sides, with the top-ranked player in the party ("ban") and without him ("noban"; not possible for five), aware and generic (122 configurations). Lock: solo, both sides, ban and noban, compared on common random numbers with always-first, always-last, a fixed turn at random, and generic picks at a random turn. Configurations sharing a party size, side and holding share members, stranger locking order and random draws, so slot comparisons are paired.

**Smoke on the 300 tournament lobbies (current data, corrected rules; not the paper's numbers).** Solo gain over generic picks, J, weighted by the real holding rate: +2.5pp (2.2 to 2.8) pooled over lock turns; +4.3 when holding the bans, +2.1 when not. By lock turn (weighted): +1.0 at the team's first turn, +1.9, +2.6, +3.3, +3.6 at the last; later is better (first minus last -2.6, -3.4 to -1.7). Party sizes (weighted, all slot sets): +2.5 (solo), +5.0, +8.1, +10.6, +14.9 (five). The judge-free record stays within noise (pooled over everything +0.4, -0.0 to 0.9).

**Sizing.** The precision table from the prototype's variance still applies to the fixed-timing cells (both sides pooled, 1,800 lobbies):

| solo turn (both sides pooled) | J gain 95% CI half-width | per side, worse side | lobbies for ±0.5 per side |
|---|---|---|---|
| 1st | ±0.46 | ±0.66 | 3,150 |
| 2nd | ±0.33 | ±0.48 | 1,670 |
| 3rd | ±0.29 | ±0.41 | 1,200 |
| 4th | ±0.24 | ±0.37 | 980 |
| 5th | ±0.17 | ±0.27 | 510 |

Decision (coordinator, 2026-10-22): the paper pools both team sides and keeps the strict 10-game rule; per-side numbers are reported with their wider CIs; the roughly 0.3-point gaps among the last turns are reported as not separable. The judge-free record can be stated by party size (±0.2 to 0.4pp for parties of one to four at 1,800 lobbies), not by turn.

- **Lobby set (`P3T_LOBBYSET=party`, `p3t_party.py select`):** up to 1,800 real tournament-window lobbies (2026-05-23 to 06-30, never the sealed build, CAP_TS 2026-09-28) whose ten players all have 50+ earlier games and 10+ judge-window games (the tournament's rule), in a fixed random order (seed 20261022); every one if fewer qualify; below 1,500 the select step flags the set (`select.json` `below_floor`, a warning in the job log). The tune lobbies and the 300 tournament lobbies (the lock rule was chosen on them) are left out, so the lock-rule comparison is scored on lobbies it was not tuned on; if that leaves fewer than 1,500 on the definitive build, the coordinator is told and the tournament lobbies stay out. On the current panel 1,000 qualify after the exclusion (1,300 before). The probe, player-prior and opponent-model fit lobbies are drawn after this set and avoid it (`p3t_data.eval_lobbies`).
- **Runtime (measured on the 5090):** on 300 lobbies effect 415 s, grid 1,246 s, lock 131 s (30 min). On the 1,300-lobby party set the same configurations take 2.3 to 2.9 times as long (1.2 for the effect plan). Projected for 1,800 lobbies: about 1.7 to 1.8 h of search per pass on the 5090 alone, about 1.2 h split across the 5090 and the 3080 tournament lane (cores 16 to 19, at most 20 GB), plus about 10 min of scoring on the 3080; both passes (new model headline, old full_ms sensitivity) about 2.5 h. With 1,000 lobbies (the current panel after the exclusion), under 1 h per pass split. Peak RSS 1.8 GB per drafting process, 2.4 GB scoring.
- **Gates:** the explore kernel's determinism gate now also covers the chance root (reruns identical, visit shares sum to 1, root value in (0, 1), differs from the decision root), with every earlier gate unchanged (`p3t_determinism.py`, DETERMINISM OK on the 5090 after the rebuild). Unit tests of the holder and timing layout: `tests/test_p3t_party_layout.py` (P1 to P5).
- **Pipeline:** `p3t_data.py ranks` in the data stage; `p3t_oppdr.py holderfit` after the opponent fit (5090), its fits pushed to the 3080 lane; the party set selected before the probe lobbies; effect, grid and lock plans sharded over both GPUs; scoring and the report on the CPU host (it holds the judge). Both passes run it (record, then replay with the new model's answers).

## Limits

- The prototype's numbers (first-turn ban rule, assigned slots) are superseded by the definitive run.
- Methods note: who holds the bans is not identifiable from our data. The holder's own pick preferences do not improve held-out ban prediction, and a random teammate as placebo holder fits as well; the highest-MMR rule rests on Max's account of the client.
- Strangers and opponents are a model (DraftRec sampling), not real players; they do not react to the party, and there is no communication between party and strangers (in-game requests, swaps, who clicks first by agreement).
- The lock-or-wait rule is the drafter's own value comparison; on the smoke it matches always-last rather than beating it.
- The judge's attenuation applies here as in the tournament.

## Files

`training/personalization/p3t_party.py` (drafts, scoring), `p3t_party_report.py` (capture ratios, position diagnostics), results `results/p3t/party.json`, `party_report.json`. Definitive: `P3T_LOBBYSET=party` with `select`, `run --plan effect`, `run --plan grid`, `run --plan lock` (results `party_party.json`, `party_party_report.json`), wired into `remote/p3t_pipeline.sh`; inputs `p3t_data.py ranks` (lobby_ranks.npz) and `p3t_oppdr.py holderfit` (opp_holder.json). Prototype rerun: `python personalization/p3t_party.py run --plan effect`, `run --plan position`, `score`, then `python personalization/p3t_party_report.py` (GPU host, run_rcf_full env).
