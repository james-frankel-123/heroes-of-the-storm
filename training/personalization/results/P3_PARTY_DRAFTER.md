# P3: the party drafter (partial control)

**Draft rules corrected 2026-10-23 (Max).** In ranked drafts the team's highest-ranked player bans, whatever the pick order, and on each team pick turn any teammate may lock (whoever clicks first takes the turn). The prototype below assumed the player on the team's first pick turn bans and treated pick slots as assigned. Its numbers stay here as the record of that setup; they are not the paper's numbers. The definitive run uses the corrected rules ("Definitive run" below).

## Definitive results (run def1027, finished 2026-10-28)

The corrected rules apply throughout: the top-ranked player bans; any teammate may lock; strangers lock in a random order. The run uses the definitive headline lobby set (`results/p3t_def1027_headline/party_party.json`, `party_party_report.json`); old-model pass in brackets.
- **Lobbies:** 1,175, all of them under the 10-game rule. Old tune, old tournament and new headline lobbies are excluded by replay ID. The set is below the 1,500 floor and ran as decided.
- **Precision:** 95% lobby-bootstrap CIs, both team sides pooled unless stated.
- **Units:** J, J_cal and CONS in pp of the party team's win probability. Record = judge-free realized record per party-team player, pp.

**Gain over generic picks by party size** (random members and lock turns; the party holds the bans when a member is the top-ranked player):

| party | J | J_cal | CONS | record | capture of the full team's J gain | vs blind: J | vs real picks: J |
|---|---|---|---|---|---|---|---|
| 1 | +2.5 (2.1, 2.8) [2.4] | +2.7 | +1.6 | +0.1 (−0.5, 0.8) | 20% (17, 23) [19%] | +2.8 | +2.2 |
| 2 | +4.8 (4.3, 5.2) [4.9] | +5.3 | +3.3 | +0.9 (0.1, 1.7) | 38% (34, 42) | +5.5 | +4.4 |
| 3 | +8.0 (7.5, 8.5) [8.1] | +8.8 | +4.9 | +0.6 (−0.3, 1.4) | 64% (60, 69) | +8.4 | +6.8 |
| 4 | +10.3 (9.8, 10.8) [10.5] | +11.3 | +6.3 | −0.1 (−1.1, 0.8) | 83% (78, 88) | +11.2 | +8.3 |
| 5 | +12.5 (11.9, 13.0) [13.2] | +13.7 | +7.6 | +1.2 (0.3, 2.1) | 100% | +13.8 | +9.8 |

- **Capture.** A solo user gets a fifth of what a full team gets, a duo 38%, a trio 64%, a four-stack 83%. The capture CIs resample whole lobbies, numerator and denominator together.
- **Sides.** First- and second-pick teams agree within noise at every size (solo 2.6 vs 2.4; five 12.5 vs 12.4).
- **Prototype comparison** (old rules, 300 lobbies): solo +2.8, five +14.5, solo capture 19%. The definitive gains are similar, a little smaller at five.

**Who holds the bans in real lobbies** (Heroes Profile party ids, all 60,500 tournament-window lobbies):

| party size | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| holds the bans | 19.0% (362,397 solo players) | 43.4% | 66.1% | 84.7% | 100% |

- The 1,175 study lobbies give 17.7%, 44.9%, 74.1%, 92.5% and 100%.
- Weighted by the real rates, the solo gain is +2.45 (2.09, 2.83).
- Split by holding: a solo user who holds the bans gains +3.6 (2.5, 4.8); one who does not, +2.2 (1.8, 2.5).
- For a duo: +6.0 holding, +3.9 not.

**Solo user, fixed lock timing** (the grid, J, both sides; "bans" = the user is the top-ranked player):

| lock at the team's | with the bans | without the bans | weighted by the real rate |
|---|---|---|---|
| 1st turn | +2.8 (2.3, 3.3) | +1.3 (0.9, 1.8) | +1.6 (1.3, 2.0) |
| 2nd | +3.4 (2.9, 3.9) | +1.7 (1.3, 2.1) | +2.0 (1.7, 2.4) |
| 3rd | +4.4 (3.9, 4.9) | +2.1 (1.8, 2.4) | +2.5 (2.2, 2.8) |
| 4th | +4.5 (4.0, 5.0) | +2.4 (2.1, 2.6) | +2.8 (2.5, 3.0) |
| 5th (last) | +4.8 (4.3, 5.2) | +2.8 (2.6, 3.0) | +3.1 (3.0, 3.3) |

- **Later is better at every step.** Weighted gaps:
  - first minus last −1.5 (−1.9, −1.1);
  - 2nd minus last −1.1;
  - 3rd minus last −0.6 (−0.9, −0.3);
  - 4th minus last −0.4 (−0.6, −0.1).
- **The bans add 1.5 to 2.3pp at every turn.**
- **Old model:** the same pattern. Weighted first minus last −1.6; the last turn +3.3 weighted.
- **Parties** (weighted, mean over all turn sets): +2.4, +5.0, +7.8, +10.4, +12.2 for sizes 1 to 5.

**Lock-or-wait vs fixed timing** (solo, weighted by the real holding rate, J):

| lock-or-wait minus | J | CONS | record |
|---|---|---|---|
| always first | +1.4 (1.0, 1.7) | +0.3 | +0.3 (−0.3, 0.9) |
| always last | **−0.6 (−1.0, −0.3)** | −0.1 | +0.3 (−0.3, 0.9) |
| a fixed turn at random | +0.2 (−0.1, 0.4) | 0.0 | +0.3 |
| generic picks at a random turn | +2.6 (2.3, 2.9) | +1.8 | +0.3 |

- **The reading is confirmed: always-last beats the adaptive rule by 0.6pp** under J (old model 0.5, CI 0.2 to 0.9). The advice for a solo user is to lock last.
- **The adaptive rule locks too early.** On the first-pick team it locks at the first turn 29 to 37% of the time.
- **Why it locks early:** it compares the drafter's own value estimates, which undervalue waiting relative to the judge.
- **CONS and the judge-free record do not separate the timing rules.**

**Achieved precision at 1,175 lobbies** (95% CI half-widths, J):
- **Solo turns, weighted, both sides pooled:** ±0.37, ±0.33, ±0.27, ±0.24, ±0.18 (turns 1 to 5). With the bans ±0.45 to ±0.52; without ±0.43 to ±0.20.
- **Per side:** worst ±0.76 (with the bans) and ±0.61 (without).
- **Slot gaps (paired):** ±0.2 to ±0.4. The plan expected gaps of about 0.3 among the last turns to be inseparable. Here they separate: 4th minus last is −0.36 (−0.62, −0.10). The paired design (common random numbers across turns) makes the gaps tighter than the projection.
- **Multi-player cells, both sides pooled:** worst ±0.55.

**The judge-free record for parties is mostly null.**
- **Solo:** +0.1 (−0.5, 0.8) against generic picks. It is null at every lock turn, with and without the bans.
- **Larger parties:**
  - Some cells are positive: duo +0.9 (0.1, 1.7); five +1.2 (0.3, 2.1); grid means over turn sets +0.6 to +1.6 for sizes 2 to 5.
  - Others are null: trio and four-stack in the random-seat plan.
- **Against the player-blind drafter** the record is clearly positive for parties of 2 to 5 (+2.3 to +4.9). That reflects the blind drafter putting players on heroes they are bad at, not the aware drafter's edge over typical picks.
- **The judged gains are not confirmed player by player at these sample sizes.**

**MMR key check (2026-10-28).** The ban-holder ranks used the shared MMR stamp key, which collided for games of one player 2^20 replay ids apart (fixed behind `P3_MMR_KEY_FIX=1`, now the pipeline default).
- **Effect on ranks:** with the fix, 302 of 1.76M rank cells change. The ban holder changes on one side of 1 of the 1,175 party lobbies, and in none of the headline or tune lobbies.
- **Effect on rates:** the real ban-holder rates are unchanged to four decimals.
- **Effect on results:** a CPU rescore on the main box reproduces `party_party.json` exactly. Leaving the affected lobby out moves every judged point estimate by at most 0.02pp and every record point estimate by at most 0.10pp. No redrafting.

**Flags.**
- The first-turn, no-ban solo cell differs between sides: +0.9 (0.3, 1.5) for the first-pick team, +1.7 (1.2, 2.3) for the second. All other turns agree within ±0.7. Pooling as decided is fine, but this is the one cell where the sides differ by more than the noise.
- The ban-holder model's holder term still adds nothing measurable. Who holds the bans is not identifiable from our data (methods note); the rule rests on Max's account of the client.

## Prototype record (2026-10-22, old draft rules; superseded)

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

- **Lobby set (`P3T_LOBBYSET=party`, `p3t_party.py select`; coordinator 2026-10-25):** every real tournament-window lobby (2026-05-23 to 06-30, never the sealed build, CAP_TS 2026-09-28) whose ten players all have 50+ earlier games and 10+ judge-window games each, except the old tune and old tournament lobbies (frozen by replay ID in `p3t_excluded_replays.json`) and the new headline 300. The new tune set is the old tune set; the headline is a fresh 300 that excludes both frozen sets. On dry1022 that leaves 1,141 lobbies (below the 1,500 floor: run anyway, no loosening). At 1,141 lobbies the solo-turn CIs (both sides pooled) scale from the 1,800-lobby table by 1.26: about ±0.58 (1st turn), ±0.41, ±0.36, ±0.30, ±0.21 (5th); the judge-free record by party size about ±0.29 (solo). The achieved CIs are reported from the run. The probe, player-prior and opponent-model fit lobbies avoid this set (`p3t_data.eval_lobbies`).
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
