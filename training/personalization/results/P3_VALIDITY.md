# P3 validity: new accounts, smurfs, premade parties (2026-09-30)

Script: `p3_val_validity.py`, run on the lag-1 count contract through `p3_fix_counts.run_fixed` (output `fix/p3_val_validity.json/.txt`; the prevalence denominator is from `p3_fix_arms.py misc`). Snapshot Storm League games from 2024-04 to 2026-05-22. Residual r = y − WP_drift, and "after the skill model" means e = r − (experience offset + posterior mean), with the phase-1 kernel online through the previous day.

Revised 2026-10-01 after the consolidated audit (experience counts from earlier days only, offset table refit on them); changes are logged in `P3_AUDIT_FIXES.md`.

## A. New accounts and smurfs

**New account:** first seen on or after 2024-07-01 with median hero level ≤ 5 over its first 10 games. There are 51,432 such accounts, and 13,049 reach 20 games.

**Residual by the new account's game index** (pp, 95% CI). Established players (300+ games, not new) sit at −0.1.

| games | 1-5 | 6-10 | 11-20 | 21-50 | 51-100 | 101-300 | 301+ |
|---|---|---|---|---|---|---|---|
| mean residual | +3.5 | +5.9 | +5.5 | +4.2 | +2.5 | +1.3 | +0.9 |

Fresh accounts win clearly more than the draft-aware WP predicts, peaking around games 6 to 20. This is the opposite sign of the "first games in the window" effect in phase 1 (−3pp). That effect came from returning veterans on old accounts; genuinely new accounts over-perform, which is the signature of alternate accounts and smurfs.

**Smurf-like flag:** first-20-game z = Σr / √Σ wp(1 − wp) > 2.33 (1% one-sided if there were no skill signal).

| | value |
|---|---|
| eligible new accounts (20+ games) | 13,049 |
| flagged | 1,378 (10.6%; 130 expected by chance) |
| flagged accounts' first-20 win rate | 83% (residual +32.8pp) |
| flagged accounts in games 21 to 100 | +9.3pp (8.9, 9.7): they stay strong, so this is not only selection noise |
| games containing a flagged account in its first 20 games | 24,211 (2.24% of the 1,082,432 represented games) |

**Bias on other players' residuals.** In those games, each opponent's residual averages −32.0pp (teammates +31.6pp). For comparison, opponents of other new accounts in their first 20 games average −3.7pp. The size is mechanical, because residuals are team outcomes: a smurf's team wins 83% against a ~50% expectation, and all five opponents absorb it. These games are 2.24% of represented games. A player in one of them is an opponent in 5 of the 9 other seats and a teammate in 4, so the average shift is about 0.0224 × (5/9 × −32 + 4/9 × +32) ≈ −0.08pp. The problem is noise more than bias: about one ±32pp shock every 45 games, and a real bias only for players who meet smurfs mostly on one side.

**Excluding those games from the skill history** (retrospective: the flag uses the account's first 20 games, which are not known when the earlier of those games are scored; the causal version is P3_EXTENSIONS section 4B, which changes the gain by −0.00001):

- **Estimates barely change.** Correlation with the full-history estimates on V2 slots is 0.979, the mean absolute change is 0.17pp, and 6.6% of slots move by more than 0.5pp.
- **Held-out prediction gets worse.** The game-level gain drops from +0.0142 (full history) to +0.0130 (smurf games removed; paired −0.0013, CI −0.0014 to −0.0011). Removing them also removes the smurfs' own strong early games, and the smurfs keep winning afterwards.
- **Verdict.** Smurfs are real (about 10% of new accounts) and inflate their own early residuals massively. For everyone else the bias is small, and excluding the games costs more than it gains. A better fix is to model the new-account effect. The experience offset already does this partly, since its "0 to 10 games seen" cells absorb some of it. Adding a new-account prior mean (+4 to +6pp for low-hero-level accounts in their first 50 games) is the natural next step, and it needs no exclusion.

## B. Premade parties

The party id is 0 for solo players (59.6% of slots). Party sizes by slot share: duo 19.3%, trio 9.9%, 4-stack 5.7%, 5-stack 5.6%.

| party size | raw residual, 2024-04+ | after skill model, 2024-04+ | after skill model, post-cutoff |
|---|---|---|---|
| solo | −0.5 | −0.6 (−0.6, −0.5) | −0.7 |
| 2 | −0.1 | −0.3 (−0.4, −0.2) | −0.2 |
| 3 | +1.2 | +0.9 (0.8, 1.1) | +1.0 (0.6, 1.4) |
| 4 | +2.0 | +1.7 (1.4, 1.9) | +1.6 (1.0, 2.2) |
| 5 | +1.5 | +1.1 (0.9, 1.4) | 0.0 (−0.7, 0.7) |

Values are per-slot residuals in pp with game-clustered 95% CIs.

- Premade groups of three or four beat the sum of their members' skill by about 1 to 1.7pp per slot. Duos do not.
- Five-stacks are +1.1pp over the full window and zero in the recent test window.
- Solo players sit slightly below expectation.

**Game level** (on top of WP + skill model, fit V1, test V2):

| added | extra log-loss gain (95% CI) | coefficient |
|---|---|---|
| number of partied players (team difference) | +0.00018 (0.00003, 0.00032) | +0.021 per player |
| largest party size | +0.00007 (−0.00007, 0.00020) | |
| all party terms | +0.00025 (0.00005, 0.00042) | |

Party effects are real but small next to individual skill (+0.0142). They are worth one term in the combiner. They also mean a party-heavy player's individual skill estimate is slightly inflated by his parties; the effect is under 1pp.
