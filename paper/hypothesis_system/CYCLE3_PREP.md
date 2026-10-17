# Cycle 3 preparation: warm and cold refits, replacement cards, sessions w3-K-*

**Wind-down (2026-10-17): cycle 3 was not run.** Max decided there is no standalone paper 3, and anything useful folds into paper 2.
The w3-K-b, w3-K-blinded and w3-K-shuffled sessions were prepared and prefit but never launched. No broker or proposer was started, and no card was submitted.
What did get done: warm and cold refit code, replacement cards, their tests, and cold refits of the three cycle-2 finals (official numbers in `CYCLE2_RESULTS.md` section 8).
Nothing is running on this box, the 3090 or the 5090.
The launch commands in section 5 are kept as a record only. Do not run them without a new decision.

Date: 2026-10-17. Code: commit d145e9a. Spec: `FRAMING.md`. Previous: `CYCLE2_PREP.md`, `CYCLE2_RESULTS.md`.

## 1. Incumbent refits (Max, 2026-10-15)

| Mode | When | Start | Stopping rule |
|---|---|---|---|
| legacy | sessions without a `fit_policy` (waves 0 to 2) | zero | 5 EB rounds x 400 iterations; registry keys unchanged |
| cold | a session's start incumbent; every end-of-cycle refit (`session cold-refit <sid>`) | zero | converged, no practical cap (200,000 iterations a round, 200 EB rounds) |
| warm | refits after an accept during a cycle | the newest registered fit sharing the most terms (converged fits first), new terms at their Q2 offset-fit values, else 0 | converged, or 6,000 iterations |

- **Convergence.** Two levels. Inner: every coefficient's projected gradient is at most 0.001 of its diagonal posterior SD. Outer: no EB precision moves more than 0.01 in log10, and blocks shrunk to nothing (precision at least 1e6) count as settled. The fitter uses L-BFGS-B on diagonally preconditioned coefficients, which leaves the optimum unchanged. Q2 term fits are converged too in fit-policy sessions.
- **Metadata.** Every registry entry and INC.json records `fit_mode`, `fit_status` (converged, inner and EB status, iterations, EB rounds, final criterion) and the warm sources. The harness ledger `fit` record carries the same fields.
- **Scheduling.** Cold and warm refits take the 3090 refit token from commit 276c9bf. At most one refit holds a slot, so one slot always serves tests, and superseded refits are still skipped.
- **Measured.** See `CYCLE2_RESULTS.md` section 8. A cold fit takes 3,400 to 3,700 iterations and about 2 hours. A warm refit is about 20% faster and reaches the same optimum (log loss within 0.00001).

## 2. Replacement cards

- **Card format.** A predictive card with `replaces: <card id of a term in the incumbent>`.
- **Test.** One Q2 test, charged 1.5 bits once, to the groups of both terms. The new term is fit on the incumbent's logit minus the old term. On dev, the incumbent as it is is compared with the incumbent after the swap.
- **Parameter cap.** Checked on the result: incumbent minus old term plus new term. A swap over 20,000 parameters is refused, and nothing is charged. Example: w2-K-blinded p03 for c10 gives 17,668 - 1,595 + 2,553 = 18,626, which fits.
- **On accept.** The terms are swapped, and `replaced` is recorded in the state and carried to later cycles.
- **Refusals.** Unknown ids, backbone terms, explanation and mechanism cards, and targets that left the incumbent while the test ran.

## 3. Tests

| Suite | Result |
|---|---|
| `tests.test_warm_cold` (new) | 12/12. Legacy fitter bit-identical to cycle 2. Cold and warm converge. Warm reaches the cold optimum (max 0.05 posterior SD, held-out log loss within 5e-6) in fewer iterations. Metadata is recorded. With a warm and a cold refit pending, one refit at most holds a slot and a short test runs in 0.11 s |
| `tests.test_replacement` (new) | 14/14. Synthetic planted pairs: sparse to dense table gains (z 14.4), the reverse loses (z -15.2), null loses (z -28.5). A same-spec swap is neutral when both fits converge (gain -3e-5 against +0.027 for a real swap). The gate, the cap on the result and the broker refusals are covered |
| Existing | refit scheduling 11/11, incumbent race 7/7, ledger 5/5, planted-leak fixtures 17/17, explain 15/15 |

Finding from the tests: with legacy (unconverged) fits, swapping a term for the same spec gained z 2.5 from the refit alone. Converged fits remove that bias, which is why Q2 term fits converge in fit-policy sessions.

## 4. Sessions w3-K-* (prepared, not launched)

- **Carried over** from the w2 finals: cards of both earlier cycles (ids reserved), Dpost group bits of both cycles, shuffle and blinding keys, replacements and spent games. The `fit_policy` makes the start incumbent the cold refit.
- **Prompt.** `prompts/cycle3.md`. All three arms get the same text, including: "A min-5000 ally-pair family scored strongly last cycle but could not fit beside the existing pair table; you may propose it as a replacement." This sentence carries a released result from w2-K-blinded p03, whose 1.5 bits were refunded, into the other arms.
- **Prefit fits (all converged).**

| Session | Incumbent (cold) | Explore (warm) | Confirm (warm) |
|---|---|---|---|
| w3-K-b | inc_ff820fcca10b247f | inc_bb5299b49e2af21c | inc_06c33bdde9d39f44 |
| w3-K-blinded | inc_377ca3fb446b1275 | inc_148beb29d3b22bf5 | inc_0dae6483a2c2a690 |
| w3-K-shuffled | inc_7f4b9cd10f5a23e2 | none (no tables) | none |

## 5. Launch commands (record only; not run)

```bash
cd /home/max/heroes-of-the-storm
export PYTHONPATH=. PY=/home/max/venvs/scilearn/bin/python
for s in w3-K-b w3-K-blinded w3-K-shuffled; do $PY -m training.scilearn.session broker $s; done
for s in w3-K-b w3-K-blinded w3-K-shuffled; do
  $PY -m training.scilearn.session wait-broker $s &&   # waits until the broker answers, not just for the socket file
  nohup $PY -m training.scilearn.session launch $s > /home/max/scilearn-sessions/$s/logs/launch.out 2>&1 &
done
for s in w3-K-b w3-K-blinded w3-K-shuffled; do $PY -m training.scilearn.session canaries $s; done
# end of cycle: stop, cold refit, closeout
for s in w3-K-b w3-K-blinded w3-K-shuffled; do $PY -m training.scilearn.session stop $s; $PY -m training.scilearn.session cold-refit $s; $PY -m training.scilearn.session closeout $s; done
```
