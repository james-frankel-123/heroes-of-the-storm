
## Cycle 3

This is your third cycle on the same dev tranche. Read /opt/hyp/previous_cycle_summary.md
(your own summary from last cycle, if present) and the record below, then the cycle-2 and
cycle-3 sections of the guide (explanation cards, mechanism cards, the M tranche, replacement
cards).

**What changed.** The terms accepted in earlier cycles are in your incumbent, refit to
convergence. If they include lookup tables (hero x map, hero x tier, ally pairs, hero-vs-hero
counters), those tables are the accurate layer of the model. A table spans every kit-level
term of the same shape, so a new kit term that predicts what a table already predicts will
gain nothing on a Q2 test.

**New: replacement cards.** A card may now replace a term in your incumbent with a new term,
in one test charged as one card (`replaces: <card id>`; see the guide). The test compares your
incumbent as it is against your incumbent with the swap, and the 20,000-parameter cap is
checked on the result. A min-5000 ally-pair family scored strongly last cycle but could not
fit beside the existing pair table; you may propose it as a replacement.

**Your work this cycle** is the same three kinds of card as last cycle.

1. **Explanation cards** (`type: explain`). Take one of your tables and propose a small kit-
   or role-based function that reproduces its pattern. The harness scores how much of the
   table's held-out signal the theory explains, whether it transfers to 20 heroes it never
   saw, whether it helps early-patch games, and whether the kit axes you named beat
   within-role permutations of themselves. Look at your tables with `hyp table <card_id>`
   first, form a mechanical theory, then test it.
2. **Mechanism cards.** Use `hyp inspect` to read games, `hyp query` with class Q3 to look at
   mediator means on M-test, and mechanism cards (`type: mechanism`, or a mechanism block on
   any card plus `hyp mtest`) to test whether a term in your model works through the route
   its theory claims.
3. **Predictive cards**, including replacements, where you have a form your incumbent cannot
   express or a better form of a term it already holds. Check first that no accepted table
   already spans it.

**Use your budget.** Plan for at least 25 cards (explanation, mechanism and predictive
together). A rejected card costs 1.5 bits and is still a result: a clean null on a
well-posed theory is worth recording. Do not retry the same theory with small variations to
pass a gate; each test on a group raises that group's bar. Spread your theories across tables
and families. Stop before 25 cards only if your budget runs out.

Finish with /work/summary.md as before: each card with its released result; each explanation
with the share released, the transfer result and the onset result; each mechanism and its
result.
