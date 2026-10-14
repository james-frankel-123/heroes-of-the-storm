
## Cycle 2

This is your second cycle on the same dev tranche. Read /opt/hyp/previous_cycle_summary.md
(your own summary from last cycle, if present) and the record below, then the cycle-2 sections
of the guide (explanation cards, mechanism cards, the M tranche).

**What changed.** The terms accepted last cycle are now in your incumbent. If they include
lookup tables (hero x map, hero x tier, ally pairs, hero-vs-hero counters), those tables are
the accurate layer of the model. A table spans every kit-level term of the same shape, so a
new kit term that predicts what a table already predicts will gain nothing on a Q2 test. That
was last cycle's main lesson, and it is why theories now earn their place differently.

**Your job this cycle is to explain, not only to predict.**

1. **Explanation cards** (`type: explain`). Take one of your tables and propose a small kit-
   or role-based function that reproduces its pattern: which kit properties make a hero good
   on some maps and bad on others, which properties make one hero counter another, which
   make two allies work together. The harness scores how much of the table's held-out signal
   the theory explains, whether it transfers to 20 heroes it never saw, whether it helps
   early-patch games, and whether the kit axes you named beat within-role permutations of
   themselves. Look at your tables with `hyp table <card_id>` first (an S-only fit, one
   sandbox number per cell), form a mechanical theory, then test it. A good theory explains
   a modest share with few parameters and transfers. A theory that fits one table well and
   fails transfer is a description, not an explanation.
2. **Mechanism cards.** Parsed 2.57 replays now exist (M). Use `hyp inspect` to read games,
   `hyp query` with class Q3 to look at mediator means on M-test, and mechanism cards
   (`type: mechanism`, or a mechanism block on any card plus `hyp mtest`) to test whether a
   term in your model works through the route its theory claims. Check `hyp meta` for how
   many M-test games exist: mechanism tests open at 2,000.
3. **Predictive cards** (the old Q2 kind) are still welcome where you have a form your
   incumbent cannot express. Check first that no accepted table already spans it.

**Use your budget.** Last cycle every session stopped at about a tenth of its budget. This
cycle, plan for at least 25 cards (explanation, mechanism and predictive together). A rejected
card costs 1.5 bits and is still a result: a clean null on a well-posed theory is worth
recording. Do not retry the same theory with small variations to pass a gate; each test on a
group raises that group's bar. Spread your theories across tables and families. Stop before
25 cards only if your budget runs out.

Finish with /work/summary.md as before, and add: each explanation tested, with the share
released, the transfer result and the onset result; each mechanism tested and its result.
