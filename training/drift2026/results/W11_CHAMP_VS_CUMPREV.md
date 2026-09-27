# W11 — direct champion (decayed90k100) vs cumprev agent head-to-head

15x15 seed pairings, 10 drafts/cell staggered, 4,500 drafts. Motivated by
W10 exposing a seed-count artifact in W8b's comparison (champion@15seeds
22.7% degen vs cumprev@5seeds 14.5%; cumprev@15seeds is 22.0%).

Crossed-RE inference (champion side positive):
- future-meta hero-WR delta: +0.14 ± 0.10 (z = 1.4)
- degen delta: +4.1pp ± 6.0 (z = 0.7)
- consensus: 0.511 ± 0.009 (z vs .5 = 1.3)
- synergy delta: -0.08 (iid ±0.03; inside seed noise)

VERDICT: statistically tied at the policy level. The W8b "best VF != best
agent" twist dissolves; the recommendation simplifies to "use decayed
aggregates" (best prediction, agent quality indistinguishable). The clean
residual finding: maintenance-variant choice moves policy quality ~0 while
maintenance-vs-none moves it +0.7-0.8pp — flavor is a prediction-level
decision, refresh-at-all is the policy-level one.
