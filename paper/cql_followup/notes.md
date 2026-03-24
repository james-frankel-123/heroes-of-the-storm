# Working Notes: Algorithmic vs Feature-Level Pessimism for OOD Value Correction

## Key Question

Can algorithmic pessimism (CQL) replace domain-specific feature engineering for
avoiding degenerate compositions in offline sequential decision-making?

## Preliminary Results

### CQL with naive multi-hot features (no compositional features)

| Alpha | Healer% | Degen% | Front% | Ranged% |
|-------|---------|--------|--------|---------|
| 0.1   | 93.5    | 13.0   | 97.0   | 98.0    |
| 0.5   | 96.0    | 10.5   | 98.5   | 100.0   |
| 1.0   | 93.5    | 9.0    | 98.5   | 99.5    |
| 2.0   | 92.0    | 12.5   | 97.5   | 98.0    |
| 5.0   | 92.0    | 13.0   | 97.5   | 97.5    |

### BEAR-style: Q(s,a) + beta * log pi_beta(a|s)

| Beta  | Healer% | Degen% | Front% | Ranged% |
|-------|---------|--------|--------|---------|
| GD only | 100.0 | 0.0  | 100.0  | 100.0   |
| 0.0 (Q only) | 2.5 | 98.0 | 100.0 | 100.0 |
| 0.1   | 98.5    | 2.0    | 99.5   | 100.0   |
| 0.5   | 99.0    | 2.5    | 99.0   | 100.0   |
| 1.0   | 99.0    | 1.5    | 99.5   | 100.0   |
| 2.0   | 97.0    | 3.0    | 100.0  | 100.0   |
| 5.0   | 99.0    | 1.5    | 99.5   | 100.0   |

### Reference baselines (from CoG 2026 aux paper)

| Method           | Features  | Healer% | Degen% |
|------------------|-----------|---------|--------|
| Naive WP greedy  | multi-hot | 49.0    | 75.8   |
| Enriched WP      | enriched  | 66.5    | 57.7   |
| Enriched+synth   | enriched  | 94.5    | 26.5   |

## Critical Observation

The Generic Draft model (behavioral cloning of human drafts) produces 100%
healer rate and 0% degenerate compositions on its own. This means:

1. **CQL's composition quality comes from mimicking the behavior policy, not
   from learned value estimation.** The CQL penalty pushes the Q-function
   toward the data distribution, which is human drafts, which always include
   healers. CQL is 93.5% healer vs GD's 100% -- it's *worse* than pure
   behavioral cloning.

2. **BEAR confirms this.** Any nonzero beta (behavior policy weight) immediately
   achieves 97-99% healer. The Q-network alone (beta=0) produces 2.5% healer,
   98% degenerate -- it learns nothing about composition.

3. **The Q-network with MC returns and no pessimism is vacuous.** Test loss
   stuck at 0.6979 (random prediction). With binary outcomes and 55-58% WR,
   there's not enough signal per-action to learn meaningful Q-values from
   offline data.

## Open Question: Is CQL Context-Aware?

CQL might still be valuable if it makes *better* picks than GD even while
maintaining composition safety. Specifically:

- Does CQL adapt to the opponent's picks (counter-picking)?
- Does CQL exploit synergies within the team?
- Does CQL adapt to map-specific needs?
- Or is it just an expensive GD with slightly worse composition?

The diagnostic showed CQL agrees with GD 75% on early picks and 50% on late
picks. The 50% divergence on late picks could be noise or genuine value-based
selection. Need richer metrics to distinguish.

## Metrics Needed

1. **Counter-pick responsiveness**: avg pairwise counter delta vs opponent
2. **Synergy exploitation**: avg within-team synergy delta
3. **Draft diversity**: distinct heroes, entropy, top-10 concentration
4. **GD similarity**: agreement rate with behavioral cloning at each step
5. **Cross-model WP evaluation**: score drafts with naive/enriched/augmented WP
6. **Map adaptation**: do picks change based on map?

## Hypotheses

### H1: CQL is just expensive GD
- High GD similarity (70%+), near-zero counter deltas, low diversity
- Paper story: "CQL avoids degenerate compositions by refusing to deviate from
  human behavior, not by understanding composition"

### H2: CQL is genuinely value-aware
- Lower GD similarity, positive counter deltas, map adaptation
- Paper story: "algorithmic pessimism discovers compositional reasoning from
  identity features alone"

### H3: Approaches are complementary
- CQL better on safety, enriched better on counter/synergy
- Paper story: "CQL constrains the policy to safe compositions; feature
  engineering enables context-aware decisions within that constraint"

## Paper Structure (tentative)

1. Problem setup (brief, reference CoG aux paper)
2. Offline RL formulation for drafting
3. CQL implementation and sweep results
4. The behavior policy confound (GD baseline reveals CQL = safe behavioral cloning)
5. Rich evaluation metrics (counter, synergy, diversity, map adaptation)
6. Comparison: algorithmic vs feature-level vs data-level pessimism
7. Discussion: when is each approach appropriate?

## Implementation Status

- [x] CQL naive features sweep (5 alphas)
- [x] BEAR-style sweep (6 betas)
- [x] GD baseline (100% healer, 0% degen)
- [x] Diagnostic: Q-value analysis, GD agreement
- [ ] Rich evaluation metrics
- [ ] CQL with enriched features
- [ ] Cross-model WP evaluation
- [ ] Map adaptation analysis
- [ ] Statistical significance tests
