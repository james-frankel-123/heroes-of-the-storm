# Expert study v6.2: confirmatory analysis (prereg v3, osf.io/fspuc)

Partials rule: lenient. Included raters: rater02, rater03, rater04, rater07, rater09, rater10, rater11, rater12, rater14, rater15.

## Gates (§3)

| rater | group | slot | rated | screener | catch | median ms | calibration acc | flag ≤50% | included (strict / lenient) |
|---|---|---|---|---|---|---|---|---|---|
| rater01 | A | 0 | 24 | 3/3 | 0/0 | 43495 | 0.476 | True | False / False |
| rater02 | A | 1 | 240 | 8/8 | 3/3 | 21664 | 0.575 | False | True / True |
| rater03 | A | 2 | 63 | 8/8 | 0/0 | 18770 | 0.575 | False | False / True |
| rater04 | A | 3 | 240 | 8/8 | 3/3 | 10878 | 0.550 | False | True / True |
| rater06 | B | 5 | 240 | 6/8 | 3/3 | 11978 | 0.500 | True | False / False |
| rater07 | B | 6 | 240 | 7/8 | 2/3 | 23576 | 0.550 | False | True / True |
| rater08 | B | 7 | 240 | 6/8 | 3/3 | 45356 | 0.500 | True | False / False |
| rater09 | B | 8 | 240 | 8/8 | 3/3 | 38972 | 0.450 | True | True / True |
| rater10 | B | 9 | 240 | 8/8 | 3/3 | 20428 | 0.500 | True | True / True |
| rater11 | B | 10 | 240 | 8/8 | 3/3 | 18138 | 0.525 | False | True / True |
| rater12 | B | 11 | 240 | 8/8 | 3/3 | 42729 | 0.575 | False | True / True |
| rater13 | B | 12 | 7 | 1/1 | 0/0 | 26573 | 0.500 | True | False / False |
| rater14 | B | 13 | 240 | 8/8 | 3/3 | 62736 | 0.500 | True | True / True |
| rater15 | A | 4 | 240 | 7/8 | 3/3 | 20006 | 0.700 | False | True / True |

## H1 and S5 (§4, §6): agreement with the model-preferred side

| test | items | judgments | intercept log-odds [95% Wald CI]; p; probability scale | model; success |
|---|---|---|---|---|
| H1 (confirmatory, 0.02) | 255 | 493 | 0.003 [-0.277, 0.282]; p 0.9853; prob 0.501 [0.431, 0.570] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 near-tie t001 | 272 | 529 | 0.063 [-0.181, 0.308]; p 0.6109; prob 0.516 [0.455, 0.576] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 near-tie t002 | 255 | 493 | 0.003 [-0.277, 0.282]; p 0.9853; prob 0.501 [0.431, 0.570] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 near-tie t005 | 213 | 413 | -0.053 [-0.374, 0.268]; p 0.7471; prob 0.487 [0.408, 0.567] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 evaluator naive | 255 | 493 | 0.016 [-0.269, 0.301]; p 0.9111; prob 0.504 [0.433, 0.575] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 evaluator herostrength | 253 | 492 | -0.006 [-0.270, 0.257]; p 0.9634; prob 0.498 [0.433, 0.564] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 evaluator enriched | 251 | 485 | -0.011 [-0.285, 0.262]; p 0.9345; prob 0.497 [0.429, 0.565] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 uncorrected consensus (descriptive) | 255 | 493 | 0.003 [-0.277, 0.282]; p 0.9853; prob 0.501 [0.431, 0.570] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 patch-2.57 (a) | 195 | 369 | -0.023 [-0.322, 0.277]; p 0.8820; prob 0.494 [0.420, 0.569] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| H1 patch-2.57 (b, descriptive) | 57 | 112 | 0.180 [-0.666, 1.025]; p 0.6772; prob 0.545 [0.339, 0.736] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| S5 (confirmatory) | 113 | 217 | -0.691 [-1.288, -0.094]; p 0.0232; prob 0.334 [0.216, 0.476] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| S5 patch-2.57 (a) | 97 | 184 | -0.728 [-1.359, -0.098]; p 0.0236; prob 0.326 [0.204, 0.476] | glmer y ~ 1 + (1|rater) + (1|item); success=False |
| S5 patch-2.57 (b, descriptive) | 16 | 35 | -23.937 [-47.478, -0.397]; p 0.0463; prob 0.000 [0.000, 0.402] | glmer y ~ 1 + (1|rater) + (1|item); success=False |

Pooled agreement rates (descriptive): H1 0.501 [0.457, 0.545] (247/493); S5 0.373 [0.312, 0.439] (81/217)

## H2 (§5): moderation by OOD covariate

| test | items | judgments | slope [95% Wald CI]; p | class (δ = −0.25) | class (δ = −0.20) | model |
|---|---|---|---|---|---|---|
| H2 ood_var_max (confirmatory) | 255 | 493 | 0.245 [-0.009, 0.500]; p 0.0591 | Robust (non-inferiority met) | Robust (non-inferiority met) | glmer y ~ ood + (1|rater) + (1|item) |
| H2 ood_var_mean | 255 | 493 | 0.309 [0.054, 0.565]; p 0.0178 | Robust (non-inferiority met) | Robust (non-inferiority met) | glmer y ~ ood + (1|rater) + (1|item) |
| H2 ood_var_matchup | 255 | 493 | 0.166 [-0.083, 0.414]; p 0.1910 | Robust (non-inferiority met) | Robust (non-inferiority met) | glmer y ~ ood + (1|rater) + (1|item) |
| H2 patch-2.57 (a) | 195 | 369 | 0.249 [-0.050, 0.548]; p 0.1021 | Robust (non-inferiority met) | Robust (non-inferiority met) | glmer y ~ ood + (1|rater) + (1|item) |
| H2 patch-2.57 (b, descriptive) | 57 | 112 | 0.528 [-0.358, 1.415]; p 0.2426 | Inconclusive | Inconclusive | glmer y ~ ood + (1|rater) + (1|item) |

## S1 (§6): anchor accuracy by tier, Holm family of 3

| set | tier | items | judgments | intercept [95% CI]; p | Holm p | above chance | pooled accuracy |
|---|---|---|---|---|---|---|---|
| full | low | 190 | 391 | 0.385 [0.140, 0.629]; p 0.0020 | 0.0061 | True | 0.583 [0.534, 0.631] (228/391) |
| full | mid | 190 | 391 | -0.021 [-0.264, 0.222]; p 0.8657 | 0.8718 | False | 0.496 [0.447, 0.546] (194/391) |
| full | high | 190 | 389 | 0.107 [-0.162, 0.376]; p 0.4359 | 0.8718 | False | 0.522 [0.472, 0.571] (203/389) |
| patch-2.57 (a) | low | 92 | 184 | 0.421 [0.072, 0.770]; p 0.0180 | 0.0539 | False | 0.592 [0.520, 0.661] (109/184) |
| patch-2.57 (a) | mid | 76 | 156 | -0.161 [-0.493, 0.170]; p 0.3406 | 0.6811 | False | 0.462 [0.385, 0.540] (72/156) |
| patch-2.57 (a) | high | 67 | 141 | 0.060 [-0.361, 0.482]; p 0.7797 | 0.7797 | False | 0.511 [0.429, 0.592] (72/141) |
| patch-2.57 (b) | low | 37 | 76 | 0.477 [-0.101, 1.056]; p 0.1055 | 0.3165 | False | 0.605 [0.493, 0.708] (46/76) |
| patch-2.57 (b) | mid | 36 | 70 | -0.179 [-0.676, 0.317]; p 0.4785 | 0.9570 | False | 0.457 [0.346, 0.573] (32/70) |
| patch-2.57 (b) | high | 24 | 48 | -0.083 [-0.650, 0.483]; p 0.7729 | 0.9570 | False | 0.479 [0.345, 0.617] (23/48) |

## S2 (§6): slider calibration against the consensus

lmer slider0 ~ p_cons + (1 + p_cons | rater), Satterthwaite: slope -0.007 [-0.219, 0.205], p 0.9437, n = 545; positive calibration = False.

## S3 (§6): low-tier anchors, Group A vs B

Status: confirmatory (Group A raters with 240/240: rater02, rater04, rater15; included: rater02, rater03, rater04, rater15).
glmer correct ~ group + (1|rater) + (1|item): group effect -0.132 [-0.588, 0.323], one-sided p 0.7157. Pooled A 0.564 [0.479, 0.645] (75/133); B 0.593 [0.532, 0.651] (153/258).

## S4 (§6): Bradley-Terry vs tournament ordering

545 judgments, 10 strategies, 45 of 45 pairings judged. Spearman rho -0.188, one-sided exact p 0.7082 over 3628800 orderings (uncorrected tournament ordering: rho -0.188, p 0.7082).

| strategy | BT log-strength | tournament mean P(win), corrected | uncorrected |
|---|---|---|---|
| gourdeau_disc | 0.824 | 0.410 | 0.409 |
| cql_enr_a2.0 | 0.521 | 0.423 | 0.424 |
| cql_naive_a1.0 | 0.515 | 0.408 | 0.406 |
| gd | 0.399 | 0.401 | 0.401 |
| constrained_mcts | -0.029 | 0.651 | 0.650 |
| constrained_greedy | -0.109 | 0.581 | 0.578 |
| mcts | -0.294 | 0.655 | 0.655 |
| gourdeau | -0.382 | 0.535 | 0.541 |
| enriched | -0.539 | 0.583 | 0.583 |
| mcq_t0.5 | -0.907 | 0.353 | 0.352 |

## Per rater (descriptive)

| rater | anchors | low | mid | high | H1 agreement (0.02) |
|---|---|---|---|---|---|
| rater02 | 0.620 [0.534, 0.699] (80/129) | 0.674 [0.525, 0.795] (29/43) | 0.581 [0.433, 0.716] (25/43) | 0.605 [0.456, 0.736] (26/43) | 0.618 [0.486, 0.735] (34/55) |
| rater03 | 0.500 [0.237, 0.763] (5/10) | 0.250 [0.046, 0.699] (1/4) | 0.750 [0.301, 0.954] (3/4) | 0.500 [0.095, 0.905] (1/2) | 0.600 [0.231, 0.882] (3/5) |
| rater04 | 0.473 [0.389, 0.559] (61/129) | 0.558 [0.411, 0.696] (24/43) | 0.419 [0.284, 0.567] (18/43) | 0.442 [0.304, 0.589] (19/43) | 0.623 [0.488, 0.741] (33/53) |
| rater07 | 0.527 [0.441, 0.611] (68/129) | 0.512 [0.368, 0.654] (22/43) | 0.581 [0.433, 0.716] (25/43) | 0.488 [0.346, 0.632] (21/43) | 0.491 [0.364, 0.619] (27/55) |
| rater09 | 0.535 [0.449, 0.619] (69/129) | 0.698 [0.549, 0.814] (30/43) | 0.442 [0.304, 0.589] (19/43) | 0.465 [0.325, 0.611] (20/43) | 0.472 [0.344, 0.603] (25/53) |
| rater10 | 0.574 [0.487, 0.656] (74/129) | 0.605 [0.456, 0.736] (26/43) | 0.535 [0.389, 0.675] (23/43) | 0.581 [0.433, 0.716] (25/43) | 0.462 [0.333, 0.595] (24/52) |
| rater11 | 0.504 [0.419, 0.589] (65/129) | 0.512 [0.368, 0.654] (22/43) | 0.465 [0.325, 0.611] (20/43) | 0.535 [0.389, 0.675] (23/43) | 0.396 [0.276, 0.531] (21/53) |
| rater12 | 0.550 [0.464, 0.634] (71/129) | 0.628 [0.479, 0.756] (27/43) | 0.442 [0.304, 0.589] (19/43) | 0.581 [0.433, 0.716] (25/43) | 0.393 [0.276, 0.524] (22/56) |
| rater14 | 0.512 [0.426, 0.596] (66/129) | 0.605 [0.456, 0.736] (26/43) | 0.419 [0.284, 0.567] (18/43) | 0.512 [0.368, 0.654] (22/43) | 0.446 [0.324, 0.576] (25/56) |
| rater15 | 0.512 [0.426, 0.596] (66/129) | 0.488 [0.346, 0.632] (21/43) | 0.558 [0.411, 0.696] (24/43) | 0.488 [0.346, 0.632] (21/43) | 0.600 [0.468, 0.719] (33/55) |
