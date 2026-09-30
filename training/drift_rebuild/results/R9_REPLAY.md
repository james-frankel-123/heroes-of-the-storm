# R9 deployment replay extensions

Oracle weighted acc 56.811. Checks: {'8->11@10': (56.693, 56.693), '20->25@24': (56.467, 56.467)}

| policy | retrains | refreshes | weighted acc | regret (pp) |
|---|---|---|---|---|
| oracle (retrain every build) | 35 | 35 | 56.811 | +0.000 |
| never retrain, never refresh (same C0 model) | 0 | 0 | 55.748 | +1.062 |
| never retrain, refresh every build | 0 | 35 | 56.080 | +0.730 |
| never retrain, blind_every3 refresh | 0 | 11 | 56.073 | +0.738 |
| never retrain, blind_even14 refresh | 0 | 14 | 56.077 | +0.733 |
| retrain every 3, refresh every build | 11 | 35 | 56.782 | +0.029 |
| retrain every 3, no refresh between retrains | 11 | 11 | 56.760 | +0.050 |
| retrain every 6, refresh every build | 5 | 35 | 56.743 | +0.067 |
| retrain every 6, no refresh between retrains | 5 | 5 | 56.705 | +0.106 |
| frozen recipe, leak-free (OOF) s42 | 0 | 0 | 55.692 | +1.118 |
| frozen recipe, leak-free (OOF) s123 | 0 | 0 | 55.763 | +1.048 |
| frozen recipe, leak-free (OOF) s777 | 0 | 0 | 55.732 | +1.079 |
| frozen recipe, leaky (paper) s42 | 0 | 0 | 55.128 | +1.682 |
