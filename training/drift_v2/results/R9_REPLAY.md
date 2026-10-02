# R9 deployment replay extensions

Oracle weighted acc 56.866. Checks: {'8->11@10': (56.541, 56.541), '20->25@24': (56.328, 56.328)}

| policy | retrains | refreshes | weighted acc | regret (pp) |
|---|---|---|---|---|
| oracle (retrain every build) | 35 | 35 | 56.866 | +0.000 |
| never retrain, never refresh (same C0 model) | 0 | 0 | 55.845 | +1.021 |
| never retrain, refresh every build | 0 | 35 | 56.204 | +0.662 |
| never retrain, blind_every3 refresh | 0 | 11 | 56.181 | +0.685 |
| never retrain, blind_even14 refresh | 0 | 14 | 56.190 | +0.675 |
| retrain every 3, refresh every build | 11 | 35 | 56.868 | -0.002 |
| retrain every 3, no refresh between retrains | 11 | 11 | 56.838 | +0.028 |
| retrain every 6, refresh every build | 5 | 35 | 56.785 | +0.081 |
| retrain every 6, no refresh between retrains | 5 | 5 | 56.720 | +0.146 |
| frozen recipe, leak-free (OOF) s42 | 0 | 0 | 55.795 | +1.071 |
| frozen recipe, leak-free (OOF) s123 | 0 | 0 | 55.856 | +1.010 |
| frozen recipe, leak-free (OOF) s777 | 0 | 0 | 55.520 | +1.346 |
| frozen recipe, leaky (paper) s42 | 0 | 0 | 55.168 | +1.698 |
