# Detector operating curves (Q11)

The W3(b) outcome-shift detector's decision rule — per-hero two-proportion WR
z-tests (>= 200 games/side) at each sizable-build boundary, Benjamini-Hochberg
FDR within boundary — swept over the FDR level q. At each operating point:
precision/recall vs the official-patch-note ground truth (micro-averaged over
the 27 validatable boundaries, W3_CHANGEPOINTS protocol), and the W2c
detector-triggered deployment replay (fixed C0 model w2c_cut08_s42, refresh
fires with honest per-boundary first-detection z_naive latency; W2C_DETECTOR_ARM
protocol). The paper's operating point is q=0.05. Figure: fig_detector_curves.pdf.

| FDR q | TP | FP | FN | precision | recall | F1 | refreshes | weighted acc % | regret vs oracle (pp) |
|---|---|---|---|---|---|---|---|---|---|
| 0.001 | 17 | 2 | 716 | 0.895 | 0.023 | 0.045 | 9 | 56.156 | +0.710 |
| 0.005 | 21 | 3 | 712 | 0.875 | 0.029 | 0.055 | 11 | 56.172 | +0.694 |
| 0.01 | 23 | 4 | 710 | 0.852 | 0.031 | 0.061 | 11 | 56.172 | +0.694 |
| 0.02 | 28 | 6 | 705 | 0.824 | 0.038 | 0.073 | 12 | 56.176 | +0.690 |
| 0.05 **(paper)** | 46 | 13 | 687 | 0.78 | 0.063 | 0.116 | 14 | 56.189 | +0.677 |
| 0.1 | 54 | 23 | 679 | 0.701 | 0.074 | 0.133 | 16 | 56.193 | +0.673 |
| 0.2 | 63 | 34 | 670 | 0.649 | 0.086 | 0.152 | 18 | 56.199 | +0.667 |
| 0.35 | 84 | 73 | 649 | 0.535 | 0.115 | 0.189 | 18 | 56.199 | +0.667 |
| 0.5 | 115 | 146 | 618 | 0.441 | 0.157 | 0.231 | 19 | 56.202 | +0.664 |

Reference rows (same fixed C0 model): never refresh 0 refreshes / 55.845% / +1.021 pp; refresh every build 35 refreshes / 56.204% / +0.662 pp; oracle (retrain every build) 56.866%.

## Verification

- q=0.05 detected sets, P/R (0.793 / 0.063), fire schedule (14 fires, per-fire latencies) and replay row (56.068 / +0.743) all reproduce the stored W3_CHANGEPOINTS / W2C_DETECTOR_ARM numbers exactly; the replayed cumprev accuracies reproduce the stored W2c matrix row to < 0.01 pp.

## Reading

- P/R tradeoff: precision falls from 0.895 at q=0.001 to 0.441 at q=0.5 while recall rises from 0.023 to 0.157 — the detector is precision-limited by design (patch notes are a lower bound; see W3_CHANGEPOINTS caveats).
- Regret frontier: every operating point sits between never-refresh (+1.021) and every-build refresh (+0.662); the best swept point is q=0.5 at +0.664 pp with 19 refreshes (paper point: q=0.05, +0.677 pp at 14 refreshes).
