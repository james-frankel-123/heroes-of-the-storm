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
| 0.001 | 17 | 2 | 716 | 0.895 | 0.023 | 0.045 | 9 | 56.048 | +0.763 |
| 0.005 | 22 | 3 | 711 | 0.88 | 0.03 | 0.058 | 11 | 56.060 | +0.751 |
| 0.01 | 22 | 5 | 711 | 0.815 | 0.03 | 0.058 | 11 | 56.060 | +0.751 |
| 0.02 | 29 | 6 | 704 | 0.829 | 0.04 | 0.076 | 12 | 56.057 | +0.754 |
| 0.05 **(paper)** | 46 | 12 | 687 | 0.793 | 0.063 | 0.116 | 14 | 56.068 | +0.743 |
| 0.1 | 54 | 23 | 679 | 0.701 | 0.074 | 0.133 | 16 | 56.068 | +0.743 |
| 0.2 | 63 | 32 | 670 | 0.663 | 0.086 | 0.152 | 18 | 56.069 | +0.742 |
| 0.35 | 78 | 70 | 655 | 0.527 | 0.106 | 0.177 | 18 | 56.071 | +0.740 |
| 0.5 | 116 | 140 | 617 | 0.453 | 0.158 | 0.235 | 19 | 56.074 | +0.737 |

Reference rows (same fixed C0 model): never refresh 0 refreshes / 55.748% / +1.063 pp; refresh every build 35 refreshes / 56.080% / +0.730 pp; oracle (retrain every build) 56.811%.

## Verification

- q=0.05 detected sets, P/R (0.793 / 0.063), fire schedule (14 fires, per-fire latencies) and replay row (56.068 / +0.743) all reproduce the stored W3_CHANGEPOINTS / W2C_DETECTOR_ARM numbers exactly; the replayed cumprev accuracies reproduce the stored W2c matrix row to < 0.01 pp.

## Reading

- P/R tradeoff: precision falls from 0.895 at q=0.001 to 0.453 at q=0.5 while recall rises from 0.023 to 0.158 — the detector is precision-limited by design (patch notes are a lower bound; see W3_CHANGEPOINTS caveats).
- Regret frontier: every operating point sits between never-refresh (+1.063) and every-build refresh (+0.730); the best swept point is q=0.5 at +0.737 pp with 19 refreshes (paper point: q=0.05, +0.743 pp at 14 refreshes).
