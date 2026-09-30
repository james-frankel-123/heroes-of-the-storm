# R8 judge-free head-to-head scores (W12 protocol, T_clean)

Paired (a minus b) deltas, crossed seed random effects; t reference with Satterthwaite df.

| file | a | b | drafts | hero WR pp | synergy | counter | degen (a-b) | degen a / b |
|---|---|---|---|---|---|---|---|---|
| M_vs_Mcut | drift2026:w4_d2c_cumprev,w8_d2c_cumprev:15 | rebuild:r6_mcut123_rg:5 | 6000 | -0.053 ± 0.108 (z -0.5, df 9) | +0.009 ± 0.122 (z 0.1, df 8) | -0.150 ± 0.166 (z -0.9, df 5) | +10.583 ± 8.258 (z 1.3, df 8) | 34.9 / 24.4 |
| M_vs_S2 | drift2026:w4_d2c_cumprev,w8_d2c_cumprev:15 | rebuild:r6_s2oof_era:6 | 7200 | +1.401 ± 0.088 (z 15.9, df 13) | +0.208 ± 0.075 (z 2.8, df 11) | -0.078 ± 0.099 (z -0.8, df 9) | +1.514 ± 7.559 (z 0.2, df 13) | 27.2 / 25.7 |
| M_vs_U | drift2026:w4_d2c_cumprev,w8_d2c_cumprev:15 | rebuild:r6_uoof_rg:6 | 7200 | +0.431 ± 0.131 (z 3.3, df 10) | +0.141 ± 0.114 (z 1.2, df 7) | +0.124 ± 0.072 (z 1.7, df 12) | -5.583 ± 7.630 (z -0.7, df 11) | 24.1 / 29.7 |
| Mcut_vs_U | rebuild:r6_mcut123_rg:5 | rebuild:r6_uoof_rg:6 | 2400 | +0.519 ± 0.134 (z 3.9, df 10) | +0.247 ± 0.154 (z 1.6, df 10) | +0.259 ± 0.099 (z 2.6, df 7) | -14.333 ± 8.477 (z -1.7, df 9) | 17.3 / 31.7 |
| Md90_vs_U | drift2026:w8_champion:15 | rebuild:r6_uoof_rg:6 | 7200 | +0.766 ± 0.109 (z 7.0, df 8) | +0.092 ± 0.136 (z 0.7, df 8) | +0.063 ± 0.087 (z 0.7, df 10) | +1.681 ± 7.241 (z 0.2, df 8) | 25.2 / 23.5 |
| Mref_vs_S2 | rebuild:r6_mref_rg:6 | rebuild:r6_s2oof_era:6 | 2880 | +1.283 ± 0.148 (z 8.7, df 9) | +0.209 ± 0.068 (z 3.1, df 12) | -0.138 ± 0.141 (z -1.0, df 11) | +14.062 ± 13.152 (z 1.1, df 7) | 37.4 / 23.3 |
| Mref_vs_U | rebuild:r6_mref_rg:6 | rebuild:r6_uoof_rg:6 | 2880 | +0.344 ± 0.169 (z 2.0, df 10) | -0.070 ± 0.158 (z -0.4, df 8) | -0.013 ± 0.103 (z -0.1, df 8) | -1.840 ± 11.496 (z -0.2, df 8) | 28.8 / 30.7 |
| Mvol_vs_S2 | drift2026:w8_volmatch:5 | rebuild:r6_s2oof_era:6 | 2400 | +1.277 ± 0.158 (z 8.1, df 7) | +0.170 ± 0.162 (z 1.0, df 5) | +0.097 ± 0.129 (z 0.8, df 9) | -8.750 ± 10.171 (z -0.9, df 9) | 19.2 / 28.0 |
| U_vs_S2 | rebuild:r6_uoof_rg:6 | rebuild:r6_s2oof_era:6 | 2880 | +1.003 ± 0.160 (z 6.3, df 8) | +0.273 ± 0.124 (z 2.2, df 10) | -0.053 ± 0.109 (z -0.5, df 10) | +11.701 ± 9.422 (z 1.2, df 10) | 43.9 / 32.2 |
| U_vs_Uleaky | rebuild:r6_uoof_rg:6 | drift2026:w4_d2b_allhist,w8_d2b_allhist:15 | 7200 | +0.350 ± 0.125 (z 2.8, df 11) | +0.094 ± 0.139 (z 0.7, df 10) | -0.389 ± 0.090 (z -4.3, df 13) | -21.542 ± 7.274 (z -3.0, df 16) | 28.3 / 49.9 |
| repro_w6 | drift2026:w4_d2c_cumprev:5 | drift2026:w4_d2b_allhist:5 | 2000 | +0.658 ± 0.154 (z 4.3, df 5) | +0.226 ± 0.120 (z 1.9, df 9) | -0.248 ± 0.140 (z -1.8, df 8) | -20.800 ± 11.412 (z -1.8, df 8) | 14.5 / 35.3 |
| paper_M_vs_U | d2c_cumprev | d2b_allhist | 2000 | +0.658 ± 0.154 (z 4.3, df 5) | +0.226 ± 0.120 (z 1.9, df 9) | -0.248 ± 0.140 (z -1.8, df 8) | -20.800 ± 11.412 (z -1.8, df 8) | 14.5 / 35.3 |
| paper_M_vs_S1 | d2c_cumprev | stale1yr | 2000 | +0.559 ± 0.208 (z 2.7, df 7) | +0.251 ± 0.143 (z 1.8, df 6) | -0.134 ± 0.130 (z -1.0, df 9) | -8.450 ± 5.966 (z -1.4, df 11) | 20.0 / 28.4 |
| paper_M_vs_S2 | d2c_cumprev | stale2yr | 2000 | +1.096 ± 0.168 (z 6.5, df 8) | +0.071 ± 0.159 (z 0.4, df 9) | -0.557 ± 0.125 (z -4.4, df 9) | -7.900 ± 7.377 (z -1.1, df 11) | 19.7 / 27.6 |
| paper_M_vs_U_15x15 | d2c_cumprev | d2b_allhist | 4500 | +0.686 ± 0.098 (z 7.0, df 32) | +0.177 ± 0.072 (z 2.5, df 31) | -0.308 ± 0.081 (z -3.8, df 26) | -14.956 ± 6.716 (z -2.2, df 30) | 22.0 / 36.9 |
