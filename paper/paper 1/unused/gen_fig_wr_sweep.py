"""Generate WR-sweep line plot (Figure 4)."""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.rcParams['font.size'] = 9

wr_labels = ['No aug', '0%', '5%', '10%', '50%']
wr_x = [0, 1, 2, 3, 4]

healer = [86.5, 91.3, 95.0, 95.0, 94.8]
degen = [19.8, 16.3, 12.6, 12.3, 8.1]
sanity = [23, 25, 25, 25, 24]
sanity_pct = [s / 28 * 100 for s in sanity]

fig, ax1 = plt.subplots(figsize=(5, 3.5))

ax1.plot(wr_x, healer, 'o-', color='#4CAF50', label='Healer %', linewidth=1.5, markersize=5)
ax1.plot(wr_x, degen, 's-', color='#F44336', label='Degen %', linewidth=1.5, markersize=5)
ax1.plot(wr_x, sanity_pct, '^--', color='#2196F3', label='Sanity %', linewidth=1.2, markersize=5)

ax1.set_xlabel('Assigned Win Rate', fontsize=10)
ax1.set_ylabel('Rate (%)', fontsize=10)
ax1.set_xticks(wr_x)
ax1.set_xticklabels(wr_labels)
ax1.set_ylim(0, 105)
ax1.legend(loc='center left', fontsize=8)
ax1.set_title('Synthetic Augmentation WR Sweep', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig("/home/max/heroes-of-the-storm/paper/cql_followup/fig_wr_sweep.png",
            dpi=200, bbox_inches='tight')
print("Saved fig_wr_sweep.png")
