import matplotlib.pyplot as plt
import numpy as np


labels = ['All', 'Gene-Chem', 'Gene-Dise', 'Chem-Dise']
AP_CELL_means = [0.577, 0.573, 0.676, 0.768]
AP_NETGAN_means = [0.5, 0.577, 0.614, 0.505]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, AP_CELL_means, width, label='CELL')
rects2 = ax.bar(x + width/2, AP_NETGAN_means, width, label='NetGAN')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Average Precision Score', fontsize=10)
ax.set_title('Average Precision Score with different types of edge', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=10)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)

fig.tight_layout()

plt.show()
