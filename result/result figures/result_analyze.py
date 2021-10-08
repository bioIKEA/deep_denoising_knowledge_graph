import numpy as np
import matplotlib.pyplot as plt

base_AUC_NetGAN = [0.686,0.654, 0.633, 0.604, 0.597, 0.595, 0.590, 0.574, 0.577]
base_AUC_CELL = [0.72
,0.656
,0.569
,0.591
,0.591
,0.578
,0.566
,0.594
,0.590
]
non_base_AUC_NetGAN = [0.752
,0.730
,0.742
,0.754
,0.724
,0.705
,0.673
,0.610
,0.614
]
non_base_AUC_CELL = [0.859
,0.856
,0.852
,0.828
,0.828
,0.810
,0.797
,0.777
,0.763
]

base_AP_NetGAN = [0.603
,0.570
,0.567
,0.553
,0.566
,0.571
,0.595
,0.614
,0.620
]
base_AP_CELL = [0.739
,0.686
,0.595
,0.616
,0.610
,0.603
,0.595
,0.618
,0.606
]

non_base_AP_NetGAN = [0.627
,0.598
,0.574
,0.573
,0.570
,0.570
,0.561
,0.543
,0.541
]
non_base_AP_CELL = [0.885
,0.883
,0.876
,0.856
,0.854
,0.834
,0.825
,0.802
,0.790
]

def plot_AUC(x, y_lists, y_label):
    fig, ax = plt.subplots()
    line1, = ax.plot(x, y_lists[0], 'tab:orange', linestyle='dashed', label='Baseline NetGAN')
    line2, = ax.plot(x, y_lists[1], 'tab:blue', linestyle='dashed', label='Baseline CELL')
    line3, = ax.plot(x, y_lists[2], 'tab:orange', label='NetGAN')
    line4, = ax.plot(x, y_lists[3], 'tab:blue', label='CELL')
    ax.legend()
    plt.ylim(0.5, 1)
    plt.xlabel('Train Ratio', fontsize=10)
    plt.ylabel(y_label, fontsize=10)
    plt.show()

x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
y1_lists = [base_AUC_NetGAN, base_AUC_CELL, non_base_AUC_NetGAN, non_base_AUC_CELL]
y2_lists = [base_AP_NetGAN, base_AP_CELL, non_base_AP_NetGAN, non_base_AP_CELL]
# Using set_dashes() to modify dashing of an existing line
#plot_AUC(x, y1_lists, 'AUC ROC Score')
plot_AUC(x, y2_lists, 'Average Precision')
