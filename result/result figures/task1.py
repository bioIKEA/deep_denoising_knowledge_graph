import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import pickle
def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list

x = np.arange(0.0, 1.1, 0.1)
y = x
true_y, pred_y, = load_variable("netgan_baseline_roc_curve.pkl")
true_y_1, pred_y_1 = load_variable("cell_baseline_roc_curve.pkl")
true_y_2, pred_y_2 = load_variable("netgan_non_baseline_roc_curve.pkl")
true_y_3, pred_y_3 = load_variable("cell_non_baseline_roc_curve.pkl")

base_fpr, base_tpr, _ = roc_curve(true_y, pred_y)
base_fpr_1, base_tpr_1, _ = roc_curve(true_y_1, pred_y_1)
non_base_fpr, non_base_tpr, _ = roc_curve(true_y_2, pred_y_2)
non_base_fpr_1, non_base_tpr_1, _ = roc_curve(true_y_3, pred_y_3)
b1, b2, _ = roc_curve(true_y, true_y)


plt.subplot(121)
plt.plot(base_fpr, base_tpr,  'tab:orange', linestyle='dashed', label='Base NetGAN')
plt.plot(base_fpr_1, base_tpr_1,  'tab:blue',  linestyle='dashed', label='Base CELL')
plt.plot(non_base_fpr, non_base_tpr, 'tab:orange', label='NetGAN')
plt.plot(non_base_fpr_1, non_base_tpr_1, 'tab:blue', label='CELL')
#plt.plot(x, y, linestyle='--', label='random classification')
#plt.plot(b1, b2, label='perfect classification')

plt.title('ROC curve with AR = 0.5', fontsize=12)
plt.xlabel('False Positive Rate', fontsize=10)
plt.ylabel('True Positive Rate', fontsize=10)
plt.legend()

plt.subplot(122)

base_AUC_NetGAN = [0.686, 0.654, 0.633, 0.604, 0.597, 0.595, 0.590, 0.574, 0.577]
base_AUC_CELL = [0.72, 0.656, 0.569, 0.591, 0.591, 0.578, 0.566, 0.594, 0.590]
non_base_AUC_NetGAN = [0.752, 0.730, 0.742, 0.754, 0.724, 0.705, 0.673, 0.610, 0.614]
non_base_AUC_CELL = [0.859, 0.856, 0.852, 0.828, 0.828, 0.810, 0.797, 0.777, 0.763]
x = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
y1_lists = [base_AUC_NetGAN, base_AUC_CELL, non_base_AUC_NetGAN, non_base_AUC_CELL]

plt.plot(x, base_AUC_NetGAN, 'tab:orange', linestyle='dashed', label = 'Base NetGAN')
plt.plot(x, base_AUC_CELL, 'tab:blue', linestyle='dashed', label='Base CELL')
plt.plot(x, non_base_AUC_NetGAN, 'tab:orange', label='NetGAN')
plt.plot(x, non_base_AUC_CELL, 'tab:blue', label='CELL')
# Using set_dashes() to modify dashing of an existing line
plt.ylim(0.5, 1)
plt.xlabel('Annotatoin Ratio (AR)', fontsize=10)
plt.ylabel('AUC ROC Score', fontsize=10)
plt.title('AUC ROC Score with different AR', fontsize=12)

plt.legend()

plt.show()

