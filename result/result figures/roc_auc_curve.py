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


print("ture is", len(true_y), len(true_y_1), len(true_y_2), len(true_y_3))
base_fpr, base_tpr, _ = roc_curve(true_y, pred_y)
base_fpr_1, base_tpr_1, _ = roc_curve(true_y_1, pred_y_1)
non_base_fpr, non_base_tpr, _ = roc_curve(true_y_2, pred_y_2)
non_base_fpr_1, non_base_tpr_1, _ = roc_curve(true_y_3, pred_y_3)
b1, b2, _ = roc_curve(true_y, true_y)

plt.plot(base_fpr, base_tpr, label='base netgan')
plt.plot(base_fpr_1, base_tpr_1, label='base cell')
plt.plot(non_base_fpr, non_base_tpr, label='non-base netgan')
plt.plot(non_base_fpr_1, non_base_tpr_1, label='non-base cell')
plt.plot(x, y, linestyle='--', label='random classification')
plt.plot(b1, b2, label='perfect classification')

plt.title('ROC curve', fontsize = 16)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()

plt.show()
