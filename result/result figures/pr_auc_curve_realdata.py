import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import pickle
def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list

x = np.arange(0.0, 1.1, 0.1)
y = x
true_y, pred_y, = load_variable("netgan_real_dataset_roc_curve.pkl")
true_y_1, pred_y_1 = load_variable("cell_realdata_roc_curve.pkl")


precsion_netgan, recall_netgan, _ = precision_recall_curve(true_y, pred_y)
base_fpr_1, base_tpr_1, _ = precision_recall_curve(true_y_1, pred_y_1)
b1, b2, _ = precision_recall_curve(true_y, true_y)

plt.plot(precsion_netgan, recall_netgan, label='A-NetGAN')
plt.plot(base_fpr_1, base_tpr_1, label='CELL')
# plt.plot(x, y, linestyle='--', label='random classification')
# plt.plot(b1, b2, label='perfect classification')

plt.title('PR curve', fontsize = 16)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

plt.show()
