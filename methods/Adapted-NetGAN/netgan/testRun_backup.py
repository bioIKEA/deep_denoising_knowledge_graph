"""
1. run netgan on train dataset by genrating graph with GAN
2. based on generator, pass the test edge indexes (grouth truth: label by human) to get the predicted edge is
"""

from netgan import *
import tensorflow as tf
import utils
import scipy.sparse as sp
import numpy as np
import os
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import pickle
import networkx as nx
import random
from sklearn.model_selection import KFold

def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list


if __name__ == '__main__':
    fold_path = '../data/'
    matrix_filename = 'cora_ml.npz'

    _A_obs, _X_obs, _z_obs = utils.load_npz(fold_path + matrix_filename)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    _N = _A_obs.shape[0]

    labeled_ratio = 0.1
    train_ratio = labeled_ratio
    noise_ratio = 0.1
    seed = 1111

    file_name = "saved_edges_with_traion_ratio_with_seed_%s.pkl" % seed

    if not os.path.isfile(file_name):
        utils.synthetic_dataset_split(_A_obs, labeled_ratio, train_ratio, noise_ratio, utils.AddNoise.REMOVE,seed, file_name)
    [labeled_ones, unknown_ones, sampling_zeros] = load_variable(file_name)

    sampling_zeros = sampling_zeros.tolist()
    labeled_ones = np.array(labeled_ones)
    print("size of labeled ones", len(labeled_ones), type(labeled_ones))
    print("size of unknown ones", len(unknown_ones), type(unknown_ones))
    print("size of sampling zeros", len(sampling_zeros), type(sampling_zeros))


    """
    10-fold on labled ones with fixed negative edges 
    """
    kf = KFold(n_splits=10)
    symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))

    test_zeros = random.Random(seed).sample(sampling_zeros, int(len(labeled_ones) * (1 - train_ratio)))
    test_zeros = np.array(test_zeros)
    test_zeros = symmetrize(test_zeros)
    not_test_zeros = []
    print("size of negative test", len(test_zeros))

    for (i, j) in sampling_zeros:
        if (i, j) in test_zeros:
            continue
        not_test_zeros.append((i,j))

    fold_iteration = 1
    ROC_list = []
    F1_list = []
    Precision_list = []
    Recall_list = []
    for train_index, test_index in kf.split(labeled_ones):

        test_ones = labeled_ones[train_index]
        train_ones = labeled_ones[test_index]

        filter_set = set()
        for (i, j) in train_ones:
            filter_set.add(i)
            filter_set.add(j)
        filter = sorted(list(filter_set))

        print("size of filter = ", len(filter_set))

        boolean_mask = [0] * _N
        for i in range(_N):
            if i in filter_set:
                boolean_mask[i] = 1

        train_ones = train_ones.tolist()

        test_ones = np.array(test_ones)
        unknown_ones = np.array(unknown_ones)
        train_ones = np.array(train_ones)

        test_ones = symmetrize(test_ones)
        unknown_ones = symmetrize(unknown_ones)
        train_ones = symmetrize(train_ones)

        """
        construct train graph
        """
        G = nx.Graph()
        for (i, j) in train_ones:
            G.add_edge(i, j)
            G[i][j]['weight'] = _A_obs[i, j]

        for (i, j) in unknown_ones:
            G.add_edge(i, j)
            G[i][j]['weight'] = 0.6

        train_graph = G
        rw_len = 16
        batch_size = 128
        filename = 'cora_ml_node_edge_transition_%s.pkl'%fold_iteration
        walk_p = 0.1
        walk_q = 10
        walker = utils.RandomWalker(train_graph, rw_len, filename, p=walk_p, q=walk_q, batch_size=batch_size)

        netgan = NetGAN(_N, rw_len, boolean_mask, walk_generator=walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
                        W_down_discriminator_size=128, W_down_generator_size=128,
                        l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                        generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=0.0003)

        # log_dict = netgan.train_then_test(A_orig=_A_obs, test_ones=test_ones, test_zeros=test_zeros, max_iters=10, eval_transitions=15e2,
        #                     transitions_per_iter=1500)
        log_dict = netgan.train_then_test(A_orig=_A_obs, test_ones=test_ones, test_zeros=test_zeros, max_iters=1000,
                               eval_transitions=15e6, transitions_per_iter=150000)
        fold_iteration += 1
        tf.reset_default_graph()

        ROC_list.append(log_dict["ROC"])
        F1_list.append(log_dict["F1"])
        Precision_list.append(log_dict["Precision"])
        Recall_list.append(log_dict["Recall"])
        os.remove(filename)
        print("---- finished ----", fold_iteration, "with current value = ", log_dict)

    print("Average ROC = ", sum(ROC_list)/len(ROC_list), \
          "Average F1 = ", sum(F1_list)/len(F1_list),\
          "Average Precision = ", sum(Precision_list) / len(Precision_list),\
          "Average Recall = ", sum(Recall_list) / len(Recall_list))
