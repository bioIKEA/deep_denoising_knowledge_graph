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
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import pickle
import networkx as nx
import random
from sklearn.model_selection import KFold
import sys


def run_experiment(res_in_one_file, case_number, labeled_ratio, train_ratio, noise_ratio, weight_decay, walk_p=0.1, walk_q=10, mode="debug", baseline=False, seed=1111, epoch_times=3):
    """
    case_number: int, for identify experiment
    k_fold: 10, 6, 4
    labeled_ratio: 1/10, 1/6, 1/4
    noise_ratio: 0.05, 0.1, 0.2
    weight_decay (noise level): 0.2, 0.4, 0.6, 0.8, 1
    walk_p (sampling parameter): 0.01, 0.1, 1, 10, 100
    walk_q (sampling parameter): 0.01, 0.1, 1, 10, 100
    baseline (train graph construction control): True or False,
    with_mask (cost function mask filter control): True or False
    seed : for split the dataset
    epoch_times (value range: 1-5): average the res of sub fold,  default=3
    """
    fold_path = '../data/'
    matrix_filename = 'cora_ml.npz'
    # symmetrize function
    symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))

    _A_obs, _X_obs, _z_obs = utils.load_npz(fold_path + matrix_filename)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    _N = _A_obs.shape[0]

    A = sp.tril(_A_obs).tocsr()
    A.eliminate_zeros()
    # labeled_ratio = 0.7
    _E, _N = A.nnz, A.shape[0]
    # graph with node =  2810  edge =  7981
    print(" Here we assume our accessible edges is half")
    _E  = int(_E * labeled_ratio)
    # train ratio = positive number in train vs positive number in test
    # in fact, test have 1:1 positive number plus negative number
    _size_train = int(_E * train_ratio)
    _size_test = _E - _size_train
    print("graph with node = ", _N, " edge = ", _E, " with train ratio = ", train_ratio)
    print("size of train = ", _size_train, " size of test ", _size_test)

    # step 1: sampling enough negative edges for later use
    negative_edges_file_name = "sampling_enough_negative_edges_from_%s" % (matrix_filename)
    _negative_sampling_total = _E*10
    sampling_zeros = []
    if os.path.exists(negative_edges_file_name):
        [sampling_zeros] = utils.load_variable(negative_edges_file_name)
    else:
        while len(sampling_zeros) < _negative_sampling_total:
            r, c = np.random.randint(0, _N, 2)
            if A.todense().item(r, c) == 0 and r > c and (r, c) not in sampling_zeros:
                sampling_zeros.append((r, c))
            utils.save_tofile([sampling_zeros], negative_edges_file_name)

    A_hold = minimum_spanning_tree(A)
    A_sample = A - A_hold
    x, y = A_hold.nonzero()
    min_span_tree = np.column_stack((x, y))
    i, j = A_sample.nonzero()
    other_ones = np.column_stack((i, j))
    print(" min span tree size = ", len(min_span_tree), "other edges size = ", len(other_ones))
    print(" total sampled negative edges = ", len(sampling_zeros), "with top 5 is: ", sampling_zeros[:5])

    # other_ones has two parts, one is our accessible ones, another part is the unlabeled ones
    accessible_ones = []
    if _E >= len(min_span_tree):
        accessible_ones = random.Random(seed).sample(other_ones.tolist(), _E - len(min_span_tree))
    unlabeled_ones = []
    for (i,j) in other_ones.tolist():
        if [i,j] in accessible_ones:
            continue
        unlabeled_ones.append((i,j))

    assert len(unlabeled_ones) + len(accessible_ones) == len(other_ones)

    test_zeros = []
    other_zeros_for_unlabeled = []
    print(" test size", _size_test)
    test_zeros = random.Random(seed).sample(sampling_zeros, _size_test)
    print(" test zeros", test_zeros[:4], "sampling zero", sampling_zeros[:3])
    for [i, j] in sampling_zeros:
        # print("i j is", i, j)
        if (i,j) in test_zeros:
            continue
        other_zeros_for_unlabeled.append((i,j))

    print(" len of samplineg ", len(sampling_zeros), "len of test", len(test_zeros), " len of other", len(other_zeros_for_unlabeled))
    assert len(other_zeros_for_unlabeled) + len(test_zeros) == len(sampling_zeros)

    test_zeros = np.array(test_zeros)
    test_zeros = symmetrize(test_zeros)

    mode_name = ""
    if baseline:
        mode_name = "baseline"
    else:
        mode_name = "non-baseline"

    exp_prefix = "%s_train_ratio_%s_noise_ratio_%s_with_%s.txt" %(mode_name, str(train_ratio), str(noise_ratio), case_number)
    # res_file = "%s-res_case.txt" % str(case_number)

    # if noise_ratio > 0:
    #     res_file = "noise_ratio_"+str(noise_ratio) + "_" + res_file
    ROC_list = []
    F1_list = []
    Precision_list = []
    Recall_list = []
    Accuracy_list = []
    boolean_mask = [1] * _N
    true_labels_list = []
    pred_labels_list = []

    while epoch_times:

        # step 1: build train graph with test ones
        G = nx.Graph()
        if _size_train >= len(min_span_tree):

            fix_train_ones = np.array(min_span_tree)
            # print(" size of fix_train", len(fix_train_ones)
            test_ones = random.Random().sample(accessible_ones, _size_test)
            # print(" test = ", len(test_ones), len(accessible_ones))
            train_ones = []
            noise_ones = []
            for (i, j) in accessible_ones:
                if [i,j] in test_ones:
                    continue
                train_ones.append((i,j))

            if noise_ratio > 0:
                removed_size = int(len(unlabeled_ones) * noise_ratio)
                noise_ones = random.Random().sample(unlabeled_ones, len(unlabeled_ones) - removed_size)
            # print(" only left====", len(train_ones))
            # print(" total train ones = ", len(train_ones) + len(fix_train_ones)//2)
            train_ones = np.array(train_ones)
            train_ones = symmetrize(train_ones)
            fix_train_ones = symmetrize(fix_train_ones)
            for (i,j) in train_ones:
                G.add_edge(i,j)
                G[i][j]['weight'] = _A_obs[i, j]

            for (i, j) in fix_train_ones:
                G.add_edge(i, j)
                G[i][j]['weight'] = _A_obs[i, j]

            if not baseline:
                noise_ones = np.array(noise_ones)
                noise_ones = symmetrize(noise_ones)
                for (i, j) in noise_ones:
                    G.add_edge(i, j)
                    G[i][j]['weight'] = weight_decay


        else:# _size_train < len(min_span_tree) noise_ratio is compared with the unlabeled size
            fix_train_ones = random.Random().sample(min_span_tree.tolist(), _size_train)
            test_ones = random.Random().sample(other_ones.tolist(), _size_test)
            print(" test ones", len(test_ones))

            complement_train_ones = []
            for (i, j) in min_span_tree.tolist():
                if [i, j] in fix_train_ones:
                    continue
                complement_train_ones.append((i,j))

            assert len(fix_train_ones) + len(complement_train_ones) == 2809

            noise_ones = []
            noise_zeros = []
            for (i, j) in other_ones.tolist():
                if [i, j] in test_ones:
                    continue
                noise_ones.append((i,j))

            assert len(noise_ones) + len(test_ones) == len(other_ones)
            if noise_ratio > 0:
                _total = len(noise_ones) / (1 - noise_ratio)
                noise_zeros_size = int(_total * noise_ratio)
                noise_zeros = random.Random(seed).sample(other_zeros_for_unlabeled, noise_zeros_size)
                print(" unlabeled edges with positive = ", len(noise_ones), "negative = ", len(noise_zeros))
            # print(" fix = ", len(fix_train_ones), "total with complement = ", len(fix_train_ones)+len(complement_train_ones))
            fix_train_ones = np.array(fix_train_ones)
            complement_train_ones = np.array(complement_train_ones)
            noise_ones = np.array(noise_ones)
            noise_zeros = np.array(noise_zeros)

            fix_train_ones = symmetrize(fix_train_ones)
            complement_train_ones = symmetrize(complement_train_ones)
            noise_ones = symmetrize(noise_ones)
            if len(noise_zeros)>=1:
                noise_zeros = symmetrize(noise_zeros)

            for (i,j) in fix_train_ones:
                G.add_edge(i, j)
                G[i][j]['weight'] = _A_obs[i, j]

            if not baseline:
                for (i, j) in complement_train_ones:
                    G.add_edge(i, j)
                    G[i][j]['weight'] = weight_decay
                for (i,j) in noise_ones:
                    G.add_edge(i, j)
                    G[i][j]['weight'] = weight_decay
                for (i,j) in noise_zeros:
                    G.add_edge(i, j)
                    G[i][j]['weight'] = weight_decay

        test_ones = np.array(test_ones)
        test_ones = symmetrize(test_ones)

        # step 2: start train network
        rw_len = 16
        batch_size = 128
        filename = 'cora_ml_node_edge_transition_%s.pkl' % str(1)

        if os.path.exists(filename):
            os.remove(filename)

        walker = utils.RandomWalker(G, rw_len, filename, p=walk_p, q=walk_q, batch_size=batch_size)

        netgan = NetGAN(_N, rw_len, boolean_mask, walk_generator=walker.walk, gpu_id=0, use_gumbel=True,
                        disc_iters=3,
                        W_down_discriminator_size=128, W_down_generator_size=128,
                        l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                        generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=0.0003)
        log_dict = None
        if mode == "debug":
            log_dict = netgan.train_then_test(A_orig=_A_obs, test_ones=test_ones, test_zeros=test_zeros, max_iters=10, eval_transitions=15e2,
                            transitions_per_iter=1500)
        else:
            log_dict = netgan.train_then_test(A_orig=_A_obs, test_ones=test_ones, test_zeros=test_zeros, max_iters=1000,
                                          eval_transitions=15e6, transitions_per_iter=150000)
        tf.compat.v1.reset_default_graph()
        ROC_list.append(log_dict["ROC"])
        F1_list.append(log_dict["F1"])
        Precision_list.append(log_dict["Precision"])
        Recall_list.append(log_dict["Recall"])
        Accuracy_list.append(log_dict["Accuracy"])
        true_labels_list.append(log_dict["actual_label"])
        pred_labels_list.append(log_dict["edge_score"]) 
        if os.path.exists(filename):
            os.remove(filename)
        print("with current value = ", log_dict)
        if os.path.exists(res_in_one_file):
            append_write = 'a'
        else:
            append_write = 'w'
        f = open(res_in_one_file, append_write)
        f.write("current ROC =  {0}\n current F1 =  {1}\n current Precision =  {2}\n current Recall = {3}\n accuracy = {4}\n true label = {5} \n pred = {6}\n".format(
                log_dict["ROC"], log_dict["F1"], log_dict["Precision"], log_dict["Recall"], log_dict["Accuracy"], log_dict["actual_label"], log_dict["edge_score"]))
        epoch_times -= 1
        utils.save_tofile([log_dict["actual_label"], log_dict["edge_score"]], "netgan_baseline_roc_curve.pkl")

    print(" ---- program finished ----")
    f.write("\n-------" + exp_prefix + "-------\n")
    f.write("Average ROC =  {0}\n Average F1 =  {1} \n Average Precision =  {2} \n Average Recall = {3} \n".format(
        sum(ROC_list) / len(ROC_list), sum(F1_list) / len(F1_list), \
        sum(Precision_list) / len(Precision_list), sum(Recall_list) / len(Recall_list),
        sum(Accuracy_list) / len(Accuracy_list)))
    f.close()
    print("Average ROC = ", sum(ROC_list) / len(ROC_list), \
          "Average F1 = ", sum(F1_list) / len(F1_list), \
          "Average Precision = ", sum(Precision_list) / len(Precision_list), \
          "Average Recall = ", sum(Recall_list) / len(Recall_list), \
          "Average Accuray = ", sum(Accuracy_list) / len(Accuracy_list))

if __name__ == '__main__':
    """
    case_number: int, for identify experiment
    k_fold: 10, 6, 4
    labeled_ratio: 1/10, 1/6, 1/4
    noise_ratio: 0.05, 0.1, 0.2
    weight_decay (noise level): 0.2, 0.4, 0.6, 0.8, 1
    walk_p (sampling parameter): 0.01, 0.1, 1, 10, 100
    walk_q (sampling parameter): 0.01, 0.1, 1, 10, 100
    baseline (train graph construction control): True or False,
    with_mask (cost function mask filter control): True or False
    seed : for split the dataset
    epoch_time (value range: 1-5): average the res of sub fold,  defualt=5
    """

    # run_experiment(case_number, train_ratio, noise_ratio, weight_decay, walk_p=0.1, walk_q=10, baseline=False,
    #                seed=1111, epoch_times=3)

    epoch_times = 3
    seed = 1111
    # train_ratio_list = [0.4, 0.3, 0.2, 0.1]
    train_ratio_list = [0.5]
    walk_p = 1
    walk_q = 1
    weight_decay = 1
    case_number = 1
    labeled_ratio = 0.5
    noise_ratio = 0

    res_in_one_file = "task_1_labeled_ratio_%s_baseline.txt" %(str(labeled_ratio))
    for train_ratio in train_ratio_list:
        tmp_name = "with_train_ratio_%s" % str(train_ratio)
        run_experiment(res_in_one_file, tmp_name + "_" + str(case_number), labeled_ratio, train_ratio, noise_ratio, weight_decay, walk_p=walk_p, walk_q=walk_q,
                       mode="non-debug", baseline=True, seed=seed, epoch_times=1)
        case_number += 1

