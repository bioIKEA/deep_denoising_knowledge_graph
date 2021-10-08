"""
1. run netgan on train dataset by genrating graph with GAN
2. based on generator, pass the test edge indexes (grouth truth: label by human) to get the predicted edge is
"""

from netgan import *
import tensorflow as tf
import netgan_utils
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
import sys

def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list

def run_experiment(case_number, labeled_ratio, noise_ratio, weight_decay, walk_p=0.1, walk_q=10, baseline=False, with_mask=False, seed=1111, epoch_times=3):
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

    _A_obs, _X_obs, _z_obs = netgan_utils.load_npz(fold_path + matrix_filename)
    _A_obs = _A_obs + _A_obs.T
    _A_obs[_A_obs > 1] = 1
    lcc = netgan_utils.largest_connected_components(_A_obs)
    _A_obs = _A_obs[lcc, :][:, lcc]
    _N = _A_obs.shape[0]


    file_name = "generate_edges_with_label_ratio_%s_noise_ratio_%s_with_seed_%s.pkl" % (str(labeled_ratio), str(noise_ratio), seed)

    if not os.path.isfile(file_name):
        netgan_utils.synthetic_dataset_split(_A_obs, labeled_ratio, file_name)
    [min_span_tree, other_ones, sampling_zeros] = load_variable(file_name)

    sampling_zeros = sampling_zeros.tolist()
    print("min tree is", len(min_span_tree))
    print("size of labeled ones except min spanning tree", len(other_ones))
    print("size of sampling zeros", len(sampling_zeros))

    total_edges = len(other_ones) + len(min_span_tree)
    print("ttoal edgves", total_edges)
    accessible_ones = int(total_edges * labeled_ratio)
    print("accessible ones", accessible_ones)
    train_ones_number = int(accessible_ones * labeled_ratio)
    print("train ones", train_ones_number)
    ratio = int((train_ones_number - len(min_span_tree)) / len(sampling_zeros))
    k_fold = ratio + 1
    print(" k_fold is", k_fold)
    sample_size = k_fold * len(sampling_zeros)
    labeled_ones = random.Random(seed).sample(other_ones.tolist(),sample_size)
    print("labled ones", len(labeled_ones))

    # k-fold on labeled ones with fixed negative edges
    kf = KFold(n_splits=k_fold)
    symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))

    test_zeros = sampling_zeros
    test_zeros = np.array(test_zeros)
    test_zeros = symmetrize(test_zeros)


    ROC_list = []
    F1_list = []
    Precision_list = []
    Recall_list = []
    Accuracy_list = []
    epoch_time_idx = 0
    res_file = "res-case-%s.txt" % str(case_number)
    if os.path.exists(res_file):
        os.remove(res_file)
    labeled_ones = np.array(labeled_ones)
    for train_index, test_index in kf.split(labeled_ones):
        print(" start evaluation of subfold with idx = ", epoch_time_idx)
        if epoch_time_idx >= epoch_times:
            # earlier stop
            print(" finish evlauation with total subfold = ", epoch_times)
            break

        test_ones = labeled_ones[test_index]
        train_ones = labeled_ones[train_index]

        print("train one ========",len(train_ones))
        noise_ones = random.Random(seed).sample(train_ones.tolist(), int(train_ones_number * noise_ratio))
        print("noise ones", len(noise_ones), type(noise_ones))
        # print("noise one", noise_ones)

        train_ones_without_noise = []
        for (i, j) in train_ones:
            if [i, j] in noise_ones:
                continue
            train_ones_without_noise.append((i,j))
        print("size of train witout noise", len(train_ones_without_noise))
        filter_set = set()
        # for (i, j) in train_ones:
        #     filter_set.add(i)
        #     filter_set.add(j)
        boolean_mask = [1] * _N
        # if not with_mask:
        #     boolean_mask = [1] * _N
        # else:
        #     for i in range(_N):
        #         if i in filter_set:
        #             boolean_mask[i] = 1

        test_ones = np.array(test_ones)
        train_ones = np.array(train_ones)
        noise_ones = np.array(noise_ones)
        train_ones_without_noise = np.array(train_ones_without_noise)

        test_ones = symmetrize(test_ones)
        train_ones = symmetrize(train_ones)
        noise_ones = symmetrize(noise_ones)
        train_ones_without_noise = symmetrize(train_ones_without_noise)

        fix_min_tree_ones = symmetrize(min_span_tree)

        """
        construct train graph
        """
        G = nx.Graph()
        for (i, j) in fix_min_tree_ones:
            G.add_edge(i,j)
            G[i][j]['weight'] = _A_obs[i, j]
        for (i, j) in train_ones_without_noise:
            G.add_edge(i, j)
            G[i][j]['weight'] = _A_obs[i, j]

        for (i, j) in noise_ones:
            G.add_edge(i, j)
            if baseline:
                G[i][j]['weight'] = _A_obs[i, j]
            else:
                G[i][j]['weight'] = weight_decay
        rw_len = 16
        batch_size = 128
        filename = 'cora_ml_node_edge_transition_%s.pkl'%epoch_time_idx

        if os.path.exists(filename):
            os.remove(filename)

        G = nx.to_scipy_sparse_matrix(G)

        # walker = utils.RandomWalker(G, rw_len, filename, p=walk_p, q=walk_q, batch_size=batch_size)
        walker = netgan_utils.Netgan_RandomWalker(G, rw_len, p=walk_p, q=walk_q, batch_size=batch_size)

        netgan = NetGAN(_N, rw_len, boolean_mask, walk_generator=walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
                        W_down_discriminator_size=128, W_down_generator_size=128,
                        l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                        generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=0.0003)

        # log_dict = netgan.train_then_test(A_orig=_A_obs, test_ones=test_ones, test_zeros=test_zeros, max_iters=10, eval_transitions=15e2,
        #                     transitions_per_iter=1500)
        log_dict = netgan.train_then_test(A_orig=_A_obs, test_ones=test_ones, test_zeros=test_zeros, max_iters=1000,
                               eval_transitions=15e6, transitions_per_iter=150000)
        epoch_time_idx += 1
        tf.reset_default_graph()
        ROC_list.append(log_dict["ROC"])
        F1_list.append(log_dict["F1"])
        Precision_list.append(log_dict["Precision"])
        Recall_list.append(log_dict["Recall"])
        Accuracy_list.append(log_dict["Accuracy"])
        if os.path.exists(filename):
            os.remove(filename)
        print("---- finished ----", epoch_time_idx, "with current value = ", log_dict)
        if os.path.exists(res_file):
            append_write = 'a'
        else:
            append_write = 'w'
        f = open(res_file, append_write)
        f.write("current ROC =  {0}\n current F1 =  {1}\n current Precision =  {2}\n current Recall = {3}\n accuracy = {4}\n".format(log_dict["ROC"], log_dict["F1"], \
                                                log_dict["Precision"], log_dict["Recall"], log_dict["Accuracy"]))

    f.write("Average ROC =  {0}\n Average F1 =  {1} \n Average Precision =  {2} \n Average Recall = {3} \n Average Accuracy {4}\n".format(sum(ROC_list)/len(ROC_list), sum(F1_list)/len(F1_list), \
                                                sum(Precision_list) / len(Precision_list),sum(Recall_list) / len(Recall_list), sum(Accuracy_list) / len(Accuracy_list), \
                                                                                                                                          sum(Accuracy_list) / len(Accuracy_list)))
    f.close()
    print("Average ROC = ", sum(ROC_list)/len(ROC_list), \
          "Average F1 = ", sum(F1_list)/len(F1_list),\
          "Average Precision = ", sum(Precision_list) / len(Precision_list),\
          "Average Recall = ", sum(Recall_list) / len(Recall_list),\
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

    # EXP1: labeld_raio/noise_ration
    # run_experiment(case_number, k_fold, labeled_ratio, noise_ratio, weight_decay, walk_p=0.1, walk_q=10, baseline=False, with_mask=False, seed=1111, epoch_times=3)

    # baseline with_mask=False, baseline=True
    run_experiment(0, 9.0/10, 0.05, 0, walk_p=1, walk_q=1, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    # run_experiment(1, 9.0/10, 0.05, 0, walk_p=1, walk_q=1, baseline=False, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(2, 8.0/9, 0.05, 0, walk_p=1, walk_q=1, baseline=False, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(3, 7.0/8, 0.05, 0, walk_p=1, walk_q=1, baseline=False, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(4, 6.0/7, 0.05, 0, walk_p=1, walk_q=1, baseline=False, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(5, 5.0/6, 0.05, 0, walk_p=1, walk_q=1, baseline=False, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(6, 4.0/5, 0.05, 0, walk_p=1, walk_q=1, baseline=False, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(7, 3.0/4, 0.05, 0, walk_p=1, walk_q=1, baseline=False, with_mask=False, seed=1111, epoch_times=6)

    # run_experiment(1, 3.0/4, 0.05, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(2, 10, 1.0/10, 0.1, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    # run_experiment(3, 10, 1.0/10, 0.2, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    #
    # run_experiment(4, 6, 1.0/6, 0.05, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    # run_experiment(5, 6, 1.0/6, 0.1, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    # run_experiment(6, 6, 1.0/6, 0.2, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    #
    # run_experiment(7, 4, 1.0/4, 0.05, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    # run_experiment(8, 4, 1.0/4, 0.1, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    # run_experiment(9, 4, 1.0/4, 0.2, 0.8, walk_p=0.1, walk_q=10, baseline=True, with_mask=False, seed=1111, epoch_times=3)
    #
    # # test case with_mask = True
    # run_experiment(11, 10, 1.0/10, 0.05, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(12, 10, 1.0/10, 0.1, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(13, 10, 1.0/10, 0.2, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    #
    # run_experiment(14, 6, 1.0/6, 0.05, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(15, 6, 1.0/6, 0.1, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(16, 6, 1.0/6, 0.2, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    #
    # run_experiment(17, 4, 1.0/4, 0.05, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(18, 4, 1.0/4, 0.1, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(19, 4, 1.0/4, 0.2, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)


    # EXP 2 test weight decay
    # run_experiment(case_number, k_fold, labeled_ratio, noise_ratio, weight_decay, walk_p=0.1, walk_q=10, baseline=False, with_mask=False, seed=1111, epoch_times=3)
    # run_experiment(0, 9.0/10, 0.05, 0.8, walk_p=1, walk_q=1, baseline=True, with_mask=False, seed=1111, epoch_times=6)
    # run_experiment(21, 6.0/7, 0.1, 0.2, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(22, 6.0/7, 0.1, 0.4, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(23, 6.0/7, 0.1, 0.6, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(24, 6.0/7, 0.1, 0.8, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(25, 6.0/7, 0.1, 1, walk_p=0.1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    #
    # # test walk_p
    # run_experiment(31, 6, 1.0/6, 0.1, 0.8, walk_p=0.01, walk_q=1, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(32, 6, 1.0/6, 0.1, 0.8, walk_p=0.1, walk_q=1, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(33, 6, 1.0/6, 0.1, 0.8, walk_p=1, walk_q=1, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(34, 6, 1.0/6, 0.1, 0.8, walk_p=10, walk_q=1, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(35, 6, 1.0/6, 0.1, 0.8, walk_p=100, walk_q=1, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    #
    # # test walk_q
    # run_experiment(41, 6, 1.0/6, 0.1, 0.8, walk_p=1, walk_q=0.1, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(42, 6, 1.0/6, 0.1, 0.8, walk_p=1, walk_q=0.01, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(43, 6, 1.0/6, 0.1, 0.8, walk_p=1, walk_q=10, baseline=False, with_mask=True, seed=1111, epoch_times=3)
    # run_experiment(44, 6, 1.0/6, 0.1, 0.8, walk_p=1, walk_q=100, baseline=False, with_mask=True, seed=1111, epoch_times=3)
