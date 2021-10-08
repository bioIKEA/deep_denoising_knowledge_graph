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

def load_variable(filename):
    with open(filename, 'rb') as f:
        var_list = pickle.load(f)
    return var_list

def save_to_file(var_list, filename):
    with open(filename, 'wb') as f:
        pickle.dump(var_list, f)

if __name__ == '__main__':
    
    symmetrize = lambda x: np.row_stack((x, np.column_stack((x[:, 1], x[:, 0]))))
    fold_path = '../data/covid_19/'
    list_dict_filename = 'node_list_dict.pkl'
    adj_matrx_graph_filename = "adj_matrix.pkl"
    test_index_filename = 'test_dataset/test_idx.pkl'
    sp_matrix_filename = 'covid_19_sp_connected.npz'


    node_list = []
    node_dict = dict()
    test_dataste_indexs = []
    graph_matrix = None

    [node_list, node_dict] = load_variable(fold_path + list_dict_filename)
    [test_dataset_indexs] = load_variable(fold_path + test_index_filename)


    sp_graph = sp.load_npz(fold_path + sp_matrix_filename)
    adj_matrix_graph = sp_graph.todense()

    # access weight example
    # print("access non value", adj_matrix_graph.item(0,0))
    # for i,j,v in test_dataste_indexs[:10]:
    #     print("i, j v", i, j, v)
    #     print(adj_matrix_graph.item(i,j))
    #     print(adj_matrix_graph.item(j,i))
    #     print("--")

    _size = adj_matrix_graph.shape[0]
    print("size = ", _size)


    # No validation dataset
    test_neg_size = 37 
    
    # graph_matrix = sp_graph.todense()
    ones = []
    zeros = []
    # MAX DEGREE ID = MDI
    MDI = 14878
    seed = 1111
    train_pos = []
    other_edges = []
    
    gene_chem_cnt = 0
    gene_dise_cnt = 0
    chem_dise_cnt = 0  

    tmp_file_name = "tmp_file.pkl"

    test_data_idx = test_dataset_indexs

    test_parts = [0, 1, 2, 3, 4]
    for test_part in test_parts:
        test_ones = []
        test_zeros = []
        if test_part == 0:
            for (i, j, val) in test_data_idx:
                if val == 1:
                    test_ones.append([i, j])
                else:
                    test_zeros.append([i, j])
            print("size of one = ", len(test_ones), len(test_zeros))
        elif test_part == 1:
            for (i, j, val) in test_data_idx[:258]:
                if val == 1:
                    test_ones.append([i, j])
                else:
                    test_zeros.append([i, j])
        elif test_part == 2:
            # for gen-dise only
            for (i, j, val) in test_data_idx[258:358]:
                if val == 1:
                    test_ones.append([i,j])
                else:
                    test_zeros.append([i,j])
        elif test_part == 3:
            # chem-dise only
            for (i, j, val) in test_data_idx[358:]:
                if val == 1:
                    test_ones.append([i, j])
                else:
                    test_zeros.append([i, j])
        elif test_part == 4:
            # gen-dise + chem-dise
            for (i, j, val) in test_data_idx[258:]:
                if val == 1:
                    test_ones.append([i, j])
                else:
                    test_zeros.append([i, j])

        test_size = min(len(test_ones), len(test_zeros))
        print("test size = ", test_size)
        if len(test_ones) != test_size:
            test_ones = random.Random(seed).sample(test_ones, test_size)
        if len(test_zeros) != test_size:
            test_zeros = random.Random(seed).sample(test_zeros, test_size)

        # test node set used for connect the test path to the whole train graph
        test_node_set = set()
        for (i, j) in test_ones:
            test_node_set.add(i)
            test_node_set.add(j)
        for (i, j) in test_zeros:
            test_node_set.add(i)
            test_node_set.add(j)
        print("valid node size ===== ", len(test_node_set))

        # build train graph
        G = nx.Graph()
        up_sp_graph = sp.trilu(sp_graph)
        for i, j in zip(*up_sp_graph.nonzero()):
            G.add_edge(i,j)
            G.add_edge(j,i)
            _weight = int(adj_matrix_graph.item(i,j))
            G[i][j]['weight'] = _weight
            G[j][i]['weight'] = _weight
            if i == MDI or j == MDI:
                cnt_other += 1
                _weight = 10
                G.add_edge(i, j)
                G.add_edge(j, i)
                G[i][j]['weight'] = _weight
                G[j][i]['weight'] = _weight
            elif i in test_node_set or j in test_node_set:
                _weight = 1
                G.add_edge(i, j)
                G.add_edge(j, i)
                G[i][j]['weight'] = _weight
                G[j][i]['weight'] = _weight

        # filter_idx, train_ones, test_ones, test_zeros = utils.train_test_split_adjacenty(_size, test_neg_size, test_dataste_indexs)

        train_graph = G
        boolean_mask = [1] * _size
        test_ones = np.array(test_ones)
        test_zeros = np.array(test_zeros)
        test_ones = symmetrize(test_ones)
        test_zeros = symmetrize(test_zeros)

        rw_len = 16
        batch_size = 128
        filename = 'alias_node_edges.pkl'
        walker = utils.RandomWalker(train_graph, rw_len,filename, p=100, q=1, batch_size=batch_size)

        netgan = NetGAN(_size, rw_len, boolean_mask, walk_generator=walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
                        W_down_discriminator_size=128, W_down_generator_size=128,
                        l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                        generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=0.0003)

        #log_dict = netgan.train_then_test(A_orig=sp_graph, test_ones=test_ones, test_zeros=test_zeros, max_iters=10, eval_transitions=15e2,transitions_per_iter=1500)
        log_dict = netgan.train_then_test(A_orig=sp_graph, test_ones=test_ones, test_zeros=test_zeros, max_iters=1000,eval_transitions=15e6, transitions_per_iter=150000)
        res_file = "netgan_real_dataset_test_part_" + str(test_part)+"_roc_curve.pkl"
        utils.save_to_file([log_dict["actual_label"], log_dict["edge_score"]], res_file)
        del netgan
