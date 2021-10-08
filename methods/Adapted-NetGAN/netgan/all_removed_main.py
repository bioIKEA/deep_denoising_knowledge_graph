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
    # [graph_matrix] = load_variable(fold_path + adj_matrx_graph_filename)
    [test_dataset_indexs] = load_variable(fold_path + test_index_filename)

    # print("test data set index", test_dataste_indexs)

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

    
#    for (x, y, val) in test_dataset_indexs[258:]:
#        if val == 1:
#            ones.append([x, y])
#        else:
#            zeros.append([x, y])
#
#    print("ones = ", len(ones), "zeros = ", len(zeros))
#    exit(-3)

    tmp_file_name = "tmp_file.pkl"
    if os.path.exists(tmp_file_name):
        [ones, zeros, train_pos, other_edges] = load_variable(tmp_file_name)
    else:
        for (x, y, val) in test_dataset_indexs[258:]:
            if val == 1:
                ones.append([x, y])
            else:
                zeros.append([x, y])

        test_neg = random.Random(seed).sample(zeros, test_neg_size)
        test_pos = random.Random(seed).sample(ones, test_neg_size)

        # fitler the train graph with the labeled gene-chem
        gene_chem_ones = []
        gene_chem_zeros= []

        for (x, y, val) in test_dataset_indexs[:258]:
            if val == 1:
                gene_chem_ones.append([x, y])
            else:
                gene_chem_zeros.append([x, y])
        print("gene_chem ones = ", len(gene_chem_ones), "gene_chem zeros = ", len(gene_chem_zeros))
        print("size of test neg = ", len(test_neg), test_neg[:3])

        print("sparse graph", sp_graph.nnz)
        up_sp_graph = sp.tril(sp_graph)
        print("upper triganular", up_sp_graph.nnz)
        print("non zero:\n", up_sp_graph.nonzero())
        for i, j in zip(*up_sp_graph.nonzero()):
            if [j, i] in test_neg:
                continue
            elif [j, i] in test_pos:
                continue
            elif [j ,i] in gene_chem_zeros:
                continue
            elif [j, i] in ones or [j, i] in gene_chem_ones:
                train_pos.append([j, i])
            else:
                other_edges.append([j, i])

        print("other edges size =", len(other_edges), other_edges[:3])
        print("train ones = ", len(train_pos), train_pos[:3])
        save_to_file([ones, zeros, train_pos, other_edges], tmp_file_name)

    test_ones = random.Random(seed).sample(ones, test_neg_size)
    test_zeros = random.Random(seed).sample(zeros, test_neg_size)
    
    test_node_set = set()
    for (i, j) in test_ones:
        test_node_set.add(i)
        test_node_set.add(j)
    for (i, j) in test_zeros:
        test_node_set.add(i)
        test_node_set.add(j)
    #for (i, j) in train_pos:
    #    test_node_set.add(i)
    #    test_node_set.add(j)
    print("valid node size ===== ", len(test_node_set))
    # build train graph
    G = nx.Graph()
    # first part train positive
    for (i, j) in train_pos:
        G.add_edge(i, j)
        G.add_edge(j, i)
        _weight = int(adj_matrix_graph.item(i,j))
        G[i][j]['weight'] = _weight
        G[j][i]['weight'] = _weight
    
    # other edges 
    cnt_other = 0
    for (i, j) in other_edges:
        _weight = int(adj_matrix_graph.item(i,j))
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

    print("use other edges size = ", cnt_other)

    # filter_idx, train_ones, test_ones, test_zeros = utils.train_test_split_adjacenty(_size, test_neg_size, test_dataste_indexs) 

    train_graph = G
    # print("train graph size ====", G.number_of_nodes())


    # train_graph = sp.coo_matrix((np.ones(len(train_ones)),(train_ones[:,0], train_ones[:,1]))).tocsr()
    boolean_mask = [0] * _size
    for i in range(_size):
        if i in test_node_set:
            boolean_mask[i] = 1

    boolean_mask = [1] * _size
    
    test_ones = np.array(test_ones)
    test_zeros = np.array(test_zeros)
    test_ones = symmetrize(test_ones)
    test_zeros = symmetrize(test_zeros)

    rw_len = 16
    batch_size = 128
    filename = 'alias_node_edges.pkl'
    walker = utils.RandomWalker(train_graph, rw_len, list(test_node_set),filename, p=100, q=1, batch_size=batch_size)

    netgan = NetGAN(_size, rw_len, boolean_mask, walk_generator=walker.walk, gpu_id=0, use_gumbel=True, disc_iters=3,
                    W_down_discriminator_size=128, W_down_generator_size=128,
                    l2_penalty_generator=1e-7, l2_penalty_discriminator=5e-5,
                    generator_layers=[40], discriminator_layers=[30], temp_start=5, learning_rate=0.0003)

    #log_dict = netgan.train_then_test(A_orig=sp_graph, test_ones=test_ones, test_zeros=test_zeros, max_iters=10, eval_transitions=15e2,transitions_per_iter=1500)
    netgan.train_then_test(A_orig=sp_graph, test_ones=test_ones, test_zeros=test_zeros, max_iters=1000,eval_transitions=15e6, transitions_per_iter=150000)
